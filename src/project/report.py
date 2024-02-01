# type: ignore

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Any

from loguru import logger

from .environment import RUNNING_IN_CLUSTER, RUNNING_IN_PYTEST


def _deep_defaultdict():
    """A recursively defined defaultdict.

    Returns:
        defaultdict: Returns defaultdict that recursively creates dictionaries.
    """
    return defaultdict(_deep_defaultdict)


class _Reporter:
    """The `_Reporter` class helps track configuration, parameters, metrics, and other
    information while running the project.
    """

    def __init__(self):
        """Initializes the `_Reporter` class."""
        self._config = _deep_defaultdict()
        self.series_files = {}
        self.wandb_run = None
        if RUNNING_IN_CLUSTER and not RUNNING_IN_PYTEST:
            self._setup_wandb()

    def _setup_wandb(self):
        key = os.environ.get("WANDB_API_KEY", "").strip()
        if key != "" and key != "your_api_key":
            if not self.wandb_run:
                import wandb

                if len(sys.argv) > 1:
                    task_name = sys.argv[1]
                    tags = [task_name]
                else:
                    task_name = ""
                    tags = []

                self.wandb_run = wandb.init(
                    project=os.environ["PROJECT_NAME"],
                    tags=tags,
                    name=os.environ["PROJECT_JOB_NAME"] + " / " + task_name,
                )
            return True
        else:
            return False

    def _select(self, path):
        """Given a dot-style path to an object in the config, resolve the final
        dictionary housing the object along with the key for the object.

        Args:
            path (str): The dot-style path to an object in the config.

        Returns:
            tuple[dict[str, Any], str]: The final dictionary housing the object and the
                key of the object in that dictionary.
        """
        components = path.split(".")
        d = self._config
        for c in components[:-1]:
            d = d[c]
        return d, components[-1]

    def set(self, path, value):
        """Given a dot-style path to an object in the config, stores the object at that
        path.

        Args:
            path (str): The dot-style path to an object in the config.
            value (Any): The object to store at the path.

        Returns:
            Any: The object stored at the path.
        """
        d, key = self._select(path)
        d[key] = value
        if RUNNING_IN_CLUSTER and not RUNNING_IN_PYTEST:
            with open(os.environ["PROJECT_CONFIG_FILE"], "w+", encoding="utf-8") as f:
                f.truncate(0)
                f.write(json.dumps(self._config, indent=2))
                f.flush()
                if self._setup_wandb():
                    import wandb

                    wandb.config.update(self._config)
        return value

    def get(self, path):
        """Given a dot-style path to an object in the config, returns the object at that
        path.

        Args:
            path (str): The dot-style path to an object in the config.

        Returns:
            Any: The object returned from the path.
        """
        d, key = self._select(path)
        return d[key]

    def series(
        self,
        series,
        data,
        dir=None,
        headers=True,
        date_columns=True,
        delimiter=",",
        ext="csv",
    ):
        """Stores a series of data as a CSV file. Each function call will write one
        row of data to the CSV. The name of the series is given by `series` and the
        data to write at each row is given by `data`.

        Args:
            series (str): The name of the series.
            data (dict[str, Any]): A dictionary of one row of data. Each key is a
                column. The dictionary can have any schema.
            dir (str, optional): The path to where to write the CSV file. Defaults to
                None.
            headers (bool, optional): Whether to output headers. Defaults to True.
            date_columns (bool, optional): Whether to output date columns. Defaults to
                True.
            delimiter (str, optional): The delimiter for the CSV file. Defaults to
                ",".
            ext (str, optional): The extension for the CSV file. Defaults to
                "csv".
        """
        if not RUNNING_IN_CLUSTER or RUNNING_IN_PYTEST:
            return
        from filelock import FileLock

        lock = FileLock(
            os.path.join(
                os.path.dirname(tempfile.mkdtemp()),
                f"{os.environ['PROJECT_NAME']}-series-{series}.lock",
            )
        )
        with lock:
            _data: OrderedDict[str, Any] = OrderedDict()
            if date_columns:
                now = datetime.now()
                _data["date"] = now.strftime("%Y-%-m-%d-%H:%M:%S")
                _data["timestamp"] = int(time.time())
            _data.update(data)
            if series not in self.series_files:
                csv_fn = f"{series}.{ext}"
                csv_path = os.path.join(os.environ["PROJECT_SERIES_DIR"], csv_fn)
                if dir:
                    csv_custom_path = os.path.join(dir, csv_fn)
                    needs_headers = not os.path.exists(csv_custom_path)
                    csv_file = open(csv_custom_path, "a+", encoding="utf-8")
                    try:
                        os.symlink(csv_custom_path, csv_path)
                    except FileExistsError:
                        os.remove(csv_path)
                        os.symlink(csv_custom_path, csv_path)
                else:
                    needs_headers = not os.path.exists(csv_path)
                    csv_file = open(csv_path, "a+", encoding="utf-8")
                csv_writer = csv.DictWriter(csv_file, _data.keys(), delimiter=delimiter)
                if headers and needs_headers:
                    csv_writer.writeheader()
                self.series_files[series] = csv_file
            csv_writer = csv.DictWriter(
                self.series_files[series], _data.keys(), delimiter=delimiter
            )
            csv_writer.writerow(_data)
            if self._setup_wandb():
                import wandb

                wandb.log(_data)
            self.series_files[series].flush()

    def upload(self, src_path, tgt=None, service="wandb", type="dataset"):
        """Upload files (datasets or models) to a service.

        Args:
            src_path (str): The source path of the file to upload.
            tgt (str): The target location to upload the file (set to None for "wandb").
            service (str): The service to upload to ("wandb", "drive", and "s3").
            type (str): The type of file being uploaded (either "dataset" or "model").
        """
        src_path = os.path.abspath(src_path)
        src_path = os.path.normpath(src_path)
        if tgt:
            logger.debug(f"Uploading '{src_path}' to '{service}' at '{tgt}'...")
        else:
            logger.debug(f"Uploading '{src_path}' to '{service}'...")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"{src_path} does not exist.")
        is_dir = os.path.isdir(src_path)
        src_path_dir = os.path.dirname(src_path)
        basename = os.path.basename(src_path)

        if service == "drive":
            if is_dir:
                zip_path = os.path.join(src_path_dir, basename + ".zip")
                logger.info(f"Zipping '{src_path}' to '{zip_path}'...")
                if os.path.exists(zip_path):
                    raise FileExistsError(f"The zip file '{zip_path}' already exists")
                subprocess.run(["zip", "-r", zip_path, src_path], check=True)
                subprocess.run(["gupload", "--to", tgt, zip_path], check=True)
                logger.info(f"Deleting zip file '{zip_path}'...")
                os.remove(zip_path)
                logger.info(f"Deleted zip file '{zip_path}'.")
            else:
                subprocess.run(["gupload", "--to", tgt, src_path], check=True)
        elif service == "s3":
            if is_dir:
                subprocess.run(
                    [
                        "awsv2",
                        "s3",
                        "cp",
                        "--recursive",
                        src_path,
                        tgt + "/" + basename,
                    ],
                    check=True,
                )
            else:
                subprocess.run(["awsv2", "s3", "cp", src_path, tgt], check=True)

        elif tgt is None:
            if self._setup_wandb():
                import wandb

                artifact = wandb.Artifact(basename, type=type)
                if is_dir:
                    artifact.add_dir(src_path)
                else:
                    artifact.add_file(src_path)
                if self.wandb_run:
                    self.wandb_run.log_artifact(artifact)


# Instantiate a reporter
reporter = _Reporter()
