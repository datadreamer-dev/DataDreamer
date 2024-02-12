import json
import logging
import os
import shutil
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from functools import cached_property, partial
from importlib import metadata
from io import BytesIO
from logging import Logger
from time import time
from typing import Any, Callable, DefaultDict, Sequence, cast

import dill
from datasets import Dataset, DatasetDict
from datasets.fingerprint import Hasher
from filelock import FileLock, Timeout
from pandas import DataFrame

from .. import __version__, logging as datadreamer_logging
from .._cachable import _Cachable
from ..datadreamer import DataDreamer
from ..datasets import (
    OutputDataset,
    OutputDatasetColumn,
    OutputIterableDataset,
    OutputIterableDatasetColumn,
)
from ..errors import StepOutputError
from ..logging import DATEFMT, logger
from ..pickling import unpickle as _unpickle
from ..pickling.pickle import _INTERNAL_PICKLE_KEY, _pickle
from ..project.environment import RUNNING_IN_PYTEST
from ..utils.arg_utils import DEFAULT, Default
from ..utils.background_utils import run_in_background_process_no_block
from ..utils.collection_utils import uniq_str
from ..utils.fingerprint_utils import stable_fingerprint
from ..utils.fs_utils import move_dir, safe_fn
from ..utils.hf_hub_utils import get_readme_contents, hf_hub_login, prepare_to_publish
from ..utils.time_utils import progress_eta
from .data_card import DataCardType, sort_data_card
from .step_background import wait
from .step_export import _path_to_split_paths, _unpickle_export
from .step_operations import (
    _INTERNAL_STEP_OPERATION_KEY,
    _INTERNAL_STEP_OPERATION_NO_SAVE_KEY,
    __concatenate,
    _create_add_item_step,
    _create_copy_step,
    _create_filter_step,
    _create_map_step,
    _create_remove_columns_step,
    _create_rename_column_step,
    _create_rename_columns_step,
    _create_reverse_step,
    _create_save_step,
    _create_select_columns_step,
    _create_select_step,
    _create_shard_step,
    _create_shuffle_step,
    _create_skip_step,
    _create_sort_step,
    _create_splits_step,
    _create_take_step,
    _step_to_dataset_dict,
)
from .step_output import (
    LazyRowBatches,
    LazyRows,
    StepOutputType,
    _monkey_patch_iterable_dataset_apply_feature_types,
    _output_to_dataset,
)

_INTERNAL_HELP_KEY = "__DataDreamer__help__"
_INTERNAL_TEST_KEY = "__DataDreamer__test__"


class StepMeta(type):
    has_base = False

    def __new__(meta, name, bases, attrs):
        if meta.has_base:
            for attribute in attrs:
                is_data_source = (
                    name == "DataSource"
                    and attrs["__module__"].endswith("steps.data_sources.data_source")
                ) or (
                    len(bases) > 0
                    and bases[0].__module__.endswith("steps.data_sources.data_source")
                )
                if attribute == "__init__" and not is_data_source:
                    raise AttributeError(
                        'Overriding of "%s" not allowed, override setup() instead.'
                        % attribute
                    )
        meta.has_base = True
        klass = super().__new__(meta, name, bases, attrs)
        return klass

    @property
    def help(self) -> str:
        if self.__name__.endswith("DataSource"):  # pragma: no cover
            return "No help string available."
        if not hasattr(self, ".__help_str__"):

            class StepHelp(self):  # type:ignore[valid-type,misc]
                pass

            StepHelp.__name__ = self.__name__
            StepHelp.__qualname__ = self.__name__
            setattr(StepHelp, _INTERNAL_HELP_KEY, True)
            help_step = StepHelp(name="help_step")
            self.__help_str__ = help_step.help
        return self.__help_str__


class Step(metaclass=StepMeta):
    """Base class for all steps.

    Args:
        name: The name of the step.
        inputs: The inputs to the step.
        args: The args to the step.
        outputs: The name mapping to rename outputs produced by the step.
        progress_interval: How often to log progress in seconds.
        force: Whether to force run the step (ignore saved results).
        verbose: Whether or not to print verbose logs.
        log_level: The logging level to use (:py:data:`~logging.DEBUG`, :py:data:`~logging.INFO`, etc.).
        save_num_proc: The number of processes to use if saving to disk.
        save_num_shards: The number of shards on disk to save the dataset into.
        background: Whether to run the operation in the background.
    """

    def __init__(  # noqa: C901
        self,
        name: str,
        inputs: None
        | dict[str, OutputDatasetColumn | OutputIterableDatasetColumn] = None,
        args: None | dict[str, Any] = None,
        outputs: None | dict[str, str] = None,
        progress_interval: None | int = 60,
        force: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ):
        # Get the cls_name
        cls_name = self.__class__.__name__

        # Check pid
        if DataDreamer.is_background_process():  # pragma: no cover
            raise RuntimeError(
                f"Steps must be initialized in the same process"
                f" ({os.getpid()}) as the DataDreamer() context manager"
                f" ({DataDreamer.ctx.pid}). Use background=True if you want to"
                " run this step in a background process."
            )

        # Check thread
        if not DataDreamer.is_registered_thread():  # pragma: no cover
            raise RuntimeError(
                "Steps cannot be run in arbitrary threads. Use the"
                " concurrent() utility function to run concurrently."
            )

        # Fill in default argument valu]es
        if not isinstance(args, dict):
            args = {}
        if not isinstance(inputs, dict):
            inputs = {}
        if not isinstance(outputs, dict):
            outputs = {}
        assert args is not None
        assert inputs is not None
        assert outputs is not None

        # Initialize variables
        self._initialized: bool = False
        self._resumed: bool = False
        self.name: str = DataDreamer._new_step_name(name)
        if len(self.name) == 0:
            raise ValueError("You must provide a name for the step.")
        self.__progress: None | float = None
        self.__progress_rows: None | int = None
        self.__progress_logging_rows: bool = False
        self.__progress_logged: bool = False
        self.progress_interval: None | int = progress_interval
        self.progress_start = time()
        self.progress_last = time()
        self.__output: None | OutputDataset | OutputIterableDataset = None
        self._pickled: bool = False
        self.__registered: dict[str, Any] = {
            "args": {},
            "required_args": {},
            "inputs": {},
            "required_inputs": {},
            "outputs": [],
            "data_card": defaultdict(lambda: defaultdict(list)),
        }
        self.output_name_mapping = {}
        self.__help: dict[str, Any] = {"args": {}, "inputs": {}, "outputs": {}}
        self.force: bool
        self.force = force or (
            DataDreamer.initialized()
            and DataDreamer._get_parent_step() is not None
            and cast(Step, DataDreamer._get_parent_step()).force
        )
        self.save_num_proc = save_num_proc
        self.save_num_shards = save_num_shards
        self._orig_background = background
        self.background = background if not isinstance(self, SuperStep) else False

        # Initialize the logger
        self.verbose = verbose
        self.log_level = log_level
        self.logger: Logger
        if not hasattr(self.__class__, _INTERNAL_HELP_KEY):
            stderr_handler = logging.StreamHandler()
            stderr_handler.setLevel(logging.NOTSET)
            self.logger = logging.getLogger(
                f"datadreamer.steps.{safe_fn(self.name, allow_slashes=True, to_lower=True)}"
            )
            if RUNNING_IN_PYTEST:
                self.logger.propagate = True
            else:
                self.logger.propagate = False  # pragma: no cover
            log_format: str = (
                logger.handlers[0].formatter and logger.handlers[0].formatter._fmt
            ) or datadreamer_logging.STANDARD_FORMAT
            log_format = log_format.replace(
                "%(message)s", f"[ ➡️ {self.name}] %(message)s"
            )
            formatter = logging.Formatter(log_format, datefmt=DATEFMT, validate=False)
            stderr_handler.setFormatter(formatter)
            self.logger.handlers.clear()
            self.logger.addHandler(stderr_handler)
            effective_level = logger.level if self.log_level is None else self.log_level
            if self.verbose:
                self.logger.setLevel((min(logging.DEBUG, effective_level)))
            elif self.verbose is False:
                self.logger.setLevel(logging.CRITICAL + 1)
            else:
                self.logger.setLevel(effective_level)

        # Run setup
        self.setup()
        if hasattr(self.__class__, _INTERNAL_HELP_KEY):
            return
        self._initialized = True

        # Validate and setup args
        if (
            not set(args.keys()).issubset(set(self.__registered["args"].keys()))
            and "**kwargs" not in self.__registered["args"]
        ) or not set(self.__registered["required_args"].keys()).issubset(
            set(args.keys())
        ):
            raise ValueError(
                f"Expected {uniq_str(self.__registered['args'].keys())} as args,"
                f" with {uniq_str(self.__registered['required_args'].keys())} required,"
                f" got {uniq_str(args.keys())}. See `{cls_name}.help`:\n{self.help}"
            )
        else:
            self.__registered["args"].update(args)
            if "**kwargs" in self.__registered["args"]:
                del self.__registered["args"]["**kwargs"]

        # Validate and setup inputs
        if (
            not set(inputs.keys()).issubset(set(self.__registered["inputs"].keys()))
            or not set(self.__registered["required_inputs"].keys()).issubset(
                set(inputs.keys())
            )
        ) and not hasattr(self.__class__, _INTERNAL_STEP_OPERATION_KEY):
            raise ValueError(
                f"Expected {uniq_str(self.__registered['inputs'].keys())} as inputs,"
                f" with {uniq_str(self.__registered['required_inputs'].keys())} required,"
                f" got {uniq_str(inputs.keys())}. See `{cls_name}.help`:\n{self.help}"
            )
        elif not all(
            [
                isinstance(
                    v, (OutputDatasetColumn, OutputIterableDatasetColumn, type(None))
                )
                for v in inputs.values()
            ]
        ):
            raise TypeError(
                "All inputs must be of type OutputDatasetColumn or"
                " OutputIterableDatasetColumn."
            )
        else:
            self.__registered["inputs"] = inputs

            # Propagate trace info from previous steps
            prev_data_card: DefaultDict[str, DefaultDict[str, list]] = defaultdict(
                lambda: defaultdict(list)
            )
            for v in inputs.values():
                if v is not None:
                    prev_data_card.update(v.step._data_card)
            prev_data_card.update(self.__registered["data_card"])
            self.__registered["data_card"] = prev_data_card

        # Initialize output names mapping
        if len(self.__registered["outputs"]) == 0 and not hasattr(
            self.__class__, _INTERNAL_STEP_OPERATION_KEY
        ):
            raise ValueError("The step must register at least one output.")
        if not set(outputs.keys()).issubset(
            set(self.__registered["outputs"])
        ) and not hasattr(self.__class__, _INTERNAL_STEP_OPERATION_KEY):
            raise ValueError(
                f"{cls_name} only defines {uniq_str(self.__registered['outputs'])} as"
                f" outputs, got {uniq_str(outputs.keys())}."
                f" See `{cls_name}.help`:\n{self.help}"
            )
        output_names = (
            (outputs or {}).keys()
            if hasattr(self.__class__, _INTERNAL_STEP_OPERATION_KEY)
            else self.__registered["outputs"]
        )
        self.output_name_mapping = {o: outputs.get(o, o) for o in output_names}
        self.output_names = tuple([self.output_name_mapping[o] for o in output_names])

        # Run (or resume) within the DataDreamer context
        self._output_folder_path: None | str = None
        if DataDreamer.initialized():
            DataDreamer._start_step(self)
            try:
                self.__setup_folder_and_resume()
            finally:
                DataDreamer._stop_step()
                if hasattr(self, "_lock"):
                    self._lock.release()

    def __setup_folder_and_resume(self):
        if DataDreamer.is_running_in_memory():
            self.__start()
            return

        # Create an output folder for the step
        self._output_folder_path = os.path.join(
            DataDreamer.get_output_folder_path(),
            safe_fn(self.name, allow_slashes=True, to_lower=True),
        )
        os.makedirs(self._output_folder_path, exist_ok=True)
        assert self._output_folder_path is not None

        # Lock working on the step
        step_lock_path = os.path.join(self._output_folder_path, "._dataset.flock")
        self._lock = FileLock(step_lock_path)
        try:
            self._lock.acquire(timeout=60)
            self._lock.release()
        except Timeout:  # pragma: no cover
            logger.info(
                f"Step '{self.name}' is being run in two different processes or"
                " threads concurrently. Waiting for others to finish before"
                " continuing here..."
            )
        finally:
            self._lock.acquire()

        # Check if we have already run this step previously and saved the results to
        # disk
        metadata_path = os.path.join(self._output_folder_path, "step.json")
        dataset_path = os.path.join(self._output_folder_path, "_dataset")
        prev_fingerprint: None | str = None
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                prev_fingerprint = metadata["fingerprint"]
        except FileNotFoundError:
            pass

        # We have already run this step, skip running it
        if prev_fingerprint == self.fingerprint and not self.force:
            self.__output = OutputDataset(
                self, Dataset.load_from_disk(dataset_path), pickled=metadata["pickled"]
            )
            self.progress = 1.0
            self._pickled = metadata["pickled"]
            self.__registered["data_card"].update(metadata["data_card"])
            self._resumed = True
            logger.info(
                f"Step '{self.name}' results loaded from disk. 🙌 It was previously run"
                " and saved."
            )
            # Skip running it
            return

        # We have already run this step, but it is outdated, back up the results
        if prev_fingerprint is not None and (
            prev_fingerprint != self.fingerprint or self.force
        ):
            # ...but it was a different version, backup the results and we'll need
            # to re-run this step
            logger.info(
                f"Step '{self.name}' was previously run and saved, but was outdated. 😞"
            )
            backup_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                "_backups",
                safe_fn(self.name, allow_slashes=True, to_lower=True),
                prev_fingerprint,
            )
            logger.debug(
                f"Step '{self.name}' outdated results are being backed up: {backup_path}"
            )
            move_dir(self._output_folder_path, backup_path)
            logger.debug(
                f"Step '{self.name}' outdated results are backed up: {backup_path}"
            )

        # Check if we have old results for this step that can be restored
        restore_path = os.path.join(
            DataDreamer.get_output_folder_path(),
            "_backups",
            safe_fn(self.name, allow_slashes=True, to_lower=True),
            self.fingerprint,
        )
        if os.path.isfile(os.path.join(restore_path, "step.json")) and not self.force:
            logger.info(
                f"Step '{self.name}' was previously run and the results were backed up. 💾"
            )
            logger.debug(
                f"Step '{self.name}' backed up results are being restored: {restore_path}"
            )
            move_dir(restore_path, self._output_folder_path)
            logger.debug(f"Step '{self.name}' backed up results were restored.")
            self.__setup_folder_and_resume()  # Retry loading
            return

        # Run the step
        self.__start()

    def __start(self):
        if self.background:
            logger.info(f"Step '{self.name}' is running in the background. ⏳")
            self._set_output(None, background_run_func=lambda: self.run())
        else:
            logger.info(f"Step '{self.name}' is running. ⏳")
            if not hasattr(self.__class__, _INTERNAL_TEST_KEY):
                self._set_output(self.run())

    def __finish(self):
        if DataDreamer.is_background_process():  # pragma: no cover
            return
        if isinstance(self.__output, OutputDataset) and hasattr(
            self.__class__, _INTERNAL_STEP_OPERATION_NO_SAVE_KEY
        ):
            self.__delete_progress_from_disk()
            logger.info(f"Step '{self.name}' finished running lazily. 🎉")
        elif isinstance(self.__output, OutputIterableDataset) or hasattr(
            self.__class__, _INTERNAL_STEP_OPERATION_NO_SAVE_KEY
        ):
            self.__delete_progress_from_disk()
            logger.info(f"Step '{self.name}' will run lazily. 🥱")
        elif not self._output_folder_path:
            self.progress = 1.0
            self.__delete_progress_from_disk()
            logger.info(
                f"Step '{self.name}' finished with results available in-memory. 🎉"
            )
        elif isinstance(self.__output, OutputDataset) and not hasattr(
            self.__class__, _INTERNAL_STEP_OPERATION_NO_SAVE_KEY
        ):
            if not self.background:
                self.__save_output_to_disk(self.__output)
            self.progress = 1.0
            self.__delete_progress_from_disk()
            self.__delete_save_cache_from_disk()
            logger.info(f"Step '{self.name}' finished and is saved to disk. 🎉")

        # Set output_names and output_name_mapping if step operation
        if hasattr(self.__class__, _INTERNAL_STEP_OPERATION_KEY) and self.__output:
            self.output_name_mapping = {n: n for n in self.__output.column_names}
            self.output_names = tuple([o for o in self.output_name_mapping.values()])

        # Propagate trace info to parent steps
        if DataDreamer.initialized():
            parent_step = DataDreamer._get_parent_step()
            if isinstance(parent_step, Step):
                parent_step.__registered["data_card"].update(self._data_card)

    def __save_output_to_disk(self, output: OutputDataset):
        if not self._output_folder_path:  # pragma: no cover
            return
        logger.debug(
            f"Step '{self.name}' is being saved to disk: {self._output_folder_path}."
        )
        metadata_path = os.path.join(self._output_folder_path, "step.json")
        dataset_path = os.path.join(self._output_folder_path, "_dataset")
        if self.save_num_shards and self.save_num_shards > 1:
            DataDreamer._enable_hf_datasets_logging()
        output.save_to_disk(
            dataset_path, num_proc=self.save_num_proc, num_shards=self.save_num_shards
        )
        DataDreamer._disable_hf_datasets_logging()
        with open(metadata_path, "w+") as f:
            json.dump(self._get_metadata(output), f, indent=4)
        logger.debug(
            f"Step '{self.name}' is now saved to disk: {self._output_folder_path}."
        )

    def register_input(
        self, input_column_name: str, required: bool = True, help: None | str = None
    ):
        """Register an input for the step. See :doc:`create your own steps
        <pages/advanced_usage/creating_a_new_datadreamer_.../step>` for more details.

        Args:
            input_column_name: The name of the input column.
            required: Whether the input is required.
            help: The help string for the input.
        """
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if not isinstance(input_column_name, str):
            raise TypeError(f"Expected str, got {type(input_column_name)}.")
        if required:
            self.__registered["required_inputs"][input_column_name] = None
        self.__registered["inputs"][input_column_name] = None
        help_optional = "(optional)"
        if not required:
            self.__help["inputs"][input_column_name] = (
                (help or "") + " " + help_optional
            ).strip()
        else:
            self.__help["inputs"][input_column_name] = help

    def register_arg(
        self,
        arg_name: str,
        required: bool = True,
        default: Any = None,
        help: None | str = None,
        default_help: None | str = None,
    ):
        """Register an argument for the step. See :doc:`create your own steps
        <pages/advanced_usage/creating_a_new_datadreamer_.../step>` for more details.

        Args:
            arg_name: The name of the argument.
            required: Whether the argument is required.
            default: The default value of the argument.
            help: The help string for the argument.
            default_help: The help string for the default value of the argument.
        """
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if not isinstance(arg_name, str):
            raise TypeError(f"Expected str, got {type(arg_name)}.")
        assert (
            default is None or not required
        ), f"`default` cannot be set if arg `{arg_name}` is required."
        if required and arg_name != "**kwargs":
            self.__registered["required_args"][arg_name] = None
        self.__registered["args"][arg_name] = self.__registered["args"][arg_name] = (
            {} if arg_name == "**kwargs" else default
        )
        help_optional = ""
        if arg_name == "**kwargs" or (default is None and default_help is None):
            help_optional = "(optional)"
        else:
            help_optional = f"(optional, defaults to {repr(default) if default_help is None else default_help})"
        if not required:
            self.__help["args"][arg_name] = ((help or "") + " " + help_optional).strip()
        else:
            self.__help["args"][arg_name] = help

    def register_output(self, output_column_name: str, help: None | str = None):
        """Register an output for the step. See :doc:`create your own steps
        <pages/advanced_usage/creating_a_new_datadreamer_.../step>` for more details.

        Args:
            output_column_name: The name of the output column.
            help: The help string for the output.
        """
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if not isinstance(output_column_name, str):
            raise TypeError(f"Expected str, got {type(output_column_name)}.")
        if output_column_name not in self.__registered["outputs"]:
            self.__registered["outputs"].append(output_column_name)
        self.__help["outputs"][output_column_name] = help

    def register_data_card(self, data_card_type: str, data_card: Any):
        """Register a data card for the step. See :doc:`create your own steps
        <pages/advanced_usage/creating_a_new_datadreamer_.../step>` for more details.

        Args:
            data_card_type: The type of the data card.
            data_card: The data card.
        """
        if not isinstance(data_card_type, str):
            raise TypeError(f"Expected str, got {type(data_card_type)}.")
        if data_card is not None:
            self.__registered["data_card"][self.name][data_card_type].append(data_card)

    @property
    def args(self) -> dict[str, Any]:
        """The args of the step."""
        return self.__registered["args"].copy()

    @property
    def inputs(self) -> dict[str, OutputDatasetColumn | OutputIterableDatasetColumn]:
        """The inputs of the step."""
        return self.__registered["inputs"].copy()

    def setup(self):
        if "SPHINX_BUILD" not in os.environ:
            raise NotImplementedError("You must implement the .setup() method in Step.")

    def run(self) -> StepOutputType | LazyRows | LazyRowBatches:
        if "SPHINX_BUILD" not in os.environ:
            raise NotImplementedError("You must implement the .run() method in Step.")
        else:  # pragma: no cover
            return None

    def get_run_output_folder_path(self) -> str:
        """Get the run output folder path that can be used by the step for writing
        persistent data."""
        if not self._output_folder_path:
            if not DataDreamer.initialized():  # pragma: no cover
                raise RuntimeError("You must run the Step in a DataDreamer() context.")
            else:
                raise RuntimeError(
                    "No run output folder available. DataDreamer is running in-memory."
                )
        run_output_folder_path = os.path.join(self._output_folder_path, "run_output")
        os.makedirs(run_output_folder_path, exist_ok=True)
        return run_output_folder_path

    def pickle(self, value: Any, *args: Any, **kwargs: Any) -> bytes:
        """Pickle a value so it can be stored in a row produced by this step. See
        :doc:`create your own steps
        <pages/advanced_usage/creating_a_new_datadreamer_.../step>` for more details.

        Args:
            value: The value to pickle.
            *args: The args to pass to :py:meth:`~dill.dumps`.
            **kwargs: The kwargs to pass to :py:meth:`~dill.dumps`.
        """
        self._pickled = True
        if self.__output:
            self.output._pickled = True
        kwargs[_INTERNAL_PICKLE_KEY] = True
        return _pickle(value, *args, **kwargs)

    def unpickle(self, value: bytes) -> Any:
        """Unpickle a value that was stored in a row produced by this step with
        :py:meth:`~Step.pickle`. See :doc:`create your own steps
        <pages/advanced_usage/creating_a_new_datadreamer_.../step>` for more details.

        Args:
            value: The value to unpickle.
        """
        return _unpickle(value)

    def __write_progress_to_disk(self):
        # Write the progress to disk in the child process to share with the parent
        if (
            self.background
            and DataDreamer.is_background_process()
            and self._output_folder_path
        ):  # pragma: no cover
            background_progress_path = os.path.join(
                self._output_folder_path, ".background_progress"
            )
            try:
                with open(background_progress_path, "w+") as f:
                    json.dump(
                        {
                            "progress": self.__progress,
                            "progress_rows": self.__progress_rows,
                        },
                        f,
                        indent=4,
                    )
            except Exception:
                pass

    def __read_progress_from_disk(self):
        # Read the progress from the disk in the parent process
        if (
            self.__progress != 1.0
            and self.background
            and not DataDreamer.is_background_process()
            and self._output_folder_path
        ):
            background_progress_path = os.path.join(
                self._output_folder_path, ".background_progress"
            )
            try:
                with open(background_progress_path, "r") as f:
                    progress_data = json.load(f)
                    if progress_data.get("progress", None) is not None:
                        self.__progress = max(
                            progress_data["progress"], self.__progress or 0.0
                        )
                    if progress_data.get("progress_rows", None) is not None:
                        self.__progress_rows = max(
                            progress_data["progress_rows"], self.__progress_rows or 0
                        )
            except Exception:
                pass

    def __delete_progress_from_disk(self):
        # Delete the progress from the disk once done
        if (
            self._output_folder_path
            and self.background
            and not DataDreamer.is_background_process()
        ):
            background_progress_path = os.path.join(
                self._output_folder_path, ".background_progress"
            )
            try:
                os.remove(background_progress_path)
            except Exception:
                pass

    def __delete_save_cache_from_disk(self):
        # Delete the save cache from the disk once done
        if self._output_folder_path:
            save_cache_path = os.path.join(
                self._output_folder_path, ".datadreamer_save_cache"
            )
            if os.path.isdir(save_cache_path):
                shutil.rmtree(save_cache_path, ignore_errors=True)

    @property
    def progress(self) -> None | float:
        """The progress of the step."""
        self.__read_progress_from_disk()
        return self.__progress

    @progress.setter
    def progress(self, value: float):
        prev_progress = self.__progress or 0.0
        if isinstance(self.__output, OutputDataset):
            value = 1.0
        value = max(min(value, 1.0), prev_progress)
        should_log = False
        if (
            (self.__progress_logged or value < 1.0)
            and self.progress_interval is not None
            and (time() - self.progress_last) > self.progress_interval
            and value > prev_progress
            and (not self.__progress_logging_rows or value < 1.0)
        ):
            should_log = True
            self.progress_last = time()
            self.__progress_logged = True
        self.__progress = value
        if should_log:
            eta = progress_eta(self.__progress, self.progress_start)
            logger.info(
                f"Step '{self.name}' progress:"
                f" {self.__get_progress_string()} 🔄 {eta}"
            )
            self.__write_progress_to_disk()
        if (
            self.__progress == 1.0
            and self.__progress > prev_progress
            and isinstance(self.__output, OutputIterableDataset)
        ):
            logger.info(f"Step '{self.name}' finished running lazily. 🎉")

    def _set_progress_rows(self, value: int):
        value = max(value, self.__progress_rows or 0)
        should_log = False
        if (
            not self.progress
            and self.progress_interval is not None
            and (time() - self.progress_last) > self.progress_interval
            and value > 0
            and value > (self.__progress_rows or 0)
        ):
            should_log = True
            self.__progress_logging_rows = True
            self.progress_last = time()
            self.__progress_logged = True
        self.__progress_rows = value
        if should_log:
            logger.info(
                f"Step '{self.name}' progress:" f" {self.__progress_rows} row(s) 🔄"
            )
            self.__write_progress_to_disk()

    def __get_progress_string(self):
        if not self.progress and self.__progress_rows:
            return f"{self.__progress_rows} row(s) processed"
        elif self.progress is not None:
            progress_int = int(self.progress * 100)
            return f"{progress_int}%"
        else:
            return "0%"

    @property
    def output(self) -> OutputDataset | OutputIterableDataset:
        """The output dataset of the step."""
        if self.__output is None:
            if self.__progress is None and not self.background:
                raise StepOutputError("Step has not been run. Output is not available.")
            elif self.background:
                raise StepOutputError(
                    f"Step is still running in the background"
                    f" ({self.__get_progress_string()})."
                    " Output is not available yet. To wait for this step to finish, you"
                    " can use the wait() utility function."
                )
            else:
                raise StepOutputError(
                    f"Step is still running ({self.__get_progress_string()})."
                    " Output is not available yet."
                )
        else:
            return self.__output

    @property
    def dataset_path(self) -> str:
        """The path to the step's output dataset on disk in HuggingFace
        :py:class:`~datasets.Dataset` format if the step has been saved to disk.
        """
        assert not DataDreamer.is_running_in_memory(), (
            "This step's dataset has not been saved to disk. DataDreamer is running"
            " in-memory."
        )
        if isinstance(self.output, OutputIterableDataset):
            raise RuntimeError(
                "This step's dataset has not been saved to disk yet."
                " Use `.save()` on the step to first save it to disk."
            )
        else:
            return os.path.join(cast(str, self._output_folder_path), "_dataset")

    def _set_output(  # noqa: C901
        self,
        value: StepOutputType | LazyRows | LazyRowBatches,
        background_run_func: None | Callable = None,
    ):
        if self.__output:
            raise StepOutputError("Step has already been run.")
        logger.debug(f"Step '{self.name}' results are being processed.")
        if background_run_func:
            _monkey_patch_iterable_dataset_apply_feature_types()

            def with_result_process(process):
                DataDreamer._add_process(process)

            def with_result(self, output):
                data_card, self.__output = dill.loads(output)
                self.__registered["data_card"].update(data_card)
                self.__finish()

            run_in_background_process_no_block(
                _output_to_dataset,
                result_process_func=with_result_process,
                result_func=partial(with_result, self),
                step=self,
                output_names=tuple(self.__registered["outputs"]),
                output_name_mapping=self.output_name_mapping,
                set_progress=partial(
                    lambda self, progress: setattr(self, "progress", progress), self
                ),
                set_progress_rows=partial(
                    lambda self, rows: self._set_progress_rows(rows), self
                ),
                get_pickled=partial(lambda self: self._pickled, self),
                value=background_run_func,
                save_output_to_disk=partial(
                    lambda self, output: self.__save_output_to_disk(output), self
                ),
            )
        else:
            self.__output = _output_to_dataset(
                pipe=None,
                step=self,
                output_names=tuple(self.__registered["outputs"]),
                output_name_mapping=self.output_name_mapping,
                set_progress=partial(
                    lambda self, progress: setattr(self, "progress", progress), self
                ),
                set_progress_rows=partial(
                    lambda self, rows: self._set_progress_rows(rows), self
                ),
                get_pickled=partial(lambda self: self._pickled, self),
                value=value,
                save_output_to_disk=partial(
                    lambda self, output: self.__save_output_to_disk(output), self
                ),
            )
            self.__finish()

    def head(self, n=5, shuffle=False, seed=None, buffer_size=1000) -> DataFrame:
        """Return the first ``n`` rows of the step's output as a pandas
        :py:class:`~pandas.DataFrame` for easy viewing.

        Args:
            n: The number of rows to return.
            shuffle: Whether to shuffle the rows before taking the first ``n``.
            seed: The seed to use if shuffling.
            buffer_size: The buffer size to use if shuffling and the step's output is
                an iterable dataset.
        """
        return self.output.head(
            n=n, shuffle=shuffle, seed=seed, buffer_size=buffer_size
        )

    @property
    def _data_card(self) -> dict:
        data_card = self.__registered["data_card"]
        if DataCardType.DATETIME not in data_card[self.name]:
            data_card[self.name][DataCardType.DATETIME] = datetime.now().isoformat()
        data_card = self.__registered["data_card"].copy()
        for step_name in data_card:
            sort_data_card(data_card[step_name])
        return json.loads(json.dumps(data_card))

    def data_card(self) -> None:
        """Print the data card for the step."""
        print(json.dumps(self._data_card, indent=4))

    def select(
        self,
        indices: Iterable,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Select rows from the step's output by their indices. See
        :py:meth:`~datasets.Dataset.select` for more details.

        Args:
            indices: The indices of the rows to select.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_select_step, **kwargs)()

    def select_columns(
        self,
        column_names: str | list[str],
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Select columns from the step's output. See
        :py:meth:`~datasets.Dataset.select_columns` for more details.

        Args:
            column_names: The names of the columns to select.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """

        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_select_columns_step, **kwargs)()

    def take(
        self,
        n: int,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Take the first ``n`` rows from the step's output. See
        :py:meth:`~datasets.IterableDataset.take` for more details.

        Args:
            n: The number of rows to take.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """

        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_take_step, **kwargs)()

    def skip(
        self,
        n: int,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Skip the first ``n`` rows from the step's output. See
        :py:meth:`~datasets.IterableDataset.skip` for more details.

        Args:
            n: The number of rows to skip.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_skip_step, **kwargs)()

    def shuffle(
        self,
        seed: None | int = None,
        buffer_size: int = 1000,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Shuffle the rows of the step's output. See
        :py:meth:`~datasets.IterableDataset.shuffle` for more details.

        Args:
            seed: The random seed to use for shuffling the step's output.
            buffer_size: The buffer size to use for shuffling the dataset, if the step's
                output is an :py:class:`~datadreamer.datasets.OutputIterableDataset`.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """

        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_shuffle_step, **kwargs)()

    def sort(
        self,
        column_names: str | Sequence[str],
        reverse: bool | Sequence[bool] = False,
        null_placement: str = "at_end",
        name: None | str = None,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Sort the rows of the step's output. See
        :py:meth:`~datasets.Dataset.sort` for more details.

        Args:
            column_names: The names of the columns to sort by.
            reverse: Whether to sort in reverse order.
            null_placement: Where to place null values in the sorted dataset.
            name: The name of the operation.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_sort_step, **kwargs)()

    def add_item(
        self,
        item: dict,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Add a row to the step's output. See
        :py:meth:`~datasets.Dataset.add_item` for more details.

        Args:
            item: The item to add to the step's output.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_add_item_step, **kwargs)()

    def map(
        self,
        function: Callable,
        with_indices: bool = False,
        input_columns: None | str | list[str] = None,
        batched: bool = False,
        batch_size: int = 1000,
        remove_columns: None | str | list[str] = None,
        total_num_rows: None | int = None,
        auto_progress: bool = True,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """
        Apply a function to the step's output. See
        :py:meth:`~datasets.Dataset.map` for more details.

        Args:
            function: The function to apply to rows of the step's output.
            with_indices: Whether to pass the indices of the rows to the function.
            input_columns: The names of the columns to pass to the function.
            batched: Whether to apply the function in batches.
            batch_size: The batch size to use if applying the function in batches.
            remove_columns: The names of the columns to remove from the output.
            total_num_rows: The total number of rows being processed (helps with
                displaying progress).
            auto_progress: Whether to automatically update the progress % for this step.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        del kwargs["auto_progress"]
        if lazy and (total_num_rows is None and auto_progress):
            warnings.warn(
                "You did not specify `total_num_rows`, so we cannot"
                " automatically update the progress % for this step. Either"
                " specify map(..., total_num_rows=#) or, to disable"
                " this warning, specify map(.., auto_progress = False)",
                stacklevel=2,
            )
        return partial(_create_map_step, **kwargs)()

    def filter(
        self,
        function: Callable,
        with_indices: bool = False,
        input_columns: None | str | list[str] = None,
        batched: bool = False,
        batch_size: int = 1000,
        total_num_rows: None | int = None,
        auto_progress: bool = True,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """
        Filter rows from the step's output. See
        :py:meth:`~datasets.Dataset.filter` for more details.

        Args:
            function: The function to use for filtering rows of the step's output.
            with_indices: Whether to pass the indices of the rows to the function.
            input_columns: The names of the columns to pass to the function.
            batched: Whether to apply the function in batches.
            batch_size: The batch size to use if applying the function in batches.
            total_num_rows: The total number of rows being processed (helps with
                displaying progress).
            auto_progress: Whether to automatically update the progress % for this step.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        del kwargs["auto_progress"]
        if lazy and (total_num_rows is None and auto_progress):
            warnings.warn(
                "You did not specify `total_num_rows`, so we cannot"
                " automatically update the progress % for this step. Either"
                " specify filter(..., total_num_rows=#) or, to disable"
                " this warning, specify filter(.., auto_progress = False)",
                stacklevel=2,
            )
        return partial(_create_filter_step, **kwargs)()

    def rename_column(
        self,
        original_column_name: str,
        new_column_name: str,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """
        Rename a column in the step's output. See
        :py:meth:`~datasets.Dataset.rename_column` for more details.

        Args:
            original_column_name: The original name of the column.
            new_column_name: The new name of the column.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_rename_column_step, **kwargs)()

    def rename_columns(
        self,
        column_mapping: dict[str, str],
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """
        Rename columns in the step's output. See
        :py:meth:`~datasets.Dataset.rename_columns` for more details.

        Args:
            column_mapping: The mapping of original column names to new column names.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_rename_columns_step, **kwargs)()

    def remove_columns(
        self,
        column_names: str | list[str],
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """
        Remove columns from the step's output. See
        :py:meth:`~datasets.Dataset.remove_columns` for more details.

        Args:
            column_names: The names of the columns to remove.
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_remove_columns_step, **kwargs)()

    def splits(
        self,
        train_size: None | float | int = None,
        validation_size: None | float | int = None,
        test_size: None | float | int = None,
        stratify_by_column: None | str = None,
        name: None | str = None,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> dict[str, "Step"]:
        """Split the step's output into multiple splits for training, validation, and
        testing. If ``train_size`` or ``validation_size`` or ``test_size`` is not
        specified, the corresponding split will not be created.

        Args:
            train_size: The size of the training split. If a float, should be between 0.0
                and 1.0 and represent the proportion of the dataset to include in the
                training split. If an int, should be the number of rows to include in
                the training split.

            validation_size: The size of the validation split. If a float, should be
                between 0.0 and 1.0 and represent the proportion of the dataset to
                include in the validation split. If an int, should be the number of rows
                to include in the validation split.
            test_size: The size of the test split. If a float, should be between 0.0 and
                1.0 and represent the proportion of the dataset to include in the test
                split. If an int, should be the number of rows to include in the test
                split.
            stratify_by_column: The name of the column to use to stratify equally
                between splits.
            name: The name of the operation.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A dictionary where the keys are the names of the splits and the values are
            new steps with the split applied.
        """

        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_splits_step, **kwargs)()

    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = False,
        name: None | str = None,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Shard the step's output into multiple shards. See
        :py:meth:`~datasets.Dataset.shard` for more details.

        Args:
            num_shards: The number of shards to split the dataset into.
            index: The index of the shard to select.
            contiguous: Whether to select contiguous blocks of indicies for shards.
            name: The name of the operation.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """

        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_shard_step, **kwargs)()

    def reverse(
        self,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Reverse the rows of the step's output.

        Args:
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_reverse_step, **kwargs)()

    def save(
        self,
        name: None | str = None,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Save the step's output to disk.

        Args:
            name: The name of the operation.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_save_step, **kwargs)()

    def copy(
        self,
        name: None | str = None,
        lazy: None | bool = None,
        progress_interval: None | int | Default = DEFAULT,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int | Default = DEFAULT,
        save_num_shards: None | int | Default = DEFAULT,
        background: bool = False,
    ) -> "Step":
        """Create a copy of the step's output.

        Args:
            name: The name of the operation.
            lazy: Whether to run the operation lazily.
            progress_interval: How often to log progress in seconds.
            force: Whether to force run the step (ignore saved results).
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            background: Whether to run the operation in the background.

        Returns:
            A new step with the operation applied.
        """
        kwargs = dict(locals())
        if lazy is None:
            wait(self)
            if isinstance(self.output, OutputDataset):
                kwargs["lazy"] = False
            else:
                kwargs["lazy"] = True
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_copy_step, **kwargs)()

    def export_to_dict(
        self,
        train_size: None | float | int = None,
        validation_size: None | float | int = None,
        test_size: None | float | int = None,
        stratify_by_column: None | str = None,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
    ) -> dict:
        """Export the step's output to a dictionary and optionally create splits.
        See :py:meth:`~Step.splits` for more details on splits behavior.

        Args:
            train_size: The size of the training split. If a float, should be between 0.0
                and 1.0 and represent the proportion of the dataset to include in the
                training split. If an int, should be the number of rows to include in
                the training split.
            validation_size: The size of the validation split. If a float, should be
                between 0.0 and 1.0 and represent the proportion of the dataset to
                include in the validation split. If an int, should be the number of rows
                to include in the validation split.
            test_size: The size of the test split. If a float, should be between 0.0 and
                1.0 and represent the proportion of the dataset to include in the test
                split. If an int, should be the number of rows to include in the test
                split.
            stratify_by_column: The name of the column to use to stratify equally
                between splits.
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.

        Returns:
            The step's output as a dictionary.
        """
        output_dataset, dataset_dict = _step_to_dataset_dict(
            self,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            stratify_by_column=stratify_by_column,
            writer_batch_size=writer_batch_size,
            save_num_proc=save_num_proc,
            save_num_shards=save_num_shards,
        )
        if len(dataset_dict) > 1:
            result = {
                split: _unpickle_export(
                    export=dataset_dict[split].to_dict(), output_dataset=output_dataset
                )
                for split in dataset_dict
            }
            logger.info(f"Step '{self.name}' splits exported to dicts. 💫")
            return result
        else:
            result = dataset_dict[list(dataset_dict.keys())[0]].to_dict()
            result = _unpickle_export(export=result, output_dataset=output_dataset)
            logger.info(f"Step '{self.name}' exported to a dict. 💫")
            return result

    def export_to_list(
        self,
        train_size: None | float | int = None,
        validation_size: None | float | int = None,
        test_size: None | float | int = None,
        stratify_by_column: None | str = None,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
    ) -> list | dict:
        """Export the step's output to a list and optionally create splits. See
        :py:meth:`~Step.splits` for more details on splits behavior.

        Args:
            train_size: The size of the training split. If a float, should be between 0.0
                and 1.0 and represent the proportion of the dataset to include in the
                training split. If an int, should be the number of rows to include in
                the training split.
            validation_size: The size of the validation split. If a float, should be
                between 0.0 and 1.0 and represent the proportion of the dataset to
                include in the validation split. If an int, should be the number of rows
                to include in the validation split.
            test_size: The size of the test split. If a float, should be between 0.0 and
                1.0 and represent the proportion of the dataset to include in the test
                split. If an int, should be the number of rows to include in the test
                split.
            stratify_by_column: The name of the column to use to stratify equally
                between splits.
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
        """
        output_dataset, dataset_dict = _step_to_dataset_dict(
            self,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            stratify_by_column=stratify_by_column,
            writer_batch_size=writer_batch_size,
            save_num_proc=save_num_proc,
            save_num_shards=save_num_shards,
        )
        if len(dataset_dict) > 1:
            result = {
                split: _unpickle_export(
                    export=dataset_dict[split].to_list(), output_dataset=output_dataset
                )
                for split in dataset_dict
            }
            logger.info(f"Step '{self.name}' splits exported to lists. 💫")
            return result
        else:
            result = dataset_dict[list(dataset_dict.keys())[0]].to_list()
            result = _unpickle_export(export=result, output_dataset=output_dataset)
            logger.info(f"Step '{self.name}' exported to a list. 💫")
            return result

    def export_to_json(
        self,
        path: str,
        train_size: None | float | int = None,
        validation_size: None | float | int = None,
        test_size: None | float | int = None,
        stratify_by_column: None | str = None,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        **to_json_kwargs,
    ) -> str | dict:
        """Export the step's output to a JSON file and optionally create splits. See
        :py:meth:`~Step.splits` for more details on splits behavior.

        Args:
            path: The path to save the JSON file to.
            train_size: The size of the training split. If a float, should be between 0.0
                and 1.0 and represent the proportion of the dataset to include in the
                training split. If an int, should be the number of rows to include in
                the training split.
            validation_size: The size of the validation split. If a float, should be
                between 0.0 and 1.0 and represent the proportion of the dataset to
                include in the validation split. If an int, should be the number of rows
                to include in the validation split.
            test_size: The size of the test split. If a float, should be between 0.0 and
                1.0 and represent the proportion of the dataset to include in the test
                split. If an int, should be the number of rows to include in the test
                split.
            stratify_by_column: The name of the column to use to stratify equally
                between splits.
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            to_json_kwargs: Additional keyword arguments to pass to
                :py:meth:`~datasets.Dataset.to_json`.

        Returns:
            The path to the JSON file or a dictionary of paths if creating splits.
        """
        output_dataset, dataset_dict = _step_to_dataset_dict(
            self,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            stratify_by_column=stratify_by_column,
            writer_batch_size=writer_batch_size,
            save_num_proc=save_num_proc,
            save_num_shards=save_num_shards,
        )
        if len(dataset_dict) > 1:
            split_paths = _path_to_split_paths(path, dataset_dict)
            for split in dataset_dict:
                dataset_dict[split].to_json(
                    split_paths[split], num_proc=save_num_proc, **to_json_kwargs
                )
            dir = os.path.dirname(path)
            logger.info(f"Step '{self.name}' splits exported as JSON files 💫 : {dir}")
            return split_paths
        else:
            dataset_dict[list(dataset_dict.keys())[0]].to_json(
                path, num_proc=save_num_proc, **to_json_kwargs
            )
            logger.info(f"Step '{self.name}' exported as JSON file 💫 : {path}")
            return path

    def export_to_csv(
        self,
        path: str,
        sep=",",
        train_size: None | float | int = None,
        validation_size: None | float | int = None,
        test_size: None | float | int = None,
        stratify_by_column: None | str = None,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        **to_csv_kwargs,
    ) -> str | dict:
        """Export the step's output to a CSV file and optionally create splits. See
        :py:meth:`~Step.splits` for more details on splits behavior.

        Args:
            path: The path to save the CSV file to.
            sep: The delimiter to use for the CSV file.
            train_size: The size of the training split. If a float, should be between 0.0
                and 1.0 and represent the proportion of the dataset to include in the
                training split. If an int, should be the number of rows to include in
                the training split.
            validation_size: The size of the validation split. If a float, should be
                between 0.0 and 1.0 and represent the proportion of the dataset to
                include in the validation split. If an int, should be the number of rows
                to include in the validation split.
            test_size: The size of the test split. If a float, should be between 0.0 and
                1.0 and represent the proportion of the dataset to include in the test
                split. If an int, should be the number of rows to include in the test
                split.
            stratify_by_column: The name of the column to use to stratify equally
                between splits.
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            to_csv_kwargs: Additional keyword arguments to pass to
                :py:meth:`~datasets.Dataset.to_csv`.

        Returns:
            The path to the CSV file or a dictionary of paths if creating splits.
        """
        output_dataset, dataset_dict = _step_to_dataset_dict(
            self,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            stratify_by_column=stratify_by_column,
            writer_batch_size=writer_batch_size,
            save_num_proc=save_num_proc,
            save_num_shards=save_num_shards,
        )
        if len(dataset_dict) > 1:
            split_paths = _path_to_split_paths(path, dataset_dict)
            for split in dataset_dict:
                dataset_dict[split].to_csv(
                    split_paths[split], num_proc=save_num_proc, sep=sep, **to_csv_kwargs
                )
            dir = os.path.dirname(path)
            logger.info(f"Step '{self.name}' splits exported as CSV files 💫 : {dir}")
            return split_paths
        else:
            dataset_dict[list(dataset_dict.keys())[0]].to_csv(
                path, num_proc=save_num_proc, sep=sep, **to_csv_kwargs
            )
            logger.info(f"Step '{self.name}' exported as CSV file 💫 : {path}")
            return path

    def export_to_hf_dataset(
        self,
        path: str,
        train_size: None | float | int = None,
        validation_size: None | float | int = None,
        test_size: None | float | int = None,
        stratify_by_column: None | str = None,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
    ) -> Dataset | DatasetDict:
        """Export the step's output to a Hugging Face :py:class:`~datasets.Dataset` and
        optionally create splits. See :py:meth:`~Step.splits` for more details on splits
        behavior.

        Args:
            path: The path to save the Hugging Face :py:class:`~datasets.Dataset` folder
                to.
            train_size: The size of the training split. If a float, should be between 0.0
                and 1.0 and represent the proportion of the dataset to include in the
                training split. If an int, should be the number of rows to include in
                the training split.
            validation_size: The size of the validation split. If a float, should be
                between 0.0 and 1.0 and represent the proportion of the dataset to
                include in the validation split. If an int, should be the number of rows
                to include in the validation split.
            test_size: The size of the test split. If a float, should be between 0.0 and
                1.0 and represent the proportion of the dataset to include in the test
                split. If an int, should be the number of rows to include in the test
                split.
            stratify_by_column: The name of the column to use to stratify equally
                between splits.
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.

        Returns:
            The step's output as a Hugging Face :py:class:`~datasets.Dataset` or
            :py:class:`~datasets.DatasetDict` if creating splits.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        output_dataset, dataset_dict = _step_to_dataset_dict(
            self,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            stratify_by_column=stratify_by_column,
            writer_batch_size=writer_batch_size,
            save_num_proc=save_num_proc,
            save_num_shards=save_num_shards,
        )
        if len(dataset_dict) > 1:
            dataset_dict.save_to_disk(
                path,
                num_proc=self.save_num_proc,
                num_shards={split: save_num_shards or 1 for split in dataset_dict},
            )
            logger.info(f"Step '{self.name}' splits exported as HF DatasetDict. 💫")
            dataset_dict = cast(
                DatasetDict,
                _unpickle_export(export=dataset_dict, output_dataset=output_dataset),
            )
            return dataset_dict
        else:
            dataset_dict[list(dataset_dict.keys())[0]].save_to_disk(
                path, num_proc=self.save_num_proc, num_shards=save_num_shards
            )
            logger.info(f"Step '{self.name}' exported as HF Dataset. 💫")
            dataset_dict = cast(
                DatasetDict,
                _unpickle_export(export=dataset_dict, output_dataset=output_dataset),
            )
            return dataset_dict[list(dataset_dict.keys())[0]]

    def publish_to_hf_hub(  # noqa: C901
        self,
        repo_id: str,
        branch: None | str = None,
        private: bool = False,
        token: None | str = None,
        train_size: None | float | int = None,
        validation_size: None | float | int = None,
        test_size: None | float | int = None,
        stratify_by_column: None | str = None,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        is_synthetic: bool = True,
        **kwargs,
    ) -> str:  # pragma: no cover
        """Publish the step's output to the Hugging Face Hub as a dataset and optionally
        create splits. See :py:meth:`~Step.splits` for more details on splits behavior.
        See :py:meth:`~datasets.Dataset.push_to_hub` for more details on publishing.

        Args:
            repo_id: The repository ID to publish the dataset to.
            branch: The branch to push the dataset to.
            private: Whether to make the dataset private.
            token: The Hugging Face API token to use for authentication.
            train_size: The size of the training split. If a float, should be between 0.0
                and 1.0 and represent the proportion of the dataset to include in the
                training split. If an int, should be the number of rows to include in
                the training split.
            validation_size: The size of the validation split. If a float, should be
                between 0.0 and 1.0 and represent the proportion of the dataset to
                include in the validation split. If an int, should be the number of rows
                to include in the validation split.
            test_size: The size of the test split. If a float, should be between 0.0 and
                1.0 and represent the proportion of the dataset to include in the test
                split. If an int, should be the number of rows to include in the test
                split.
            stratify_by_column: The name of the column to use to stratify equally
                between splits.
            writer_batch_size: The batch size to use if saving to disk.
            save_num_proc: The number of processes to use if saving to disk.
            save_num_shards: The number of shards on disk to save the dataset into.
            is_synthetic: Whether the dataset is synthetic (applies certain metadata
                when publishing).
            **kwargs: Additional keyword arguments to pass to
                :py:meth:`~datasets.Dataset.push_to_hub`.

        Returns:
            The URL to the published dataset.
        """
        # Login
        api = hf_hub_login(token=token)
        if "/" not in repo_id:
            repo_id = f"{api.whoami()['name']}/{repo_id}"

        # Prepare for publishing
        (tags, dataset_names, model_names, upload_metadata) = prepare_to_publish(
            step_metadata=self._get_metadata(self.output),
            api=api,
            repo_id=repo_id,
            repo_type="dataset",
            branch=branch,
            is_synthetic=is_synthetic,
        )

        # Push data
        output_dataset, dataset_dict = _step_to_dataset_dict(
            self,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            stratify_by_column=stratify_by_column,
            writer_batch_size=writer_batch_size,
            save_num_proc=save_num_proc,
            save_num_shards=save_num_shards,
        )
        DataDreamer._enable_hf_datasets_logging()
        if len(dataset_dict) > 1:
            dataset_dict.push_to_hub(
                repo_id=repo_id, revision=branch, private=private, **kwargs
            )
        else:
            dataset_dict[list(dataset_dict.keys())[0]].push_to_hub(
                repo_id=repo_id, revision=branch, private=private, **kwargs
            )
        DataDreamer._disable_hf_datasets_logging()

        # Upload metadata
        upload_metadata()

        # Calculate dataset size category
        total_num_rows = sum([len(dataset_dict[key]) for key in dataset_dict.keys()])
        size_categories = {
            (1000 * (10**0)): "n<1K",
            (1000 * (10**1)): "1K<n<10K",
            (1000 * (10**2)): "10K<n<100K",
            (1000 * (10**3)): "100K<n<1M",
            (1000 * (10**4)): "1M<n<10M",
            (1000 * (10**5)): "10M<n<100M",
            (1000 * (10**6)): "100M<n<1B",
            (1000 * (10**7)): "1B<n<10B",
            (1000 * (10**8)): "10B<n<100B",
            (1000 * (10**9)): "100B<n<1T",
            (1000 * (10**10)): "n>1T",
        }
        size_category = "n>1T"
        for size_category_thresh, size_category_label in size_categories.items():
            if total_num_rows < size_category_thresh:
                size_category = size_category_label
                break

        # Update README.md
        readme_contents = get_readme_contents(
            repo_id, repo_type="dataset", revision=branch
        )
        readme_contents = readme_contents.replace(
            "More Information needed", "Add more information here"
        )
        if "tags:" not in readme_contents:
            title_and_body = ""
            if "# Dataset Card" not in readme_contents:
                title_and_body = (
                    "# Dataset Card\n\n[Add more information here]"
                    "(https://huggingface.co/datasets/templates/dataset-card-example)\n"
                )
            tags = tags + model_names
            if len(dataset_names) > 0:
                source_datasets = (
                    "source_datasets:\n- " + ("\n- ".join(dataset_names)) + "\n"
                )
            else:
                source_datasets = ""
            readme_contents = readme_contents.replace("---\n", "___header_temp___\n", 1)
            readme_contents = readme_contents.replace(
                "---\n",
                (
                    source_datasets
                    + (
                        "library_name: datadreamer\n"
                        f"size_categories:\n- {size_category}"
                        "\ntags:\n- " + ("\n- ".join(tags)) + "\n---\n" + title_and_body
                    )
                ),
            )
            readme_contents = readme_contents.replace("___header_temp___\n", "---\n", 1)
        if "DataDreamer" not in readme_contents:
            readme_contents += (
                f"\n\n---\n"
                f"This dataset was produced with [DataDreamer 🤖💤](https://datadreamer.dev)."
                f" The {'synthetic ' if is_synthetic else ''}dataset card can be"
                f" found [here](datadreamer.json)."
            )
        api.upload_file(
            path_or_fileobj=BytesIO(bytes(readme_contents, "utf8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            revision=branch,
            commit_message="Pushed by DataDreamer",
            commit_description="Update README.md",
        )

        # Construct and return URL
        url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(
            f"Dataset produced by step '{self.name}' published to HF Hub 💫 : {url}"
        )
        return url

    def _get_metadata(
        self, output: None | OutputDataset | OutputIterableDataset
    ) -> dict:
        def package_exists(req: str) -> bool:
            try:
                metadata.version(req)
                return True
            except metadata.PackageNotFoundError:
                return False

        return {
            "data_card": self._data_card,
            "__version__": __version__,
            "datetime": datetime.now().isoformat(),
            "type": type(self).__name__,
            "name": self.name,
            "version": self.version,
            "fingerprint": self.fingerprint,
            "pickled": output._pickled if output else False,
            "req_versions": {
                req: metadata.version(req)
                for req in [
                    "dill",
                    "sqlitedict",
                    "torch",
                    "numpy",
                    "transformers",
                    "datasets",
                    "huggingface_hub",
                    "accelerate",
                    "peft",
                    "tiktoken",
                    "tokenizers",
                    "petals",
                    "openai",
                    "auto_gptq",
                    "ctransformers",
                    "optimum",
                    "bitsandbytes",
                    "faiss",
                    "litellm",
                    "trl",
                    "setfit",
                    "vllm",
                    "ai21",
                    "together",
                    "anthropic",
                    "cohere",
                    "boto3",
                    "google.generativeai",
                    "google-cloud-aiplatform",
                ]
                if package_exists(req)
            },
            "interpreter": sys.version,
        }

    @property
    def version(self) -> float:
        return 1.0

    @cached_property
    def fingerprint(self) -> str:
        def filter_arg_name(arg_name: str) -> bool:
            return arg_name not in [
                "batch_size",
                "batch_scheduler_buffer_size",
                "adaptive_batch_size",
                "progress_interval",
                "force",
                "cache_only",
                "verbose",
                "log_level",
                "total_num_prompts",
            ]

        def map_value(val: Any) -> str:
            if isinstance(val, _Cachable):
                return Hasher.hash((val.version, val._cache_name))
            return stable_fingerprint(val)

        return Hasher.hash(
            [
                str(type(self).__name__),
                self.name,
                self.version,
                {
                    arg_name: map_value(val)
                    for arg_name, val in self.__registered["args"].items()
                    if filter_arg_name(arg_name)
                },
                list(self.__registered["inputs"].keys()),
                list(
                    [
                        (c.step.fingerprint, c.column_names) if c is not None else None
                        for c in self.__registered["inputs"].values()
                    ]
                ),
                list(self.__registered["outputs"]),
                self.output_name_mapping,
                self.save_num_shards,
            ]
        )

    @cached_property
    def help(self) -> str:
        # Representation helpers
        def dict_to_str(d: dict, delim: str = ": "):
            dict_repr = ",".join(
                [f"\n\t\t{repr(k)}{delim}{repr(v)}" for k, v in d.items()]
            )
            if len(d) > 0:
                dict_repr += "\n\t"
            return dict_repr

        def repr_var(name: str, value: Any):
            return f"\t{name} = {repr(value)},\n"

        def repr_dict_var(name: str, value: dict, delim: str = ": "):
            return f"\t{name} = {{" + dict_to_str(value, delim=delim) + "},\n"

        name_repr = repr_var("name", "The name of the step.")
        if len(self.__help["args"]) > 0:
            args_repr = repr_dict_var("args", self.__help["args"])
        else:
            args_repr = ""
        if len(self.__help["inputs"]) > 0:
            inputs_repr = repr_dict_var("inputs", self.__help["inputs"])
        else:
            inputs_repr = ""
        outputs_repr = repr_dict_var("outputs", self.__help["outputs"])
        return (
            f"{type(self).__name__}(\n"
            f"{name_repr}"
            f"{inputs_repr}"
            f"{args_repr}"
            f"{outputs_repr}"
            ")"
        )

    def __repr__(self) -> str:
        # Representation helpers
        def dict_to_str(d: dict, delim: str = ": "):
            dict_repr = ",".join(
                [f"\n\t\t{repr(k)}{delim}{repr(v)}" for k, v in d.items()]
            )
            if len(d) > 0:
                dict_repr += "\n\t"
            return dict_repr

        def repr_var(name: str, value: Any):
            return f"\t{name}={repr(value)},\n"

        def repr_dict_var(name: str, value: dict, delim: str = ": "):
            return f"\t{name}={{" + dict_to_str(value, delim=delim) + "},\n"

        # Build representations
        name_repr = repr_var("name", self.name)
        if len(self.__registered["args"]) > 0:
            args_repr = repr_dict_var("args", self.__registered["args"])
        else:
            args_repr = ""
        inputs_repr = repr_dict_var("inputs", self.__registered["inputs"])
        outputs_repr = repr_dict_var("outputs", self.output_name_mapping, delim=" => ")
        output_repr = repr_var("output", self.__output)
        progress_repr = repr_var("progress", self.__get_progress_string())
        return (
            f"{type(self).__name__}(\n"
            f"{name_repr}"
            f"{inputs_repr}"
            f"{args_repr}"
            f"{outputs_repr}"
            f"{progress_repr}"
            f"{output_repr}"
            ")"
        )

    def __getstate__(self):
        state = self.__dict__.copy()

        # Remove any locks (we don't need the locks in a background process)
        state.pop("_lock", None)

        return state

    def __setstate__(self, state):  # pragma: no cover
        self.__dict__.update(state)


#############################
# Step utilities
#############################
def concat(
    *steps: Step,
    name: None | str = None,
    lazy: bool = True,
    progress_interval: None | int | Default = DEFAULT,
    force: bool = False,
    writer_batch_size: None | int = 1000,
    save_num_proc: None | int | Default = DEFAULT,
    save_num_shards: None | int | Default = DEFAULT,
    background: bool = False,
) -> Step:
    """
    Concatenate the rows of the outputs of the input steps.

    Args:
        steps: The steps to concatenate.
        name: The name of the operation.
        lazy: Whether to run the operation lazily.
        progress_interval: How often to log progress in seconds.
        force: Whether to force run the operation (ignore saved results).
        writer_batch_size: The batch size to use if saving to disk.
        save_num_proc: The number of processes to use if saving to disk.
        save_num_shards: The number of shards on disk to save the dataset into.
        background: Whether to run the operation in the background.

    Returns:
        A new step with the operation applied.
    """
    kwargs = dict(locals())
    steps = kwargs["steps"]
    del kwargs["steps"]

    kwargs["op_cls"] = ConcatStep
    kwargs["op_name"] = "concat"
    kwargs["axis"] = 0

    return __concatenate(*steps, **kwargs)


def zipped(
    *steps: Step,
    name: None | str = None,
    lazy: bool = True,
    progress_interval: None | int | Default = DEFAULT,
    force: bool = False,
    writer_batch_size: None | int = 1000,
    save_num_proc: None | int | Default = DEFAULT,
    save_num_shards: None | int | Default = DEFAULT,
    background: bool = False,
) -> Step:
    """Zip the outputs of the input steps. This is similar to the built-in Python
    :py:func:`zip` function, essentially concatenating the columns of the outputs of
    the input steps.

    Args:
        steps: The steps to zip.
        name: The name of the operation.
        lazy: Whether to run the operation lazily.
        progress_interval: How often to log progress in seconds.
        force: Whether to force run the operation (ignore saved results).
        writer_batch_size: The batch size to use if saving to disk.
        save_num_proc: The number of processes to use if saving to disk.
        save_num_shards: The number of shards on disk to save the dataset into.
        background: Whether to run the operation in the background.

    Returns:
        A new step with the operation applied.
    """
    kwargs = dict(locals())
    steps = kwargs["steps"]
    del kwargs["steps"]

    kwargs["op_cls"] = ZippedStep
    kwargs["op_name"] = "zipped"
    kwargs["axis"] = 1

    return __concatenate(*steps, **kwargs)


#############################
# Classes for step operations
#############################


class ConcatStep(Step):
    pass


class ZippedStep(Step):
    pass


class SelectStep(Step):
    pass


class SelectColumnsStep(Step):
    pass


class TakeStep(Step):
    pass


class SkipStep(Step):
    pass


class ShuffleStep(Step):
    pass


class SortStep(Step):
    pass


class AddItemStep(Step):
    pass


class MapStep(Step):
    pass


class FilterStep(Step):
    pass


class RenameColumnStep(Step):
    pass


class RenameColumnsStep(Step):
    pass


class RemoveColumnsStep(Step):
    pass


class SplitStep(Step):
    pass


class ShardStep(Step):
    pass


class ReverseStep(Step):
    pass


class SaveStep(Step):
    pass


class CopyStep(Step):
    pass


#############################
# Class for higher-order steps
#############################


class SuperStep(Step):  # pragma: no cover
    """The class to subclass if you want to create a step that runs other steps.
    See :doc:`create your own steps
    <pages/advanced_usage/creating_a_new_datadreamer_.../step>` for more details.
    """

    @property
    def output(self) -> OutputDataset | OutputIterableDataset:
        return super().output


__all__ = [
    "LazyRowBatches",
    "LazyRows",
    "StepOutputType",
    "concat",
    "zipped",
    "ConcatStep",
    "ZippedStep",
    "SelectStep",
    "SelectColumnsStep",
    "TakeStep",
    "SkipStep",
    "ShuffleStep",
    "SortStep",
    "AddItemStep",
    "MapStep",
    "FilterStep",
    "RenameColumnStep",
    "RenameColumnsStep",
    "RemoveColumnsStep",
    "SplitStep",
    "ShardStep",
    "ReverseStep",
    "SaveStep",
    "CopyStep",
    "SuperStep",
]
