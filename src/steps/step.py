import json
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from functools import cached_property, partial
from logging import Logger
from time import time
from typing import Any, Callable, Sequence

import dill
from pandas import DataFrame

from datasets import Dataset, DatasetDict
from datasets.fingerprint import Hasher

from .. import __version__
from ..datadreamer import DataDreamer
from ..datasets import (
    OutputDataset,
    OutputDatasetColumn,
    OutputIterableDataset,
    OutputIterableDatasetColumn,
)
from ..errors import StepOutputError
from ..logging import DATEFMT, STANDARD_FORMAT, logger
from ..pickling import unpickle as _unpickle
from ..pickling.pickle import _INTERNAL_PICKLE_KEY, _pickle
from ..project.environment import RUNNING_IN_PYTEST
from ..utils.background_utils import run_in_background_process_no_block
from ..utils.fs_utils import move_dir, safe_fn
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
    _create_take_step,
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
                if attribute == "__init__":
                    raise AttributeError(
                        'Overriding of "%s" not allowed, override setup() instead.'
                        % attribute
                    )
        meta.has_base = True
        klass = super().__new__(meta, name, bases, attrs)
        return klass

    @property
    def help(self) -> str:
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
    def __init__(  # noqa: C901
        self,
        name: str,
        args: None | dict[str, Any] = None,
        inputs: None
        | dict[str, OutputDatasetColumn | OutputIterableDatasetColumn] = None,
        outputs: None | dict[str, str] = None,
        progress_interval: None | int = 60,
        force: bool = False,
        verbose: bool = False,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ):
        # Check pid
        if DataDreamer.is_background_process():  # pragma: no cover
            raise RuntimeError(
                f"Steps must be initialized in the same process"
                f" ({os.getpid()}) as the DataDreamer() context manager"
                f" ({DataDreamer.ctx.pid}). Use background=True if you want to"
                " run this step in a background process."
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
        self.name: str = name
        self.version: float = 1.0
        if len(self.name) == 0:
            raise ValueError("You must provide a name for the step.")
        if DataDreamer.initialized():
            DataDreamer._add_step(self)
        self.__progress: None | float = None
        self.__progress_rows: None | int = None
        self.__progress_logging_rows: bool = False
        self.progress_interval: None | int = progress_interval
        self.progress_last = time()
        self.__output: None | OutputDataset | OutputIterableDataset = None
        self._pickled: bool = False
        self.__registered: dict[str, Any] = {
            "args": {},
            "inputs": {},
            "outputs": [],
            "trace_info": defaultdict(lambda: defaultdict(list)),
        }
        self.output_name_mapping = {}
        self.__help: dict[str, Any] = {
            "args": {},
            "inputs": {},
            "outputs": {},
        }
        self.force = force
        self.save_num_proc = save_num_proc
        self.save_num_shards = save_num_shards
        self.background = background

        # Initialize the logger
        self.verbose = verbose
        self.log_level = log_level
        self.logger: Logger
        if not hasattr(self.__class__, _INTERNAL_HELP_KEY):
            stderr_handler = logging.StreamHandler()
            stderr_handler.setLevel(logging.DEBUG)
            self.logger = logging.getLogger(f"datadreamer.steps.{safe_fn(self.name)}")
            if RUNNING_IN_PYTEST:
                self.logger.propagate = True
            else:
                self.logger.propagate = False  # pragma: no cover
            log_format: str = (
                logger.handlers[0].formatter and logger.handlers[0].formatter._fmt
            ) or STANDARD_FORMAT
            log_format = log_format.replace(
                "%(message)s", f"[ ➡️ {self.name}] %(message)s"
            )
            formatter = logging.Formatter(log_format, datefmt=DATEFMT, validate=False)
            stderr_handler.setFormatter(formatter)
            self.logger.addHandler(stderr_handler)
            if self.verbose:
                self.logger.setLevel(self.log_level or max(logger.level, logging.INFO))
            else:
                self.logger.setLevel(logging.CRITICAL + 1)

        # Run setup
        self.setup()
        if hasattr(self.__class__, _INTERNAL_HELP_KEY):
            return
        self._initialized = True

        # Validate and setup args
        if not set(args.keys()).issubset(set(self.__registered["args"].keys())):
            raise ValueError(
                f"Expected {set(self.__registered['args'].keys())} as args keys,"
                f" got {set(args.keys())}."
            )
        else:
            self.__registered["args"].update(args)

        # Validate and setup inputs
        if set(self.__registered["inputs"].keys()) != set(inputs.keys()):
            raise ValueError(
                f"Expected {set(self.__registered['inputs'].keys())} as inputs keys,"
                f" got {set(inputs.keys())}."
            )
        elif not all(
            [
                isinstance(v, (OutputDatasetColumn, OutputIterableDatasetColumn))
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
            prev_trace_info = {}
            for v in inputs.values():
                prev_trace_info.update(v.step.trace_info)
            prev_trace_info.update(self.__registered["trace_info"])
            self.__registered["trace_info"] = prev_trace_info

        # Initialize output names mapping
        if len(self.__registered["outputs"]) == 0 and not hasattr(
            self.__class__, _INTERNAL_STEP_OPERATION_KEY
        ):
            raise ValueError("The step must register at least one output.")
        if not set(outputs.keys()).issubset(set(self.__registered["outputs"])):
            raise ValueError(
                f"Expected {set(self.__registered['outputs'])} as output keys,"
                f" got {set(outputs.keys())}."
            )
        self.output_name_mapping = {
            o: outputs.get(o, o) for o in self.__registered["outputs"]
        }
        self.output_names = tuple(
            [self.output_name_mapping[o] for o in self.__registered["outputs"]]
        )

        # Run (or resume) within the DataDreamer context
        self._output_folder_path: None | str = None
        if DataDreamer.initialized():
            self.__setup_folder_and_resume()

    def __setup_folder_and_resume(self):
        if DataDreamer.is_running_in_memory():
            self.__start()
            return

        # Create an output folder for the step
        self._output_folder_path = os.path.join(
            DataDreamer.get_output_folder_path(), safe_fn(self.name)
        )
        os.makedirs(self._output_folder_path, exist_ok=True)
        assert self._output_folder_path is not None

        # Check if we have already run this step previously and saved the results to
        # disk
        metadata_path = os.path.join(self._output_folder_path, "step.json")
        dataset_path = os.path.join(self._output_folder_path, "dataset")
        prev_fingerprint: None | str = None
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                prev_fingerprint = metadata["fingerprint"]
        except FileNotFoundError:
            pass

        # We have already run this step
        if prev_fingerprint == self.fingerprint and not self.force:
            self.__output = OutputDataset(
                self, Dataset.load_from_disk(dataset_path), pickled=metadata["pickled"]
            )
            self.progress = 1.0
            self._pickled = metadata["pickled"]
            self._resumed = True
            logger.info(
                f"Step '{self.name}' results loaded from disk. 🙌 It was previously run"
                " and saved."
            )
        elif prev_fingerprint is not None and (
            prev_fingerprint != self.fingerprint or self.force
        ):
            # ...but it was a different version, delete the results and we'll need
            # to re-run this step
            logger.info(
                f"Step '{self.name}' was previously run and saved, but was outdated. 😞"
                " It will be re-run."
            )
            backup_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".backups",
                safe_fn(self.name),
                prev_fingerprint,
            )
            logger.debug(
                f"Step '{self.name}''s outdated results are being backed up: {backup_path}"
            )
            move_dir(self._output_folder_path, backup_path)
            logger.debug(
                f"Step '{self.name}''s outdated results are backed up: {backup_path}"
            )
            self.__start()
        else:
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
            logger.info(f"Step '{self.name}' finished and is saved to disk. 🎉")

        # Set output_names and output_name_mapping if step operation
        if hasattr(self.__class__, _INTERNAL_STEP_OPERATION_KEY) and self.__output:
            self.output_name_mapping = {n: n for n in self.__output.column_names}
            self.output_names = tuple([o for o in self.output_name_mapping.values()])

    def __save_output_to_disk(self, output: OutputDataset):
        if not self._output_folder_path:  # pragma: no cover
            return
        logger.debug(
            f"Step '{self.name}' is being saved to disk: {self._output_folder_path}."
        )
        metadata_path = os.path.join(self._output_folder_path, "step.json")
        dataset_path = os.path.join(self._output_folder_path, "dataset")
        output.save_to_disk(
            dataset_path,
            num_proc=self.save_num_proc,
            num_shards=self.save_num_shards,
        )
        with open(metadata_path, "w+") as f:
            json.dump(
                {
                    "__version__": __version__,
                    "datetime": datetime.now().isoformat(),
                    "type": type(self).__name__,
                    "name": self.name,
                    "version": self.version,
                    "fingerprint": self.fingerprint,
                    "pickled": output._pickled,
                    "trace_info": self.trace_info,
                },
                f,
                indent=4,
            )
        logger.debug(
            f"Step '{self.name}' is now saved to disk: {self._output_folder_path}."
        )

    def register_arg(self, arg_name: str, help: None | str = None):
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if type(arg_name) is not str:
            raise TypeError(f"Expected str, got {type(arg_name)}.")
        self.__registered["args"][arg_name] = None
        self.__help["args"][arg_name] = help

    def register_input(self, input_column_name: str, help: None | str = None):
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if type(input_column_name) is not str:
            raise TypeError(f"Expected str, got {type(input_column_name)}.")
        if input_column_name not in self.__registered["inputs"]:
            self.__registered["inputs"][input_column_name] = None
        self.__help["inputs"][input_column_name] = help

    def register_output(self, output_column_name: str, help: None | str = None):
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if type(output_column_name) is not str:
            raise TypeError(f"Expected str, got {type(output_column_name)}.")
        if output_column_name not in self.__registered["outputs"]:
            self.__registered["outputs"].append(output_column_name)
        self.__help["outputs"][output_column_name] = help

    def register_trace_info(self, trace_info_type: str, trace_info: Any):
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if type(trace_info_type) is not str:
            raise TypeError(f"Expected str, got {type(trace_info_type)}.")
        self.__registered["trace_info"][self.name][trace_info_type].append(trace_info)

    @property
    def args(self) -> dict[str, Any]:
        return self.__registered["args"]

    @property
    def inputs(self) -> dict[str, OutputDatasetColumn | OutputIterableDatasetColumn]:
        return self.__registered["inputs"]

    def setup(self):
        raise NotImplementedError("You must implement the .setup() method in Step.")

    def run(self) -> StepOutputType | LazyRows | LazyRowBatches:
        raise NotImplementedError("You must implement the .run() method in Step.")

    def get_run_output_folder_path(self) -> str:
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
        self._pickled = True
        if self.__output:
            self.output._pickled = True
        kwargs[_INTERNAL_PICKLE_KEY] = True
        return _pickle(value, *args, **kwargs)

    def unpickle(self, value: bytes) -> Any:
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

    @property
    def progress(self) -> None | float:
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
            self.progress_interval is not None
            and (time() - self.progress_last) > self.progress_interval
            and value > prev_progress
            and (not self.__progress_logging_rows or value < 1.0)
        ):
            should_log = True
            self.progress_last = time()
        self.__progress = value
        if should_log:
            logger.info(
                f"Step '{self.name}' progress:" f" {self.__get_progress_string()} 🔄"
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

    def _set_output(  # noqa: C901
        self,
        value: StepOutputType | LazyRows | LazyRowBatches,
        background_run_func: None | Callable = None,
    ):
        if self.__output:
            raise StepOutputError("Step has already been run.")
        logger.debug(f"Step '{self.name}''s results are being processed.")
        if background_run_func:
            _monkey_patch_iterable_dataset_apply_feature_types()

            def with_result_process(process):
                DataDreamer._add_process(process)

            def with_result(self, output):
                self.__output = dill.loads(output)
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
        return self.output.head(
            n=n, shuffle=shuffle, seed=seed, buffer_size=buffer_size
        )

    @cached_property
    def trace_info(self):
        return json.loads(json.dumps(self.__registered["trace_info"]))

    def select(
        self,
        indices: Iterable,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_select_step, **kwargs)()

    def select_columns(
        self,
        column_names: str | list[str],
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_select_columns_step, **kwargs)()

    def take(
        self,
        n: int,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_take_step, **kwargs)()

    def skip(
        self,
        n: int,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
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
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
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
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_sort_step, **kwargs)()

    def add_item(
        self,
        item: dict,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
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
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
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
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
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
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_rename_column_step, **kwargs)()

    def rename_columns(
        self,
        column_mapping: dict[str, str],
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_rename_columns_step, **kwargs)()

    def remove_columns(
        self,
        column_names: str | list[str],
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_remove_columns_step, **kwargs)()

    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = False,
        name: None | str = None,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_shard_step, **kwargs)()

    def reverse(
        self,
        name: None | str = None,
        lazy: bool = True,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_reverse_step, **kwargs)()

    def save(
        self,
        name: None | str = None,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
        kwargs["step"] = self
        del kwargs["self"]
        return partial(_create_save_step, **kwargs)()

    def copy(
        self,
        name: None | str = None,
        progress_interval: None | int = None,
        force: bool = False,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ) -> "Step":
        kwargs = dict(locals())
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
        pass

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
        pass

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
    ) -> str:
        pass

    def export_to_csv(
        self,
        path: str,
        train_size: None | float | int = None,
        validation_size: None | float | int = None,
        test_size: None | float | int = None,
        stratify_by_column: None | str = None,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        **to_csv_kwargs,
    ) -> str:
        pass

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
        pass

    def publish_to_hf(
        self,
        repo_id: str,
        branch: None | str,
        private: bool = False,
        token: None | str = None,
        train_size: None | float | int = None,
        validation_size: None | float | int = None,
        test_size: None | float | int = None,
        stratify_by_column: None | str = None,
        writer_batch_size: None | int = 1000,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
    ) -> str:
        pass

    @cached_property
    def fingerprint(self) -> str:
        return Hasher.hash(
            [
                str(type(self).__name__),
                self.name,
                self.version,
                self.__registered["args"],
                list(self.__registered["inputs"].keys()),
                list(
                    [c.step.fingerprint for c in self.__registered["inputs"].values()]
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
            return f"\t{name}={repr(value)},\n"

        def repr_dict_var(name: str, value: dict, delim: str = ": "):
            return f"\t{name}={{" + dict_to_str(value, delim=delim) + "},\n"

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
            f"{args_repr}"
            f"{inputs_repr}"
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
            f"{args_repr}"
            f"{inputs_repr}"
            f"{outputs_repr}"
            f"{progress_repr}"
            f"{output_repr}"
            ")"
        )


#############################
# Step utilities
#############################
def concat(
    *steps: Step,
    name: None | str = None,
    lazy: bool = True,
    progress_interval: None | int = None,
    force: bool = False,
    writer_batch_size: None | int = 1000,
    save_num_proc: None | int = None,
    save_num_shards: None | int = None,
    background: bool = False,
) -> Step:
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
    progress_interval: None | int = None,
    force: bool = False,
    writer_batch_size: None | int = 1000,
    save_num_proc: None | int = None,
    save_num_shards: None | int = None,
    background: bool = False,
) -> Step:
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


class ShardStep(Step):
    pass


class ReverseStep(Step):
    pass


class SaveStep(Step):
    pass


class CopyStep(Step):
    pass


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
    "ShardStep",
    "ReverseStep",
    "SaveStep",
    "CopyStep",
]
