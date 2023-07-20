import json
import logging
import os
from collections import defaultdict
from functools import cached_property, partial
from logging import Logger
from typing import Any

from pandas import DataFrame

from datasets import Dataset
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
from ..utils.fs_utils import move_dir, safe_fn
from .step_output import LazyRowBatches, LazyRows, StepOutputType, _output_to_dataset

_INTERNAL_HELP_KEY = "__DataDreamer__help__"


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
        force: bool = False,
        verbose: bool = False,
        log_level: None | int = None,
    ):
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
        self.name: str = name
        self.version: float = 1.0
        if len(self.name) == 0:
            raise ValueError("You must provide a name for the step.")
        self.__progress: None | float = None
        self.__output: None | OutputDataset | OutputIterableDataset = None
        self.__pickled: bool = False
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

        # Initialize the logger
        self.logger: Logger
        if not hasattr(self.__class__, _INTERNAL_HELP_KEY):
            stderr_handler = logging.StreamHandler()
            stderr_handler.setLevel(logging.DEBUG)
            self.logger = logging.getLogger(f"datadreamer.steps.{self.name}")
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
            if verbose:
                self.logger.setLevel(log_level or max(logger.level, logging.INFO))
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
        if len(self.__registered["outputs"]) == 0:
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

        # Initialize from/to the context
        self.__output_folder_path = None
        if hasattr(DataDreamer.ctx, "initialized"):
            # Register the Step in the context
            for step in DataDreamer.ctx.steps:
                if step.name == self.name or safe_fn(step.name) == safe_fn(self.name):
                    raise ValueError(
                        f"A step already exists with the name: {self.name}"
                    )
            DataDreamer.ctx.steps.append(self)
            self.__output_folder_path = os.path.join(
                DataDreamer.ctx.output_folder_path, safe_fn(self.name)
            )
            self._setup_folder_and_resume()

    def _save_to_disk(self, num_proc: None | int = None):
        if not self.__output_folder_path:
            return
        if isinstance(self.__output, OutputDataset):
            logger.debug(
                f"Step '{self.name}' is being saved to disk: {self.__output_folder_path}."
            )
            metadata_path = os.path.join(self.__output_folder_path, "step.json")
            dataset_path = os.path.join(self.__output_folder_path, "dataset")
            self.__output.save_to_disk(dataset_path, num_proc=num_proc)
            with open(metadata_path, "w+") as f:
                json.dump(
                    {
                        "__version__": __version__,
                        "fingerprint": self.fingerprint,
                        "pickled": self.__output._pickled,
                    },
                    f,
                    indent=4,
                )
            logger.debug(
                f"Step '{self.name}' is now saved to disk: {self.__output_folder_path}."
            )
            logger.info(f"Step '{self.name}' finished and is saved to disk. 🎉")
        elif isinstance(self.__output, OutputIterableDataset):
            logger.info(f"Step '{self.name}' will run lazily. 🥱")

    def _setup_folder_and_resume(self):
        if self.__output_folder_path is None:
            return  # pragma: no cover

        # Create an output folder for the step
        self.__output_folder_path = os.path.join(
            DataDreamer.ctx.output_folder_path, safe_fn(self.name)
        )
        os.makedirs(self.__output_folder_path, exist_ok=True)

        # Check if we have already run this step previously and saved the results to
        # disk
        metadata_path = os.path.join(self.__output_folder_path, "step.json")
        dataset_path = os.path.join(self.__output_folder_path, "dataset")
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
            self.__pickled = metadata["pickled"]
            logger.info(
                f"Step '{self.name}' results loaded from disk. 🙌 It was previously run"
                " and saved."
            )
            return
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
                DataDreamer.ctx.output_folder_path,
                ".backups",
                safe_fn(self.name),
                prev_fingerprint,
            )
            logger.debug(
                f"Step '{self.name}''s outdated results are being backed up: {backup_path}"
            )
            move_dir(self.__output_folder_path, backup_path)
            logger.debug(
                f"Step '{self.name}''s outdated results are backed up: {backup_path}"
            )

        # We still need to run this step
        logger.info(f"Step '{self.name}' is running. ⏳")

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

    def setup(self):
        raise NotImplementedError("You must implement the .setup() method in Step.")

    def get_run_output_folder_path(self) -> str:
        if not self.__output_folder_path:
            raise RuntimeError(
                "You must run the Step in a DataDreamer() context."
            )  # pragma: no cover
        run_output_folder_path = os.path.join(self.__output_folder_path, "run_output")
        os.makedirs(run_output_folder_path, exist_ok=True)
        return run_output_folder_path

    def pickle(self, value: Any, *args: Any, **kwargs: Any) -> bytes:
        self.__pickled = True
        if self.__output:
            self.output._pickled = True
        kwargs[_INTERNAL_PICKLE_KEY] = True
        return _pickle(value, *args, **kwargs)

    def unpickle(self, value: bytes) -> Any:
        return _unpickle(value)

    @property
    def progress(self) -> None | float:
        return self.__progress

    @progress.setter
    def progress(self, value: float):
        if isinstance(self.__output, Dataset):
            value = 1.0
        self.__progress = max(min(value, 1.0), self.__progress or 0.0)

    def __get_progress_string(self):
        if self.__progress is not None:
            progress_int = int(self.__progress * 100)
            return f"{progress_int}%"
        else:
            return "0%"

    @property
    def output(self) -> OutputDataset | OutputIterableDataset:
        if self.__output is None:
            if self.__progress is None:
                raise StepOutputError("Step has not been run. Output is not available.")
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
        num_proc: None | int = None,
    ):
        if self.__output:
            raise StepOutputError("Step has already been run.")
        logger.debug(f"Step '{self.name}''s results are being processed.")
        self.__output = _output_to_dataset(
            step=self,
            output_names=tuple(self.__registered["outputs"]),
            output_name_mapping=self.output_name_mapping,
            set_progress=partial(
                lambda self, progress: setattr(self, "progress", progress), self
            ),
            get_pickled=partial(lambda self: self.__pickled, self),
            value=value,
        )
        self._save_to_disk(num_proc=num_proc)

    def head(self, n=5, shuffle=False, seed=None, buffer_size=1000) -> DataFrame:
        return self.output.head(
            n=n, shuffle=shuffle, seed=seed, buffer_size=buffer_size
        )

    @cached_property
    def trace_info(self):
        return json.loads(json.dumps(self.__registered["trace_info"]))

    def save(
        self,
        name: None | str = None,
        writer_batch_size: None | int = 1000,
        num_proc: None | int = None,
    ) -> "Step":
        final_name = name or f"{self.name}_save"

        def create_save_step(
            self: Step,
            name: str,
            output: OutputDataset | OutputIterableDataset,
            writer_batch_size: None | int,
            num_proc: None | int,
        ) -> Step:
            class SaveStep(Step):
                def setup(self):
                    for column_name in output.column_names:
                        self.register_output(column_name)

            save_step = SaveStep(name=name)
            if isinstance(output, OutputDataset):
                save_step._set_output(output, num_proc=num_proc)
            else:

                def dataset_generator(dataset):
                    yield from dataset

                logger.debug(
                    f"Iterating through all of '{self.name}''s lazy results in"
                    " preparation for saving."
                )
                dataset = Dataset.from_generator(
                    partial(dataset_generator, output.dataset),
                    features=output._features,
                    writer_batch_size=writer_batch_size,
                    num_proc=num_proc,
                )
                save_step.__pickled = output._pickled
                save_step._set_output(dataset)
            return save_step

        return partial(
            create_save_step,
            self=self,
            name=final_name,
            output=self.output,
            writer_batch_size=writer_batch_size,
            num_proc=num_proc,
        )()

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
        return (
            f"{type(self).__name__}(\n"
            f"{name_repr}"
            f"{args_repr}"
            f"{inputs_repr}"
            f"{outputs_repr}"
            f"{output_repr}"
            ")"
        )


__all__ = ["LazyRowBatches", "LazyRows", "StepOutputType"]
