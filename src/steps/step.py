import json
import os
from collections import defaultdict
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
from ..pickling import unpickle as _unpickle
from ..pickling.pickle import _INTERNAL_PICKLE_KEY, _pickle
from ..utils.fs_utils import clear_dir, safe_fn
from .step_output import LazyRowBatches, LazyRows, StepOutputType, _output_to_dataset


class StepProtector(type):
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


class Step(metaclass=StepProtector):
    def __init__(  # noqa: C901
        self,
        name: str,
        inputs: None
        | dict[str, OutputDatasetColumn | OutputIterableDatasetColumn] = None,
        args: None | dict[str, Any] = None,
        outputs: None | dict[str, str] = None,
    ):
        # Fill in default argument values
        if not isinstance(inputs, dict):
            inputs = {}
        if not isinstance(args, dict):
            args = {}
        if not isinstance(outputs, dict):
            outputs = {}
        assert inputs is not None
        assert args is not None
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
            "inputs": {},
            "args": {},
            "outputs": [],
            "trace_info": defaultdict(lambda: defaultdict(list)),
        }

        # Run setup
        self.setup()
        self._initialized = True

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

        # Validate and setup args
        if not set(args.keys()).issubset(set(self.__registered["args"].keys())):
            raise ValueError(
                f"Expected {set(self.__registered['args'].keys())} as args keys,"
                f" got {set(args.keys())}."
            )
        else:
            self.__registered["args"].update(args)

        # Initialize output names mapping
        if len(self.__registered["outputs"]) == 0:
            raise ValueError("The step must register at least one output.")
        if not set(outputs.keys()).issubset(set(self.__registered["outputs"])):
            raise ValueError(
                f"Expected {set(self.__registered['outputs'])} as output keys,"
                f" got {set(outputs.keys())}."
            )
        self.output_name_mapping: dict[str, str] = {
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

    def _save_to_disk(self):
        if self.__output_folder_path and isinstance(self.__output, OutputDataset):
            metadata_path = os.path.join(self.__output_folder_path, "step.json")
            dataset_path = os.path.join(self.__output_folder_path, "dataset")
            self.__output.save_to_disk(dataset_path)
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
        if prev_fingerprint == self.fingerprint:
            self.__output = OutputDataset(
                self, Dataset.load_from_disk(dataset_path), pickled=metadata["pickled"]
            )
            self.progress = 1.0
            self.__pickled = metadata["pickled"]
        elif prev_fingerprint != self.fingerprint and prev_fingerprint is not None:
            # ...but it was a different version, delete the results and we'll need
            # to re-run this step
            clear_dir(self.__output_folder_path)

    def register_input(self, input_column_name: str):
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if type(input_column_name) is not str:
            raise TypeError(f"Expected str, got {type(input_column_name)}.")
        if input_column_name not in self.__registered["inputs"]:
            self.__registered["inputs"][input_column_name] = None

    def register_arg(self, arg_name: str):
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if type(arg_name) is not str:
            raise TypeError(f"Expected str, got {type(arg_name)}.")
        self.__registered["args"][arg_name] = None

    def register_output(self, output_column_name: str):
        if self._initialized:
            raise RuntimeError(
                "The step is already initialized, you can only run"
                " .register_xxx() functions in the setup() method."
            )
        if type(output_column_name) is not str:
            raise TypeError(f"Expected str, got {type(output_column_name)}.")
        if output_column_name not in self.__registered["outputs"]:
            self.__registered["outputs"].append(output_column_name)

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

    def pickle(self, value: Any, *args: Any, **kwargs: Any) -> bytes:
        self.__pickled = True
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
    ):
        if self.__output:
            raise StepOutputError("Step has already been run.")
        self.__output = _output_to_dataset(
            step=self,
            output_names=tuple(self.__registered["outputs"]),
            output_name_mapping=self.output_name_mapping,
            set_progress=lambda progress: setattr(self, "progress", progress),
            pickled=self.__pickled,
            value=value,
        )
        self._save_to_disk()

    def head(self, n=5, shuffle=False, seed=None, buffer_size=1000) -> DataFrame:
        return self.output.head(
            n=n, shuffle=shuffle, seed=seed, buffer_size=buffer_size
        )

    @property
    def trace_info(self):
        return json.loads(json.dumps(self.__registered["trace_info"]))

    @property
    def fingerprint(self) -> str:
        return Hasher.hash(
            [
                str(type(self)),
                self.name,
                self.version,
                list(self.__registered["inputs"].keys()),
                list(self.__registered["args"].keys()),
                list(self.__registered["outputs"]),
            ]
        )

    def get_run_output_folder_path(self) -> str:
        if not self.__output_folder_path:
            raise RuntimeError(
                "You must run the Step in a DataDreamer() context."
            )  # pragma: no cover
        run_output_folder_path = os.path.join(self.__output_folder_path, "run_output")
        os.makedirs(run_output_folder_path, exist_ok=True)
        return run_output_folder_path


__all__ = ["LazyRowBatches", "LazyRows", "StepOutputType"]
