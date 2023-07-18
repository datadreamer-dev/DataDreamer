import os
from collections import defaultdict
from typing import Any

from pandas import DataFrame

from datasets import Dataset

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
from ..utils.class_utils import protect
from ..utils.fs_utils import safe_fn
from .step_output import LazyRowBatches, LazyRows, StepOutputType, _output_to_dataset


class Step(metaclass=protect("__init__")):  # type:ignore[misc]
    def __init__(
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

        # Initialize variables
        self.name: str = name
        if len(self.name) == 0:
            raise ValueError("You provide a name for the step.")
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
        if set(self.__registered["inputs"].keys()) != set(inputs.keys()):
            raise ValueError(
                f"Expected {set(self.__registered['inputs'].keys())} as inputs keys,"
                f" got {set(inputs.keys())}."
            )
        else:
            self.__registered["inputs"] = inputs
        if not set(args.keys()).issubset(set(self.__registered["args"].keys())):
            raise ValueError(
                f"Expected {set(self.__registered['args'].keys())} as args keys,"
                f" got {set(args.keys())}."
            )
        else:
            self.__registered["args"].update(args)
        if len(self.__registered["outputs"]) == 0:
            raise ValueError("The step must register at least one output.")

        # Initialize output names
        self.output_name_mapping = {o: o for o in self.__registered["outputs"]}
        if not set(outputs.keys()).issubset(set(self.__registered["outputs"])):
            raise ValueError(
                f"Expected {set(self.__registered['outputs'])} as output keys,"
                f" got {set(outputs.keys())}."
            )
        self.output_names = tuple(
            [self.output_name_mapping[o] for o in self.__registered["outputs"]]
        )

        # Initialize from context
        self.output_folder_path = None
        if hasattr(DataDreamer.ctx, "steps"):
            # Register the Step in the context
            for step in DataDreamer.ctx.steps:
                if step.name == self.name or safe_fn(step.name) == safe_fn(self.name):
                    raise ValueError(
                        f"A step already exists with the name: {self.name}"
                    )
            DataDreamer.ctx.steps.append(self)

            # Create an output folder for the step
            self.output_folder_path = os.path.join(
                DataDreamer.ctx.output_folder_path, safe_fn(self.name)
            )
            os.makedirs(self.output_folder_path)

    def register_input(self, input_column_name: str):
        if type(input_column_name) is not str:
            raise ValueError(f"Expected str, got {type(input_column_name)}.")
        if input_column_name not in self.__registered["inputs"]:
            self.__registered["inputs"][input_column_name] = None

    def register_arg(self, arg_name: str):
        if type(arg_name) is not str:
            raise ValueError(f"Expected str, got {type(arg_name)}.")
        self.__registered["args"][arg_name] = None

    def register_output(self, output_column_name: str):
        if type(output_column_name) is not str:
            raise ValueError(f"Expected str, got {type(output_column_name)}.")
        if output_column_name not in self.__registered["outputs"]:
            self.__registered["outputs"].append(output_column_name)

    def register_trace_info(self, trace_info_type: str, trace_info: Any):
        if type(trace_info_type) is not str:
            raise ValueError(f"Expected str, got {type(trace_info_type)}.")
        self.__registered["trace_info"][trace_info_type][self.name].append(trace_info)

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
            output_names=self.output_names,
            set_progress=lambda progress: setattr(self, "progress", progress),
            pickled=self.__pickled,
            value=value,
        )

    def head(self, n=5, shuffle=False, seed=None, buffer_size=1000) -> DataFrame:
        return self.output.head(
            n=n, shuffle=shuffle, seed=seed, buffer_size=buffer_size
        )

    @property
    def trace_info(self):
        return self.__registered["trace_info"]


__all__ = ["LazyRowBatches", "LazyRows", "StepOutputType"]
