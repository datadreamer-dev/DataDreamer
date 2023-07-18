from collections import defaultdict
from typing import Any

from pandas import DataFrame

from datasets import Dataset, IterableDataset

from ..datasets import OutputDataset, OutputIterableDataset
from ..errors import StepOutputError
from ..pickling import unpickle as _unpickle
from ..pickling.pickle import _INTERNAL_PICKLE_KEY, _pickle
from .step_output import (
    LazyRowBatches,
    LazyRows,
    StepOutputType,
    _is_list_or_tuple_type,
    _output_to_dataset,
)


class Step:
    def __init__(
        self,
        name: str,
        input: None | Dataset | IterableDataset,
        outputs: str | list[str] | tuple[str, ...],
    ):
        self.name: str = name
        self.__progress: None | float = None
        self.__registered: dict[str, Any] = {
            "inputs": [],
            "outputs": [],
            "args": {},
            "trace_info": defaultdict(lambda: defaultdict(list)),
        }
        self.input = input
        self.__output: None | OutputDataset | OutputIterableDataset = None
        self.__pickled: bool = False
        if _is_list_or_tuple_type(outputs) and len(outputs) == 0:
            raise ValueError("The step must name its outputs.")
        self.output_names: tuple[str, ...]
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            self.output_names = tuple(outputs)
        else:
            self.output_names = (outputs,)

    def register_input(self, input_column_name: str):
        if input_column_name not in self.__registered["inputs"]:
            self.__registered["inputs"].append(input_column_name)

    def register_output(self, output_column_name: str):
        if output_column_name not in self.__registered["outputs"]:
            self.__registered["outputs"].append(output_column_name)

    def register_arg(self, arg_name: str):
        self.__registered["args"][arg_name] = None

    def register_trace_info(self, trace_info_type: str, trace_info: Any):
        self.__registered["trace_info"][trace_info_type][self.name].append(trace_info)

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
