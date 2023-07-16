from typing import Any

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
        self.__output = _output_to_dataset(
            output_names=self.output_names,
            set_progress=lambda progress: setattr(self, "progress", progress),
            pickled=self.__pickled,
            value=value,
        )


__all__ = ["LazyRowBatches", "LazyRows", "StepOutputType"]
