from collections.abc import Generator, Iterable
from functools import partial
from typing import Any, Callable, Iterator, Optional, TypeGuard, Union

from datasets import Dataset, IterableDataset
from datasets.features.features import Features

from ..datasets.utils import dataset_zip, get_column_names, iterable_dataset_zip


def _is_list_or_tuple_type(v) -> TypeGuard[Union[list, tuple]]:
    return isinstance(v, list) or isinstance(v, tuple)


def _is_dataset_type(v) -> TypeGuard[Union[Dataset, IterableDataset]]:
    return isinstance(v, Dataset) or isinstance(v, IterableDataset)


def _iterable_or_generator_to_iterable(
    v: Union[Iterable[Any], Callable[[], Generator[Any, None, None]]]
) -> Iterator[Any]:
    if callable(v):
        return v()
    else:
        return iter(v)


class Step:
    def __init__(
        self,
        name: str,
        input: Optional[
            Union[
                Dataset,
                IterableDataset,
            ]
        ],
        outputs: Union[str, list[str], tuple[str, ...]],
    ):
        self._name: str = name
        self.__progress: Optional[float] = None
        self.input = input
        self.__output: Optional[Union[Dataset, IterableDataset]] = None
        if _is_list_or_tuple_type(outputs) and len(outputs) == 0:
            raise ValueError("The step must name its outputs")
        self.output_names: tuple[str, ...]
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            self.output_names = tuple(outputs)
        else:
            self.output_names = (outputs,)

    @property
    def progress(self) -> Optional[float]:
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
    def output(self) -> Union[Dataset, IterableDataset]:
        if self.__output is None:
            if self.__progress is None:
                raise AttributeError("Step has not been run. Output is not available.")
            else:
                raise AttributeError(
                    f"Step is still running ({self.__get_progress_string()})."
                    " Output is not available yet."
                )
        else:
            return self.__output

    # TODO: Generator function of batches

    def _set_output(  # noqa: C901
        self,
        value: Union[
            Dataset,
            IterableDataset,
            dict[str, Union[Iterable[Any], Callable[[], Generator[Any, None, None]]]],
            list[Any],
            tuple[Union[Iterable[Any], Callable[[], Generator[Any, None, None]]], ...],
            Callable[
                [],
                Generator[
                    dict[str, Union[Iterable[Any], list[Any], tuple[Any, ...]]],
                    None,
                    None,
                ],
            ],
        ],
    ):
        # Set progress to 0.0
        self.progress = 0.0

        # Create a Dataset if given a list or tuple of Datasets
        # or create an IterableDataset if given a list or tuple of IterableDatasets
        if _is_list_or_tuple_type(value):
            if all(isinstance(d, Dataset) for d in value):
                value = dataset_zip(*value)
            elif all(_is_dataset_type(d) for d in value):
                value = iterable_dataset_zip(*value)

        # Create a Dataset/generator function if given a dict
        if isinstance(value, dict) and set(self.output_names) == set(value.keys()):
            if any([callable(v) for v in value.values()]):
                # One of the values of the dictionary is a generator function,
                # create a generator function of dicts
                iters = [
                    _iterable_or_generator_to_iterable(value[k])
                    for k in self.output_names
                ]
                rows = zip(*iters)

                def to_dict_generator_wrapper(rows, output_names):
                    for row in rows:
                        yield {k: v for k, v in zip(output_names, row)}

                value = partial(to_dict_generator_wrapper, rows, self.output_names)
            else:
                value = Dataset.from_dict({k: value[k] for k in self.output_names})
        elif isinstance(value, dict):
            raise AttributeError(
                f"Expected {self.output_names} dict keys instead of {list(value.keys())}."
            )

        # If given a single list when more than one output force it into a tuple
        if (
            isinstance(value, list)
            and len(self.output_names) > 1
            and len(value) == len(self.output_names)
            and [isinstance(v, Iterable) for v in value]
        ):
            value = tuple(value)
        elif (
            isinstance(value, list)
            and len(self.output_names) > 1
            and _is_list_or_tuple_type(value[0])
            and len(value[0]) == len(self.output_names)
        ):
            value = tuple(zip(*value))

        # If given a single list
        if isinstance(value, list) and len(self.output_names) > 1:
            raise AttributeError(
                f"Expected {len(self.output_names)} outputs ({self.output_names})"
            )

        # If given a tuple with the wrong number of elements
        if isinstance(value, tuple) and len(self.output_names) != len(value):
            raise AttributeError(
                f"Expected {len(self.output_names)} outputs ({self.output_names})"
            )

        # Create a generator function if given a tuple with a generator function
        if isinstance(value, tuple) and any([callable(v) for v in value]):
            iters = [_iterable_or_generator_to_iterable(v) for v in value]
            rows = zip(*iters)

            def to_dict_generator_wrapper(rows, output_names):
                for row in rows:
                    yield {k: v for k, v in zip(output_names, row)}

            value = partial(to_dict_generator_wrapper, rows, self.output_names)

        # If given a Dataset with the wrong number of
        if _is_dataset_type(value) and set(get_column_names(value)) != set(
            self.output_names
        ):
            raise AttributeError(
                f"Expected {self.output_names} columns instead of {get_column_names(value)}."
            )

        # If IterableDataset with no columns convert to a generator function
        if isinstance(value, IterableDataset) and value.column_names is None:

            def to_dict_generator_wrapper(value, output_names):
                return iter(value)

            value = partial(to_dict_generator_wrapper, value, self.output_names)

        # Create an IterableDataset if given a generator function of dicts
        if callable(value):
            # Make sure the generator returns a dict and the keys are correct
            try:
                first_row = next(value())
                if isinstance(first_row, dict) and set(self.output_names) != set(
                    first_row.keys()
                ):
                    raise AttributeError(
                        f"Expected {self.output_names} dict keys from generator"
                        f" function instead of {list(first_row.keys())}."
                    )
                elif _is_list_or_tuple_type(first_row):
                    if len(first_row) != len(self.output_names):
                        raise AttributeError(
                            f"Expected {len(self.output_names)} outputs"
                            " ({self.output_names}) from generator function"
                        )

                    def to_dict_generator_wrapper(value, output_names):
                        for row in value():
                            yield {k: v for k, v in zip(output_names, row)}

                    value = partial(to_dict_generator_wrapper, value, self.output_names)
            except StopIteration:
                pass

            # If so, convert the generator to an IterableDataset
            features = Features([(n, None) for n in self.output_names])

            # Wrap the generator so that we can set progress = 1.0 when complete
            def generator_wrapper(value):
                total_row_count = None
                for i, row in enumerate(value()):
                    yield row
                    if total_row_count is not None:
                        self.progress = (i + 1) / total_row_count
                self.progress = 1.0

            value = partial(generator_wrapper, value)
            value = IterableDataset.from_generator(value, features=features)

        if _is_dataset_type(value):
            self.__output = value
            if isinstance(value, Dataset):
                self.progress = 1.0
        elif isinstance(value, tuple):
            self.__output = Dataset.from_dict(
                {k: v for k, v in zip(self.output_names, value)}
            )
            self.progress = 1.0
        elif isinstance(value, list):
            self.__output = Dataset.from_dict({self.output_names[0]: value})
            self.progress = 1.0
