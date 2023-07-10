import warnings
from collections.abc import Generator, Iterable
from functools import partial
from typing import Any, Callable, Iterator, Optional, TypeAlias, TypeGuard, Union

from datasets import Dataset, IterableDataset
from datasets.features.features import Features

from ..datasets.utils import dataset_zip, get_column_names, iterable_dataset_zip


def _is_list_or_tuple_type(v) -> TypeGuard[Union[list, tuple]]:
    return isinstance(v, list) or isinstance(v, tuple)


def _is_dataset_type(v, is_lazy) -> TypeGuard[Union[Dataset, IterableDataset]]:
    if not is_lazy and isinstance(v, IterableDataset):
        raise AttributeError(
            "You must use LazyRows if you want to output an IterableDataset."
        )
    return isinstance(v, Dataset) or isinstance(v, IterableDataset)


def _iterable_or_generator_func_to_iterator(
    v: Union[Iterable[Any], Callable[[], Generator[Any, None, None]]],
    _value_is_batched: bool = False,
) -> Iterator[Any]:
    iterator: Iterator[Any]
    if callable(v):
        iterator = v()
    else:
        iterator = iter(v)
    if _value_is_batched:

        def unbatch(iterator: Any) -> Generator[Any, None, None]:
            for batch in iterator:
                if isinstance(batch, dict):
                    keys = list(batch.keys())
                    value_batch = [batch[k] for k in keys]
                    for values in zip(*value_batch):
                        yield {k: v for k, v in zip(keys, values)}
                else:
                    for v in batch:
                        yield v

        return partial(unbatch, iterator)()
    else:
        return iterator


LazyStepOutput: TypeAlias = Union[
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
]

LazyBatchStepOutput: TypeAlias = Union[
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
]

StepOutput: TypeAlias = Union[
    Dataset, dict[str, Iterable[Any]], list[Any], tuple[Iterable[Any], ...]
]


class LazyRows:
    def __init__(
        self,
        value: LazyStepOutput,
        total_num_rows: Optional[int] = None,
        auto_progress: bool = True,
    ) -> None:
        self.__value: LazyStepOutput = value
        self.total_num_rows: Optional[int] = total_num_rows
        if total_num_rows is None and auto_progress:
            warnings.warn(
                "You did not specify `total_num_rows`, so we cannot"
                " automatically update the progress % for this step. Either"
                " specify LazyRows(..., total_num_rows=#) or, to disable"
                " this warning, specify LazyRows(.., auto_progress = False)",
                stacklevel=2,
            )

    @property
    def value(self) -> LazyStepOutput:
        return self.__value


class LazyRowBatches:
    def __init__(
        self,
        value: LazyBatchStepOutput,
        total_num_rows: Optional[int] = None,
        auto_progress: bool = True,
    ) -> None:
        self.__value: LazyBatchStepOutput = value
        self.total_num_rows: Optional[int] = total_num_rows
        if total_num_rows is None and auto_progress:
            warnings.warn(
                "You did not specify `total_num_rows`, so we cannot"
                " automatically update the progress % for this step. Either"
                " specify LazyRowBatches(..., total_num_rows=#) or, to"
                " disable this warning, specify LazyRowBatches(..,"
                " auto_progress = False)",
                stacklevel=2,
            )

    @property
    def value(self) -> LazyBatchStepOutput:
        return self.__value


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
        value: Union[StepOutput, LazyRows, LazyRowBatches],
    ):
        # Set progress to 0.0
        self.progress = 0.0

        # Unpack LazyRows and LazyRowsBatches
        _value: Union[StepOutput, LazyStepOutput]
        is_lazy = False
        total_num_rows = None
        _value_is_batched = False
        if isinstance(value, LazyRows):
            _value = value.value
            is_lazy = True
            total_num_rows = value.total_num_rows
        elif isinstance(value, LazyRowBatches):
            _value = value.value
            is_lazy = True
            total_num_rows = value.total_num_rows
            if not _is_dataset_type(_value, is_lazy):
                _value_is_batched = True
        else:
            _value = value
        del value

        # Create a Dataset if given a list or tuple of Datasets
        # or create an IterableDataset if given a list or tuple of IterableDatasets
        if _is_list_or_tuple_type(_value):
            if all(isinstance(d, Dataset) for d in _value):
                _value = dataset_zip(*_value)
            elif all(_is_dataset_type(d, is_lazy) for d in _value):
                _value = iterable_dataset_zip(*_value)

        # Create a Dataset/generator function if given a dict
        if isinstance(_value, dict) and set(self.output_names) == set(_value.keys()):
            if is_lazy and any([callable(v) for v in _value.values()]):
                # One of the values of the dictionary is a generator function,
                # create a generator function of dicts
                def to_dict_generator_wrapper(_value, output_names, _value_is_batched):
                    iters = [
                        _iterable_or_generator_func_to_iterator(
                            _value[k], _value_is_batched
                        )
                        for k in output_names
                    ]
                    rows = zip(*iters)
                    for row in rows:
                        yield {k: v for k, v in zip(output_names, row)}

                _value = partial(
                    to_dict_generator_wrapper,
                    _value,
                    self.output_names,
                    _value_is_batched,
                )
                _value_is_batched = False
            else:
                _value = Dataset.from_dict({k: _value[k] for k in self.output_names})
        elif isinstance(_value, dict):
            raise AttributeError(
                f"Expected {self.output_names} dict keys instead of {list(_value.keys())}."
            )

        # If given a single list when more than one output force it into a tuple
        if (
            isinstance(_value, list)
            and len(self.output_names) > 1
            and _is_list_or_tuple_type(_value[0])
            and len(_value[0]) == len(self.output_names)
        ):
            _value = tuple(zip(*_value))
        elif (
            isinstance(_value, list)
            and len(self.output_names) > 1
            and len(_value) == len(self.output_names)
            and [isinstance(v, Iterable) for v in _value]
        ):
            _value = tuple(_value)
        elif (
            isinstance(_value, list)
            and len(self.output_names) == 1
            and len(_value) == len(self.output_names)
            and callable(_value[0])
        ):
            _value = tuple(_value)

        # If given a single list
        if isinstance(_value, list) and len(self.output_names) > 1:
            raise AttributeError(
                f"Expected {len(self.output_names)} outputs ({self.output_names})"
            )

        # If given a tuple with the wrong number of elements
        if isinstance(_value, tuple) and len(self.output_names) != len(_value):
            raise AttributeError(
                f"Expected {len(self.output_names)} outputs ({self.output_names})"
            )

        # Create a generator function if given a tuple with a generator function
        if isinstance(_value, tuple) and is_lazy and any([callable(v) for v in _value]):

            def to_dict_generator_wrapper(_value, output_names, _value_is_batched):
                iters = [
                    _iterable_or_generator_func_to_iterator(v, _value_is_batched)
                    for v in _value
                ]
                rows = zip(*iters)
                for row in rows:
                    yield {k: v for k, v in zip(output_names, row)}

            _value = partial(
                to_dict_generator_wrapper, _value, self.output_names, _value_is_batched
            )
            _value_is_batched = False

        # If given a Dataset with the wrong number of
        if _is_dataset_type(_value, is_lazy) and set(get_column_names(_value)) != set(
            self.output_names
        ):
            raise AttributeError(
                f"Expected {self.output_names} columns instead of {get_column_names(_value)}."
            )

        # If IterableDataset convert to a generator function
        if is_lazy and isinstance(_value, IterableDataset):

            def to_dict_generator_wrapper(_value, output_names, _value_is_batched):
                return iter(_value)

            _value = partial(
                to_dict_generator_wrapper, _value, self.output_names, _value_is_batched
            )

        # Create an IterableDataset if given a generator function of dicts
        if is_lazy and callable(_value):
            # Make sure the generator returns a dict and the keys are correct
            try:
                first_row = next(
                    _iterable_or_generator_func_to_iterator(_value, _value_is_batched)
                )
                if isinstance(first_row, dict) and set(self.output_names) != set(
                    first_row.keys()
                ):
                    raise AttributeError(
                        f"Expected {self.output_names} dict keys from generator"
                        f" function instead of {list(first_row.keys())}."
                    )
                elif _is_list_or_tuple_type(first_row):
                    if len(self.output_names) > 1 and len(first_row) == len(
                        self.output_names
                    ):

                        def to_dict_generator_wrapper(
                            _value, output_names, _value_is_batched
                        ):
                            for row in _iterable_or_generator_func_to_iterator(
                                _value, _value_is_batched
                            ):
                                yield {k: v for k, v in zip(output_names, row)}

                    elif len(self.output_names) == 1:

                        def to_dict_generator_wrapper(
                            _value, output_names, _value_is_batched
                        ):
                            for v in _iterable_or_generator_func_to_iterator(
                                _value, _value_is_batched
                            ):
                                yield {output_names[0]: v}

                    else:
                        raise AttributeError(
                            f"Expected {len(self.output_names)} outputs"
                            f" ({self.output_names}) from generator function"
                        )

                    _value = partial(
                        to_dict_generator_wrapper,
                        _value,
                        self.output_names,
                        _value_is_batched,
                    )
                    _value_is_batched = False
            except StopIteration:
                pass

            # If so, convert the generator to an IterableDataset
            features = Features([(n, None) for n in self.output_names])

            # Wrap the generator so that we can set progress = 1.0 when complete
            def generator_wrapper(_value, total_num_rows, _value_is_batched):
                for i, row in enumerate(
                    _iterable_or_generator_func_to_iterator(_value, _value_is_batched)
                ):
                    if total_num_rows is not None:
                        self.progress = (i + 1) / total_num_rows
                    yield row
                self.progress = 1.0

            _value = partial(
                generator_wrapper, _value, total_num_rows, _value_is_batched
            )
            _value_is_batched = False
            _value = IterableDataset.from_generator(_value, features=features)

        if _is_dataset_type(_value, is_lazy):
            self.__output = _value
            if isinstance(_value, Dataset):
                self.progress = 1.0
        elif isinstance(_value, tuple):
            self.__output = Dataset.from_dict(
                {k: v for k, v in zip(self.output_names, _value)}
            )
            self.progress = 1.0
        elif isinstance(_value, list):
            self.__output = Dataset.from_dict({self.output_names[0]: _value})
            self.progress = 1.0
        elif len(self.output_names) == 1:
            self.__output = Dataset.from_dict({self.output_names[0]: [_value]})
            self.progress = 1.0
        else:
            raise AttributeError(f"Invalid output type: {type(_value)}")
