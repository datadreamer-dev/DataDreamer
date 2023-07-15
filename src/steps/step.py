import warnings
from collections.abc import Generator, Iterable, Iterator, Mapping, Sized
from functools import partial
from typing import Any, Callable, Type, TypeAlias, TypeGuard

from pyarrow.lib import ArrowInvalid, ArrowTypeError

from datasets import Dataset, IterableDataset, iterable_dataset
from datasets.features.features import Features
from datasets.iterable_dataset import _apply_feature_types_on_example

from ..datasets.utils import dataset_zip, get_column_names, iterable_dataset_zip
from ..errors import StepOutputError, StepOutputTypeError
from ..pickling import unpickle as _unpickle
from ..pickling import unpickle_transform
from ..pickling.pickle import _INTERNAL_PICKLE_KEY, _pickle

_CATCH_TYPE_ERRORS_KEY = "__DataDreamer__catch_type_error__"


def _is_iterable(v: Any) -> bool:
    return isinstance(v, Iterable) and not isinstance(v, (str, bytes, Mapping))


def _is_list_or_tuple_type(v) -> TypeGuard[list | tuple]:
    return isinstance(v, (list, tuple))


def _is_dataset_type(v, is_lazy) -> TypeGuard[Dataset | IterableDataset]:
    if not is_lazy and isinstance(v, IterableDataset):
        raise StepOutputError(
            "You must use LazyRows() if you want to output an IterableDataset."
        )
    return isinstance(v, Dataset) or isinstance(v, IterableDataset)


def _normalize(v: Any) -> Any:
    if _is_iterable(v) and not isinstance(v, Sized):
        return list(v)
    else:
        return v


def _catch_type_error(
    action: Callable[..., Dataset | IterableDataset], *args, **kwargs
) -> Dataset | IterableDataset:
    try:
        return action(*args, **kwargs)
    except (ArrowInvalid, ArrowTypeError) as e:
        raise StepOutputTypeError(str(e))


def _catch_type_error_apply_feature_types_on_example(example, *args, **kwargs):
    if type(example) is dict and _CATCH_TYPE_ERRORS_KEY in example:
        del example[_CATCH_TYPE_ERRORS_KEY]
        try:
            return _apply_feature_types_on_example(example, *args, **kwargs)
        except (ArrowInvalid, ArrowTypeError, ValueError, TypeError) as e:
            raise StepOutputTypeError(str(e))
    else:
        return _apply_feature_types_on_example(example, *args, **kwargs)


def _iterable_or_generator_func_to_iterator(  # noqa: C901
    v: Iterable[Any] | Callable[[], Generator[Any, None, None]],
    _value_is_batched: bool,
    output_names: tuple[str, ...],
) -> Iterator[Any]:
    iterator: Iterator[Any]
    if callable(v):
        iterator = v()
    else:
        iterator = iter(v)
    if _value_is_batched:

        def _unbatch(iterator: Any) -> Generator[Any, None, None]:
            column_state = False
            for batch in iterator:
                # Unbatch depending on type
                if isinstance(batch, dict):
                    keys = list(batch.keys())
                    value_batch = [batch[k] for k in keys]
                    for values in zip(*value_batch):
                        yield {k: _normalize(v) for k, v in zip(keys, values)}
                elif isinstance(batch, tuple) and len(output_names) != len(batch):
                    raise StepOutputError(
                        f"Expected {len(output_names)} outputs {output_names}"
                    )
                elif isinstance(batch, tuple):
                    for row in zip(*batch):
                        yield {k: _normalize(v) for k, v in zip(output_names, row)}
                elif (
                    isinstance(batch, list)
                    and len(batch) == len(output_names)
                    and len(batch) > 0
                    and (
                        not _is_list_or_tuple_type(batch[0])
                        or len(batch[0]) != len(output_names)
                        or column_state
                    )
                    and all([_is_iterable(c) for c in batch])
                ):
                    column_state = True
                    for v in zip(*batch):
                        yield v
                else:
                    for v in batch:
                        if isinstance(v, dict) and set(output_names) != set(v.keys()):
                            raise StepOutputError(
                                f"Expected {output_names} as dict keys instead of"
                                f" {tuple(v.keys())}."
                            )
                        yield v

        return partial(_unbatch, iterator)()
    else:
        return iterator


def _untuple(v: Any, output_names: tuple[str, ...]) -> Any:
    if isinstance(v, tuple) and len(v) == 1 and len(v) == len(output_names):
        return v[0]
    else:
        return v


LazyStepOutput: TypeAlias = (
    IterableDataset
    | dict[str, Any]
    | list[Any]
    | Iterator[Any]
    | tuple[Any, ...]
    | Callable[
        [],
        Generator[
            dict[str, Iterable[Any] | list[Any] | tuple[Any, ...]],
            None,
            None,
        ],
    ]
)

LazyBatchStepOutput: TypeAlias = (
    dict[str, Any]
    | list[Any]
    | Iterator[Any]
    | tuple[Any, ...]
    | Callable[
        [],
        Generator[
            dict[str, Iterable[Any] | list[Any] | tuple[Any, ...]],
            None,
            None,
        ],
    ]
)

StepOutput: TypeAlias = (
    None | Dataset | dict[str, Any] | list[Any] | Iterator[Any] | tuple[Any, ...]
)


class LazyRows:
    def __init__(
        self,
        value: LazyStepOutput,
        total_num_rows: None | int = None,
        auto_progress: bool = True,
    ) -> None:
        self.__value: LazyStepOutput = value
        self.total_num_rows: None | int = total_num_rows
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
        total_num_rows: None | int = None,
        auto_progress: bool = True,
    ) -> None:
        self.__value: LazyBatchStepOutput = value
        self.total_num_rows: None | int = total_num_rows
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
        input: None | Dataset | IterableDataset,
        outputs: str | list[str] | tuple[str, ...],
    ):
        self._name: str = name
        self.__progress: None | float = None
        self.input = input
        self.__output: None | Dataset | IterableDataset = None
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
    def output(self) -> Dataset | IterableDataset:
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
        value: StepOutput | LazyRows | LazyRowBatches,
    ):
        # Set progress to 0.0
        self.progress = 0.0

        # Unpack LazyRows and LazyRowsBatches
        _value: StepOutput | LazyStepOutput
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

        # If given None, convert to an empty list
        if _value is None:
            _value = []

        # If given Iterator, convert to a list
        if isinstance(_value, Iterator):
            _value = list(_value)

        # Create a Dataset if given a list or tuple of Datasets
        # or create an IterableDataset if given a list or tuple of IterableDatasets
        if _is_list_or_tuple_type(_value) and len(_value) > 0:
            if all(isinstance(d, Dataset) for d in _value):
                _value = dataset_zip(*_value)
            elif all(_is_dataset_type(d, is_lazy) for d in _value):
                _value = iterable_dataset_zip(*_value)
            elif any(_is_dataset_type(d, True) for d in _value) and len(_value) <= len(
                self.output_names
            ):
                raise StepOutputError(
                    f"Invalid output type: all elements in {_value} must be of type"
                    " Dataset or IterableDataset if one element is."
                )

        # Create a Dataset if given a list or tuple of dicts
        if _is_list_or_tuple_type(_value) and len(_value) > 0:
            if (
                isinstance(_value[0], dict)
                and len(set(_value[0].keys()).intersection(set(self.output_names))) > 0
                and all((isinstance(_v, dict) for _v in _value))
            ):
                for _v in _value:
                    if set(self.output_names) != set(_v.keys()):
                        raise StepOutputError(
                            f"Expected {self.output_names} as dict keys instead of"
                            f" {tuple(_v.keys())}."
                        )
                _value = _catch_type_error(Dataset.from_list, list(_value))

        # Create a Dataset/generator function if given a dict
        if isinstance(_value, dict) and set(self.output_names) == set(_value.keys()):
            if is_lazy and any([callable(v) for v in _value.values()]):
                # One of the values of the dictionary is a generator function,
                # create a generator function of dicts
                def to_dict_generator_wrapper(_value, output_names, _value_is_batched):
                    iters = [
                        _iterable_or_generator_func_to_iterator(
                            _value[k],
                            _value_is_batched,
                            output_names,
                        )
                        for k in output_names
                    ]
                    rows = zip(*iters)
                    for row in rows:
                        yield {k: _normalize(v) for k, v in zip(output_names, row)}

                _value = partial(
                    to_dict_generator_wrapper,
                    _value,
                    self.output_names,
                    _value_is_batched,
                )
                _value_is_batched = False
            elif all([not _is_iterable(v) for v in _value.values()]):
                _value = _catch_type_error(
                    Dataset.from_dict, {k: [_value[k]] for k in self.output_names}
                )
            else:
                _value = _catch_type_error(
                    Dataset.from_dict, {k: _value[k] for k in self.output_names}
                )
        elif isinstance(_value, dict):
            raise StepOutputError(
                f"Expected {self.output_names} as dict keys instead of {tuple(_value.keys())}."
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
            and [_is_iterable(v) for v in _value]
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
            raise StepOutputError(
                f"Expected {len(self.output_names)} outputs {self.output_names}."
            )

        # If given a tuple with the wrong number of elements
        if isinstance(_value, tuple) and len(self.output_names) != len(_value):
            raise StepOutputError(
                f"Expected {len(self.output_names)} outputs {self.output_names}."
            )

        # Create a generator function if given a tuple with a generator function
        if isinstance(_value, tuple) and is_lazy and any([callable(v) for v in _value]):

            def to_dict_generator_wrapper(_value, output_names, _value_is_batched):
                iters = [
                    _iterable_or_generator_func_to_iterator(
                        v, _value_is_batched, output_names
                    )
                    for v in _value
                ]
                rows = zip(*iters)
                for row in rows:
                    yield {k: _normalize(v) for k, v in zip(output_names, row)}

            _value = partial(
                to_dict_generator_wrapper, _value, self.output_names, _value_is_batched
            )
            _value_is_batched = False

        # If given a Dataset with the wrong number of
        if _is_dataset_type(_value, is_lazy) and set(get_column_names(_value)) != set(
            self.output_names
        ):
            raise StepOutputError(
                f"Expected {len(self.output_names)} columns {self.output_names}"
                f" instead of {tuple(get_column_names(_value))}."
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
                    _iterable_or_generator_func_to_iterator(
                        _value, _value_is_batched, self.output_names
                    )
                )
                if _is_iterable(first_row) and not isinstance(first_row, Sized):
                    first_row = list(first_row)
                if isinstance(first_row, dict) and set(self.output_names) != set(
                    first_row.keys()
                ):
                    raise StepOutputError(
                        f"Expected {self.output_names} dict keys from generator"
                        f" function instead of {tuple(first_row.keys())}."
                    )
                elif _is_list_or_tuple_type(first_row):
                    if len(self.output_names) > 1 and len(first_row) == len(
                        self.output_names
                    ):

                        def to_dict_generator_wrapper(
                            _value, output_names, _value_is_batched
                        ):
                            for row in _iterable_or_generator_func_to_iterator(
                                _value, _value_is_batched, output_names
                            ):
                                yield {
                                    k: _normalize(v) for k, v in zip(output_names, row)
                                }

                    elif len(self.output_names) == 1:

                        def to_dict_generator_wrapper(
                            _value, output_names, _value_is_batched
                        ):
                            for v in _iterable_or_generator_func_to_iterator(
                                _value, _value_is_batched, output_names
                            ):
                                yield {
                                    output_names[0]: _normalize(
                                        _untuple(v, output_names)
                                    )
                                }

                    else:
                        raise StepOutputError(
                            f"Expected {len(self.output_names)} outputs"
                            f" {self.output_names} from generator function."
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

            # If so, convert the generator to an IterableDataset) but first,
            # wrap the generator so that we can set progress = 1.0 when complete
            def generator_wrapper(
                _value,
                total_num_rows,
                output_names,
                _value_is_batched,
                not_preview,
            ):
                column_types: dict[str, Type] = {}
                for i, row in enumerate(
                    _iterable_or_generator_func_to_iterator(
                        _value, _value_is_batched, output_names
                    )
                ):
                    # Update and check types
                    for k, v in row.items():
                        prev_type = column_types.get(k, None)
                        new_type = type(v)
                        if new_type is not None:
                            if prev_type is None:
                                column_types[k] = new_type
                            elif new_type != prev_type:
                                raise StepOutputTypeError(
                                    f"Expected {prev_type} got {new_type}"
                                )

                    # Update progress
                    if total_num_rows is not None and not_preview:
                        self.progress = (i + 1) / total_num_rows

                    # Yield a row
                    if not_preview and type(row) is dict:
                        row[_CATCH_TYPE_ERRORS_KEY] = True
                    yield row

                # Update progress
                if not_preview:
                    self.progress = 1.0

            _value_preview = partial(
                generator_wrapper,
                _value,
                total_num_rows,
                self.output_names,
                _value_is_batched,
                False,
            )
            _value = partial(
                generator_wrapper,
                _value,
                total_num_rows,
                self.output_names,
                _value_is_batched,
                True,
            )
            _value_is_batched = False
            try:
                features = _catch_type_error(
                    Dataset.from_list, [next(_value_preview())]
                ).info.features
            except StopIteration:
                features = Features([(n, None) for n in self.output_names])
            _value = _catch_type_error(
                IterableDataset.from_generator, _value, features=features
            )
            iterable_dataset._apply_feature_types_on_example = (
                _catch_type_error_apply_feature_types_on_example
            )

        if _is_dataset_type(_value, is_lazy):
            if _value.info and _value.info.features:
                features = _value.info.features
            else:
                features = Features()
            if self.__pickled:
                _value.set_transform(partial(unpickle_transform, features=features))
            self.__output = _value
            if isinstance(_value, Dataset):
                self.progress = 1.0
        elif isinstance(_value, tuple):
            if all([not _is_iterable(v) for v in _value]):
                self.__output = _catch_type_error(
                    Dataset.from_dict,
                    {k: [_normalize(v)] for k, v in zip(self.output_names, _value)},
                )
            else:
                self.__output = _catch_type_error(
                    Dataset.from_dict,
                    {k: _normalize(v) for k, v in zip(self.output_names, _value)},
                )
            self.progress = 1.0
        elif isinstance(_value, list):
            self.__output = _catch_type_error(
                Dataset.from_dict,
                {
                    self.output_names[0]: [
                        _normalize(_untuple(v, self.output_names)) for v in _value
                    ]
                },
            )
            self.progress = 1.0
        elif len(self.output_names) == 1:
            self.__output = _catch_type_error(
                Dataset.from_dict, {self.output_names[0]: [_normalize(_value)]}
            )
            self.progress = 1.0
        else:
            raise StepOutputError(f"Invalid output type: {type(_value)}.")
