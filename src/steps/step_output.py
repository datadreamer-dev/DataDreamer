import os
import warnings
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Type, TypeAlias, TypeGuard, cast

import dill
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    iterable_dataset,
)
from datasets.builder import DatasetGenerationError
from datasets.features.features import Features, Value
from datasets.iterable_dataset import (
    _apply_feature_types_on_batch,
    _apply_feature_types_on_example,
)
from pyarrow.lib import ArrowInvalid, ArrowTypeError

from .. import DataDreamer
from ..datasets import (
    OutputDataset,
    OutputDatasetColumn,
    OutputIterableDataset,
    OutputIterableDatasetColumn,
)
from ..datasets.utils import (
    dataset_zip,
    drop_unsupported_features,
    get_column_names,
    iterable_dataset_zip,
)
from ..errors import StepOutputError, StepOutputTypeError
from ..utils.background_utils import get_generator_in_background
from ..utils.fingerprint_utils import _DatasetGeneratorPickleHack
from .step_operations import _INTERNAL_STEP_OPERATION_KEY

if TYPE_CHECKING:  # pragma: no cover
    from ..steps import Step

_CATCH_TYPE_ERRORS_KEY = "__DataDreamer__catch_type_error__"


def _is_lazy_type(v: Any) -> bool:
    return callable(v) or isinstance(v, OutputIterableDatasetColumn)


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
    if _is_iterable(v) and not isinstance(v, Sequence):
        return list(v)
    else:
        return v


def _catch_type_error(
    action: Callable[..., Dataset | IterableDataset], *args, **kwargs
) -> Dataset | IterableDataset:
    try:
        return action(*args, **kwargs)
    except (ArrowInvalid, ArrowTypeError) as e:
        raise StepOutputTypeError(str(e)) from None


def _catch_type_error_apply_feature_types_on_example(example, *args, **kwargs):
    if isinstance(example, dict) and _CATCH_TYPE_ERRORS_KEY in example:
        del example[_CATCH_TYPE_ERRORS_KEY]
        try:
            return _apply_feature_types_on_example(example, *args, **kwargs)
        except (ArrowInvalid, ArrowTypeError, ValueError, TypeError) as e:
            raise StepOutputTypeError(str(e)) from None
    else:
        return _apply_feature_types_on_example(example, *args, **kwargs)


def _catch_type_error_apply_feature_types_on_batch(
    batch, *args, **kwargs
):  # pragma: no cover
    if isinstance(batch, dict) and _CATCH_TYPE_ERRORS_KEY in batch:
        del batch[_CATCH_TYPE_ERRORS_KEY]
        try:
            return _apply_feature_types_on_batch(batch, *args, **kwargs)
        except (ArrowInvalid, ArrowTypeError, ValueError, TypeError) as e:
            raise StepOutputTypeError(str(e)) from None
    else:
        return _apply_feature_types_on_batch(batch, *args, **kwargs)


def _monkey_patch_iterable_dataset_apply_feature_types():
    iterable_dataset._apply_feature_types_on_example = (
        _catch_type_error_apply_feature_types_on_example
    )
    iterable_dataset._apply_feature_types_on_batch = (
        _catch_type_error_apply_feature_types_on_batch
    )
    if DataDreamer.initialized():
        DataDreamer.ctx._monkey_patched_iterable_dataset_apply_feature_types = True


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


LazyStepOutputType: TypeAlias = (
    OutputIterableDatasetColumn
    | OutputIterableDataset
    | IterableDataset
    | dict[str, Any]
    | list[Any]
    | Iterator[Any]
    | tuple[Any, ...]
    | Callable[
        [],
        Generator[dict[str, Iterable[Any] | list[Any] | tuple[Any, ...]], None, None],
    ]
)

LazyBatchStepOutputType: TypeAlias = (
    dict[str, Any]
    | list[Any]
    | Iterator[Any]
    | tuple[Any, ...]
    | Callable[
        [],
        Generator[dict[str, Iterable[Any] | list[Any] | tuple[Any, ...]], None, None],
    ]
)

StepOutputType: TypeAlias = (
    None
    | OutputDatasetColumn
    | OutputDataset
    | Dataset
    | dict[str, Any]
    | list[Any]
    | Iterator[Any]
    | tuple[Any, ...]
)


class LazyRows:
    """A wrapper for lazy rows output from a step. See
    :doc:`create your own steps
    <pages/advanced_usage/creating_a_new_datadreamer_.../step>` for more details.

    Args:
        value (lazy output): The lazy rows output.
        total_num_rows: The total number of rows being processed (helps with
            displaying progress).
        auto_progress: Whether to automatically update the progress % for this step.
        writer_batch_size: The batch size to use if saving to disk.
    """

    def __init__(
        self,
        value: LazyStepOutputType,
        total_num_rows: None | int = None,
        auto_progress: bool = True,
        save: bool = False,
        save_writer_batch_size: None | int = None,
    ) -> None:
        self.__value: LazyStepOutputType = value
        if total_num_rows is not None:
            self.total_num_rows: None | int = max(0, total_num_rows)
        else:
            if isinstance(self.__value, OutputIterableDataset):
                self.total_num_rows = self.__value.num_rows
            else:
                self.total_num_rows = None
        if self.total_num_rows is None and auto_progress:
            warnings.warn(
                "You did not specify `total_num_rows`, so we cannot"
                " automatically update the progress % for this step. Either"
                " specify LazyRows(..., total_num_rows=#) or, to disable"
                " this warning, specify LazyRows(.., auto_progress = False)",
                stacklevel=2,
            )
        self.save = save
        self.save_writer_batch_size = save_writer_batch_size

    @property
    def value(self) -> LazyStepOutputType:
        return self.__value


class LazyRowBatches:
    """
    A wrapper for lazy row batches output from a step. See
    :doc:`create your own steps
    <pages/advanced_usage/creating_a_new_datadreamer_.../step>` for more details.

    Args:
        value (lazy output): The lazy row batches output.
        total_num_rows: The total number of rows being processed (helps with
            displaying progress).
        auto_progress: Whether to automatically update the progress % for this step.
        writer_batch_size: The batch size to use if saving to disk.
    """

    def __init__(
        self,
        value: LazyBatchStepOutputType,
        total_num_rows: None | int = None,
        auto_progress: bool = True,
        save: bool = False,
        save_writer_batch_size: None | int = None,
    ) -> None:
        self.__value: LazyBatchStepOutputType = value
        if total_num_rows is not None:
            self.total_num_rows: None | int = max(0, total_num_rows)
        else:
            if isinstance(self.__value, OutputIterableDataset):
                self.total_num_rows = self.__value.num_rows
            else:
                self.total_num_rows = None
        if self.total_num_rows is None and auto_progress:
            warnings.warn(
                "You did not specify `total_num_rows`, so we cannot"
                " automatically update the progress % for this step. Either"
                " specify LazyRowBatches(..., total_num_rows=#) or, to"
                " disable this warning, specify LazyRowBatches(..,"
                " auto_progress = False)",
                stacklevel=2,
            )
        self.save = save
        self.save_writer_batch_size = save_writer_batch_size

    @property
    def value(self) -> LazyBatchStepOutputType:
        return self.__value


def __output_to_dataset(  # noqa: C901
    step: "Step",
    output_names: tuple[str, ...],
    output_name_mapping: dict[str, str],
    set_progress: Callable[[float], None],
    set_progress_rows: Callable[[float], None],
    get_pickled: Callable[[], bool],
    value: StepOutputType | LazyRows | LazyRowBatches,
) -> OutputDataset | tuple[Callable, Features, None | int]:
    # Set progress to 0.0
    set_progress(0.0)

    # Unpack LazyRows and LazyRowsBatches
    _value: StepOutputType | LazyStepOutputType
    is_lazy = False
    total_num_rows = None
    _value_is_batched = False
    should_save = False
    save_writer_batch_size: None | int = None
    if isinstance(value, LazyRows):
        _value = value.value
        is_lazy = True
        total_num_rows = value.total_num_rows
        should_save = value.save
        save_writer_batch_size = value.save_writer_batch_size
    elif isinstance(value, LazyRowBatches):
        _value = value.value
        is_lazy = True
        total_num_rows = value.total_num_rows
        if not _is_dataset_type(_value, is_lazy) and type(_value) not in [
            OutputDataset,
            OutputIterableDataset,
        ]:
            _value_is_batched = True
        should_save = value.save
        save_writer_batch_size = value.save_writer_batch_size
    else:
        _value = value
    del value

    # If given DatasetDict-type, raise an error
    if isinstance(_value, (DatasetDict, IterableDatasetDict)):
        raise StepOutputError("You supplied a DatasetDict, supply a Dataset instead.")

    # If given None, convert to an empty list
    if _value is None:
        _value = []

    # If given Iterator, convert to a list
    if isinstance(_value, Iterator):
        _value = list(_value)

    # If given OutputDatasetColumn, convert to a list
    if isinstance(_value, OutputDatasetColumn):
        _value = list(_value)

    # If given instance of OutputDataset or OutputIterableDataset, unpack them to
    # Dataset or IterableDataset
    if type(_value) is OutputDataset or type(_value) is OutputIterableDataset:
        if _value._pickled:
            step.pickle(True)
        _value = _value.dataset

    # If given a list of OutputDatasets or OutputIterableDatasets, unpack them to
    # Datasets or IterableDatasets
    if _is_list_or_tuple_type(_value) and len(_value) > 0:
        if all(
            (
                type(d) in [OutputDataset, OutputIterableDataset]
                or _is_dataset_type(d, is_lazy)
            )
            for d in _value
        ) or (
            len(_value) <= len(output_names)
            and any(
                (
                    type(d) in [OutputDataset, OutputIterableDataset]
                    or _is_dataset_type(d, is_lazy)
                )
                for d in _value
            )
        ):
            new_value: list[Any] = []
            for d in _value:
                if type(d) in [OutputDataset, OutputIterableDataset]:
                    if d._pickled:
                        step.pickle(True)
                    new_value.append(d.dataset)
                else:
                    new_value.append(d)
            _value = new_value

    # If given a Dataset or IterableDataset, make a copy and reset the format
    if _is_dataset_type(_value, is_lazy):
        _value = deepcopy(_value)
        drop_unsupported_features(_value)

    # Create a Dataset if given a list or tuple of Datasets
    # or create an IterableDataset if given a list or tuple of IterableDatasets
    if _is_list_or_tuple_type(_value) and len(_value) > 0:
        if all(isinstance(d, Dataset) for d in _value):
            _value = dataset_zip(*_value)
        elif all(_is_dataset_type(d, is_lazy) for d in _value):
            _value = iterable_dataset_zip(*_value)
        elif len(_value) <= len(output_names) and any(
            _is_dataset_type(d, True) for d in _value
        ):
            raise StepOutputError(
                f"Invalid output type: all elements in {_value} must be of type"
                " Dataset or IterableDataset if one element is."
            )

    # Create a Dataset if given a list or tuple of dicts
    if _is_list_or_tuple_type(_value) and len(_value) > 0:
        if (
            isinstance(_value[0], dict)
            and len(set(_value[0].keys()).intersection(set(output_names))) > 0
            and all((isinstance(_v, dict) for _v in _value))
        ):
            for _v in _value:
                if set(output_names) != set(_v.keys()):
                    raise StepOutputError(
                        f"Expected {output_names} as dict keys instead of"
                        f" {tuple(_v.keys())}."
                    )
            _value = _catch_type_error(Dataset.from_list, list(_value))

    # Create a Dataset/generator function if given a dict
    if isinstance(_value, dict) and set(output_names) == set(_value.keys()):
        if is_lazy and any([_is_lazy_type(v) for v in _value.values()]):
            # One of the values of the dictionary is a generator function,
            # create a generator function of dicts
            def to_dict_generator_wrapper(_value, output_names, _value_is_batched):
                iters = [
                    _iterable_or_generator_func_to_iterator(
                        _value[k], _value_is_batched, output_names
                    )
                    for k in output_names
                ]
                rows = zip(*iters)
                for row in rows:
                    yield {k: _normalize(v) for k, v in zip(output_names, row)}

            _value = partial(
                to_dict_generator_wrapper, _value, output_names, _value_is_batched
            )
            _value_is_batched = False
        elif all([not _is_iterable(v) for v in _value.values()]):
            _value = _catch_type_error(
                Dataset.from_dict, {k: [_normalize(_value[k])] for k in output_names}
            )
        else:
            _value = _catch_type_error(
                Dataset.from_dict, {k: _normalize(_value[k]) for k in output_names}
            )
    elif isinstance(_value, dict):
        raise StepOutputError(
            f"Expected {output_names} as dict keys instead of {tuple(_value.keys())}."
        )

    # If given a single list when more than one output force it into a tuple
    if (
        isinstance(_value, list)
        and len(output_names) > 1
        and len(_value) > 0
        and _is_list_or_tuple_type(_value[0])
        and len(_value[0]) == len(output_names)
    ):
        _value = tuple(zip(*_value))
    elif (
        isinstance(_value, list)
        and len(output_names) > 1
        and len(_value) == len(output_names)
        and [_is_iterable(v) for v in _value]
    ):
        _value = tuple(_value)
    elif (
        isinstance(_value, list)
        and len(output_names) == 1
        and len(_value) == len(output_names)
        and isinstance(_value[0], OutputDatasetColumn)
    ):
        _value = tuple(_value)
    elif (
        isinstance(_value, list)
        and len(output_names) == 1
        and len(_value) == len(output_names)
        and _is_lazy_type(_value[0])
    ):
        _value = tuple(_value)

    # If given a single list
    if isinstance(_value, list) and len(output_names) > 1:
        raise StepOutputError(f"Expected {len(output_names)} outputs {output_names}.")

    # If given a tuple with the wrong number of elements
    if isinstance(_value, tuple) and len(output_names) != len(_value):
        raise StepOutputError(f"Expected {len(output_names)} outputs {output_names}.")

    # Create a generator function if given a tuple with a generator function
    if (
        isinstance(_value, tuple)
        and is_lazy
        and any([_is_lazy_type(v) for v in _value])
    ):

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
            to_dict_generator_wrapper, _value, output_names, _value_is_batched
        )
        _value_is_batched = False

    # If given a Dataset with the wrong number of columns
    if (
        _is_dataset_type(_value, is_lazy)
        and set(get_column_names(_value)) != set(output_names)
        and not hasattr(step.__class__, _INTERNAL_STEP_OPERATION_KEY)
    ):
        raise StepOutputError(
            f"Expected {len(output_names)} columns {output_names}"
            f" instead of {tuple(get_column_names(_value))}."
        )

    # If IterableDataset convert to a generator function
    if is_lazy and isinstance(_value, IterableDataset):

        def to_dict_generator_wrapper(_value, output_names, _value_is_batched):
            return iter(_value)

        _value = partial(
            to_dict_generator_wrapper, _value, output_names, _value_is_batched
        )

    # Create an IterableDataset if given a generator function of dicts
    features = Features([(n, Value("null")) for n in output_names])
    if is_lazy and _is_lazy_type(_value):
        # Make sure the generator returns a dict and the keys are correct
        try:
            first_row = next(
                _iterable_or_generator_func_to_iterator(
                    _value, _value_is_batched, output_names
                )
            )
            if _is_iterable(first_row) and not isinstance(first_row, Sequence):
                first_row = list(first_row)
            if (
                isinstance(first_row, dict)
                and set(output_names) != set(first_row.keys())
                and not hasattr(step.__class__, _INTERNAL_STEP_OPERATION_KEY)
            ):
                raise StepOutputError(
                    f"Expected {output_names} dict keys from generator"
                    f" function instead of {tuple(first_row.keys())}."
                )
            elif _is_list_or_tuple_type(first_row):
                if len(output_names) > 1 and len(first_row) == len(output_names):

                    def to_dict_generator_wrapper(
                        _value, output_names, _value_is_batched
                    ):
                        for row in _iterable_or_generator_func_to_iterator(
                            _value, _value_is_batched, output_names
                        ):
                            yield {k: _normalize(v) for k, v in zip(output_names, row)}

                elif len(output_names) == 1:

                    def to_dict_generator_wrapper(
                        _value, output_names, _value_is_batched
                    ):
                        for v in _iterable_or_generator_func_to_iterator(
                            _value, _value_is_batched, output_names
                        ):
                            yield {
                                output_names[0]: _normalize(_untuple(v, output_names))
                            }

                else:
                    raise StepOutputError(
                        f"Expected {len(output_names)} outputs"
                        f" {output_names} from generator function."
                    )

                _value = partial(
                    to_dict_generator_wrapper, _value, output_names, _value_is_batched
                )
                _value_is_batched = False
            elif not isinstance(first_row, dict) and len(output_names) == 1:

                def to_dict_generator_wrapper(_value, output_names, _value_is_batched):
                    for v in _iterable_or_generator_func_to_iterator(
                        _value, _value_is_batched, output_names
                    ):
                        yield {output_names[0]: _normalize(v)}

                _value = partial(
                    to_dict_generator_wrapper, _value, output_names, _value_is_batched
                )
                _value_is_batched = False
        except StopIteration:
            pass

        # If so, convert the generator to an IterableDataset, but first,
        # wrap the generator so that we can set progress = 1.0 when complete
        def generator_wrapper(
            name, _value, total_num_rows, output_names, _value_is_batched, not_preview
        ):
            column_types: dict[str, Type] = {}
            i = None
            for i, row in enumerate(
                _iterable_or_generator_func_to_iterator(
                    _value, _value_is_batched, output_names
                )
            ):
                # Update and check types
                for k, v in row.items():
                    prev_type = column_types.get(k, None)
                    new_type = type(v)
                    if not isinstance(v, type(None)):
                        if prev_type is None:
                            column_types[k] = new_type
                        elif new_type != prev_type:
                            raise StepOutputTypeError(
                                f"Expected {prev_type} got {new_type}"
                            )

                # Update progress
                if total_num_rows is not None and not_preview:
                    set_progress((i + 1) / total_num_rows)
                elif total_num_rows is None and not_preview:
                    set_progress_rows(i + 1)

                # Yield a row
                if not_preview and isinstance(row, dict):
                    row[_CATCH_TYPE_ERRORS_KEY] = True
                yield row

            # Update progress
            if not_preview:
                if i is not None:
                    set_progress_rows(i + 1)
                set_progress(1.0)

        _value_preview = partial(
            generator_wrapper,
            step.name,
            _value,
            total_num_rows,
            output_names,
            _value_is_batched,
            False,
        )
        _value = partial(
            generator_wrapper,
            step.name,
            _value,
            total_num_rows,
            output_names,
            _value_is_batched,
            True,
        )
        _value_is_batched = False
        try:
            features = _catch_type_error(
                Dataset.from_list, [next(_value_preview())]
            ).info.features
        except StopIteration:
            pass

    # Note: This type can be misleading, but __output will only ever be a Dataset
    # or a Callable (a generator function). We never actually return an
    # IterableDataset here, it is only annotated that way for mypy.
    #
    # If __output is a Callable (a generator function) it becomes an IterableDataset
    # in _output_to_dataset() after we return it from this function.
    __output: Dataset | IterableDataset | Callable

    if _is_dataset_type(_value, is_lazy) or (is_lazy and callable(_value)):
        __output = _value
        if isinstance(_value, Dataset):
            set_progress(1.0)
    elif isinstance(_value, tuple):
        if all([not _is_iterable(v) for v in _value]):
            __output = _catch_type_error(
                Dataset.from_dict,
                {k: [_normalize(v)] for k, v in zip(output_names, _value)},
            )
        else:
            __output = _catch_type_error(
                Dataset.from_dict,
                {k: _normalize(v) for k, v in zip(output_names, _value)},
            )
        set_progress(1.0)
    elif isinstance(_value, list):
        __output = _catch_type_error(
            Dataset.from_dict,
            {output_names[0]: [_normalize(_untuple(v, output_names)) for v in _value]},
        )
        set_progress(1.0)
    elif len(output_names) == 1:
        __output = _catch_type_error(
            Dataset.from_dict, {output_names[0]: [_normalize(_value)]}
        )
        set_progress(1.0)
    else:
        if callable(_value) or isinstance(
            _value,
            (OutputIterableDatasetColumn, OutputIterableDataset, IterableDataset),
        ):
            raise StepOutputError(
                f"Invalid output type: {type(_value)}."
                " You may want to use LazyRows() or LazyRowBatches()."
            )
        else:
            raise StepOutputError(f"Invalid output type: {type(_value)}.")

    # If the user requested to save, then convert the generator to a Dataset
    if should_save and callable(__output) and step._output_folder_path:
        _value = _catch_type_error(
            IterableDataset.from_generator,
            _DatasetGeneratorPickleHack(_value),
            features=features,
        )
        _monkey_patch_iterable_dataset_apply_feature_types()

        def save_generator(_value):
            yield from iter(_value)

        cache_path = os.path.join(step._output_folder_path, ".datadreamer_save_cache")
        try:
            __output = _catch_type_error(
                Dataset.from_generator,
                partial(save_generator, _value),
                features=features,
                cache_dir=cache_path,
                writer_batch_size=save_writer_batch_size,
                num_proc=step.save_num_proc,
            )
        except DatasetGenerationError as e:
            raise e.__cause__ from None

    # Return
    if callable(__output):
        return __output, features, total_num_rows
    else:
        # Rename columns and create OutputDataset
        rename_mapping = {k: v for k, v in output_name_mapping.items() if k != v}
        __output = __output.rename_columns(rename_mapping)
        assert isinstance(__output, Dataset)
        return OutputDataset(step=step, dataset=__output, pickled=get_pickled())


def _output_to_dataset(  # noqa: C901
    pipe: Any,
    step: "Step",
    output_names: tuple[str, ...],
    output_name_mapping: dict[str, str],
    set_progress: Callable[[float], None],
    set_progress_rows: Callable[[float], None],
    get_pickled: Callable[[], bool],
    value: StepOutputType | LazyRows | LazyRowBatches,
    save_output_to_disk: Callable[[OutputDataset], None],
) -> OutputDataset | OutputIterableDataset:
    # Check whether we are running in the background or not
    if pipe:  # pragma: no cover
        # We are running in the background
        output = __output_to_dataset(
            step=step,
            output_names=output_names,
            output_name_mapping=output_name_mapping,
            set_progress=set_progress,
            set_progress_rows=set_progress_rows,
            get_pickled=get_pickled,
            value=cast(Callable, value)(),  # Here: value = the run() function itself
        )
        if isinstance(output, tuple):
            _value, features, total_num_rows = output

            # Wrapping the generator like this make it pickle-able to be returned
            # from the process, this way, every time the generator is called from the
            # parent process, it spawns a child process that runs the generation.
            _value = partial(get_generator_in_background, _value)

            # Convert the background-process generator to IterableDataset
            __output = _catch_type_error(
                IterableDataset.from_generator,
                _DatasetGeneratorPickleHack(_value),
                features=features,
            )
            _monkey_patch_iterable_dataset_apply_feature_types()

            # Rename columns and create OutputIterableDataset
            rename_mapping = {k: v for k, v in output_name_mapping.items() if k != v}
            __output = __output.rename_columns(rename_mapping)
            assert isinstance(__output, IterableDataset)
            iterable_return_val = OutputIterableDataset(
                step=step,
                dataset=__output,
                pickled=get_pickled(),
                total_num_rows=total_num_rows,
            )

            # This sets a dummy output on the child process, but that doesn't really do
            # anything. The only purpose for this is because we are logging progress
            # from the child process, the logger needs to know if the output is an
            # OutputIterableDataset.
            def dummy_empty_generator():
                return iter(())

            step._set_output(
                LazyRows(
                    dummy_empty_generator,
                    total_num_rows=total_num_rows,
                    auto_progress=False if total_num_rows is None else True,
                )
            )

            # Send the data card (sometimes this gets updated in .run())
            # + send the pickled version of the OutputIterableDataset to the parent process
            pipe.put(dill.dumps((step._data_card, iterable_return_val)))

            return iterable_return_val  # Meaningless, see: pipe.put()
        else:
            # Saving the data to disk makes the OutputDataset pickle-able to be returned
            # from the child process. By saving to disk, we pickle an OutputDataset that
            # holds a memory-mapped Dataset, which is very memory-cheap to pickle and
            # send back to the parent process.
            if step._output_folder_path:
                save_output_to_disk(output)
            try:
                return_val = output

                # Send the data card (sometimes this gets updated in .run())
                # + send the pickled version of the OutputDataset to the parent process
                pipe.put(dill.dumps((step._data_card, return_val)))

                return return_val  # Meaningless, see: pipe.put()
            except dill.PicklingError as e:
                raise StepOutputTypeError(str(e)) from None
    else:
        # We are not running in the background
        output = __output_to_dataset(
            step=step,
            output_names=output_names,
            output_name_mapping=output_name_mapping,
            set_progress=set_progress,
            set_progress_rows=set_progress_rows,
            get_pickled=get_pickled,
            value=value,  # Here: value = the result of the run() function
        )
        if isinstance(output, tuple):
            _value, features, total_num_rows = output

            # Convert generator to IterableDataset
            __output = _catch_type_error(
                IterableDataset.from_generator,
                _DatasetGeneratorPickleHack(_value),
                features=features,
            )
            _monkey_patch_iterable_dataset_apply_feature_types()

            # Rename columns and create OutputIterableDataset
            rename_mapping = {k: v for k, v in output_name_mapping.items() if k != v}
            __output = __output.rename_columns(rename_mapping)
            assert isinstance(__output, IterableDataset)
            return OutputIterableDataset(
                step=step,
                dataset=__output,
                pickled=get_pickled(),
                total_num_rows=total_num_rows,
            )
        else:
            return output


__all__ = ["LazyRowBatches", "LazyRows", "StepOutputType", "_output_to_dataset"]
