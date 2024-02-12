import os
from collections import OrderedDict
from collections.abc import Iterable
from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

from datasets import Dataset, DatasetDict, IterableDataset, concatenate_datasets
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from datasets.fingerprint import Hasher
from pyarrow.lib import ArrowInvalid, ArrowTypeError

from .. import DataDreamer
from ..datasets import OutputDataset, OutputIterableDataset
from ..errors import StepOutputTypeError
from ..pickling import unpickle_transform
from ..utils.arg_utils import Default, default_to
from ..utils.fingerprint_utils import stable_fingerprint
from .step_background import wait

if TYPE_CHECKING:  # pragma: no cover
    from .step import Step

_INTERNAL_STEP_OPERATION_KEY = "__DataDreamer__step_operation__"
_INTERNAL_STEP_OPERATION_NO_SAVE_KEY = "__DataDreamer__step_operation_no_save__"


##################################
# Helpers step operations
##################################


def _iterable_dataset_to_dataset(  # noqa: C901
    self,
    step: "Step",
    iterable_dataset: IterableDataset,
    writer_batch_size: None | int,
    save_num_proc: None | int | Default,
) -> Dataset:
    def dataset_generator(iterable_dataset):
        i = None
        for i, row in enumerate(iterable_dataset):
            if self and step.output.num_rows is not None:
                self.progress = (i + 1) / step.output.num_rows
            elif self:
                self._set_progress_rows(i + 1)
            yield row
        if self:
            if i is not None:
                self._set_progress_rows(i + 1)
            self.progress = 1.0

    step_to_use: Step = self if self is not None else step

    step_to_use.logger.debug(
        f"Iterating through all of the lazy results of '{step.name}'."
    )

    try:
        if step_to_use._output_folder_path:
            cache_path = os.path.join(
                step_to_use._output_folder_path, ".datadreamer_save_cache"
            )
            for features in [None, step.output._features]:
                try:
                    dataset = Dataset.from_generator(
                        partial(dataset_generator, iterable_dataset),
                        features=features,
                        cache_dir=cache_path,
                        writer_batch_size=writer_batch_size,
                        num_proc=default_to(save_num_proc, None),
                    )
                    break
                except ValueError as e:
                    if "corresponds to no data" in str(e):
                        dataset = Dataset.from_dict(
                            {col_name: [] for col_name in step.output._features},
                            features=step.output._features,
                        )  # type: ignore[call-arg]
                        break
                except DatasetGenerationError as e:  # pragma: no cover
                    if features is None and isinstance(
                        e.__cause__, SchemaInferenceError
                    ):
                        continue
                    else:
                        raise e
        else:
            dataset = Dataset.from_list(list(dataset_generator(iterable_dataset)))
    except DatasetGenerationError as e:
        raise e.__cause__ from None
    if self:
        self._pickled = step.output._pickled
    return dataset


def _step_to_dataset_dict(
    step: "Step",
    train_size: None | float | int = None,
    validation_size: None | float | int = None,
    test_size: None | float | int = None,
    stratify_by_column: None | str = None,
    writer_batch_size: None | int = 1000,
    save_num_proc: None | int | Default = None,
    save_num_shards: None | int | Default = None,
) -> tuple[OutputDataset, DatasetDict]:
    # Wait for a lazy step to complete
    wait(step)
    output: OutputDataset
    dataset: Dataset
    if isinstance(step.output, OutputIterableDataset):
        dataset = _iterable_dataset_to_dataset(
            None,
            step,
            iterable_dataset=step.output.dataset,
            writer_batch_size=writer_batch_size,
            save_num_proc=save_num_proc,
        )
        output = OutputDataset(step=step, dataset=dataset, pickled=step._pickled)
    else:
        output = step.output
        dataset = output.dataset
    splits: OrderedDict[str, Any] = OrderedDict(
        [
            (
                split_name,
                Decimal(str(proportion))
                if isinstance(proportion, float)
                else proportion,
            )
            for split_name, proportion in [
                ("train", train_size),
                ("validation", validation_size),
                ("test", test_size),
            ]
            if proportion is not None and proportion != 0
        ]
    )
    if len(splits) == 0:
        splits = OrderedDict({"train": Decimal("1.0")})
    split_names = list(splits.keys())
    proportions = list(splits.values())
    total = sum(proportions)
    if total != Decimal("1.0") and total != len(dataset):
        raise ValueError(
            "If not None, train_size, validation_size, test_size must sum up to 1.0 or"
            f" the number of rows ({len(dataset)}) in the dataset. Instead, got a total"
            f" of: {total}."
        )

    # Split the dataset
    total_proportion = Decimal("1.0")
    remainder_dataset = dataset
    while len(proportions) > 1:
        split_name = split_names.pop()
        proportion = proportions.pop()
        new_dataset_dict = remainder_dataset.train_test_split(
            test_size=float(proportion / total_proportion)
            if total == Decimal("1.0")
            else int(proportion),
            shuffle=False,
            stratify_by_column=stratify_by_column,
            writer_batch_size=writer_batch_size,
        )
        splits[split_name] = new_dataset_dict["test"]
        remainder_dataset = new_dataset_dict["train"]
        if total == Decimal("1.0"):
            total_proportion = total_proportion - proportion

    # Finally, assign the remainder of the splits
    split_name = split_names.pop()
    splits[split_name] = remainder_dataset

    return output, DatasetDict(splits)


def _user_transform(
    self,
    step: "Step",
    function: Callable,
    with_indices: bool,
    batched: bool,
    x: dict,
    idx,
    *args,
    **kwargs,
):
    finished_idx = min(idx) if isinstance(idx, Iterable) else idx
    if step.output.num_rows is not None:
        self.progress = (finished_idx) / step.output.num_rows
    else:
        self._set_progress_rows(finished_idx)
    if step.output._pickled or step.output._pickled_inferred:
        x = unpickle_transform(x, features=step.output._features, batched=batched)
    if with_indices:
        return function(x, idx, *args, **kwargs)
    else:
        return function(x, *args, **kwargs)


##################################
# Constructors for step operations
##################################


def __create_step_operation_step(  # noqa: C901
    step: "Step",
    name: None | str,
    op_cls: type["Step"],
    op_name: str,
    run: Callable,
    no_save: bool = False,
    setup: None | Callable = None,
    **kwargs,
) -> "Step":
    from .step import LazyRows

    kwargs["progress_interval"] = default_to(
        kwargs.get("progress_interval", None), step.progress_interval
    )
    kwargs["save_num_proc"] = default_to(
        kwargs.get("save_num_proc", None), step.save_num_proc
    )
    kwargs["save_num_shards"] = default_to(
        kwargs.get("save_num_shards", None), step.save_num_shards
    )
    writer_batch_size = kwargs.pop("writer_batch_size", 1000)

    # Hash the fingerprint
    kwargs["args"]["fingerprint"] = Hasher.hash(kwargs["args"]["fingerprint"])

    class _StepOpStep(op_cls):  # type:ignore[valid-type,misc]
        def setup(self):
            self.register_arg("fingerprint")
            if callable(setup):
                setup(self)  # pragma: no cover

        def run(self):
            total_num_rows = step.output.num_rows

            try:
                run_output = run(self)
            except (ArrowInvalid, ArrowTypeError, ValueError, TypeError) as e:
                raise StepOutputTypeError(str(e)) from None

            if no_save and isinstance(run_output, LazyRows):
                return run_output
            elif isinstance(run_output, LazyRows):
                total_num_rows = run_output.total_num_rows
                run_output = run_output.value

            if not no_save and isinstance(run_output, IterableDataset):
                run_output = _iterable_dataset_to_dataset(
                    self=self,
                    step=step,
                    iterable_dataset=run_output,
                    writer_batch_size=writer_batch_size,
                    save_num_proc=kwargs["save_num_proc"],
                )

            if step._pickled or step.output._pickled:
                self.pickle(True)

            if isinstance(run_output, IterableDataset):
                return LazyRows(run_output, total_num_rows=total_num_rows)
            else:
                return run_output

    setattr(_StepOpStep, _INTERNAL_STEP_OPERATION_KEY, True)
    if no_save:
        setattr(_StepOpStep, _INTERNAL_STEP_OPERATION_NO_SAVE_KEY, True)
    _StepOpStep.__name__ = op_cls.__name__
    _StepOpStep.__qualname__ = op_cls.__name__
    _StepOpStep.__module__ = op_cls.__module__
    final_name: str = (
        name
        or DataDreamer._new_step_name(
            step.name.split(" / ")[-1], transform=op_name, record=False
        ).split(" / ")[-1]
    )
    wait(step)

    # Create the op step
    step_op_step = _StepOpStep(
        name=final_name,
        inputs={
            column_name: step.output[column_name]
            for column_name in step.output.column_names
        },
        verbose=step.verbose,
        log_level=step.log_level,
        **kwargs,
    )

    # Make sure features are the same if the user did something like filter() -> 0 rows
    from .step import SelectColumnsStep

    if (
        (step_op_step.output.dataset.info.features is None)
        and step.output.dataset.info.features != {}
        and not isinstance(op_cls, SelectColumnsStep)
    ):
        step_op_step.output.dataset.info.features = step.output.dataset.info.features
    return step_op_step


def __concatenate(*steps: "Step", axis: int, **kwargs):  # noqa: C901
    lazy = kwargs["lazy"]
    del kwargs["lazy"]

    from .step import LazyRows, Step

    if len(steps) == 0:
        raise ValueError(
            f"You must provide at least one step to {kwargs['op_name']}()."
        )
    if not all([isinstance(step, Step) for step in steps]):
        raise TypeError(f"All arguments to {kwargs['op_name']}() must be of type Step.")
    wait(*steps)

    if kwargs["name"] is None:
        step_names = ", ".join([step.name.split(" / ")[-1] for step in steps])
        kwargs["name"] = DataDreamer._new_step_name(
            kwargs["op_name"] + f"({step_names})", record=False
        ).split(" / ")[-1]

    def run(self):
        datasets: list[Dataset | IterableDataset] = []
        if not lazy:
            for step in steps:
                if step._pickled or step.output._pickled:
                    self.pickle(True)
                if isinstance(step.output, OutputDataset):
                    datasets.append(step.output.dataset)
                else:
                    datasets.append(
                        _iterable_dataset_to_dataset(
                            self=self,
                            step=step,
                            iterable_dataset=step.output.dataset,
                            writer_batch_size=kwargs["writer_batch_size"],
                            save_num_proc=kwargs["save_num_proc"],
                        )
                    )
            return concatenate_datasets(datasets, axis=axis)
        else:
            total_num_rows: None | int = 0
            max_total_num_rows: None | int = None
            if all([step.output.num_rows is not None for step in steps]):
                max_total_num_rows = max(
                    [cast(int, step.output.num_rows) for step in steps]
                )
            for step in steps:
                if step._pickled or step.output._pickled:
                    self.pickle(True)
                if isinstance(step.output, OutputDataset):
                    if total_num_rows is not None:
                        total_num_rows += step.output.num_rows
                    datasets.append(step.output.dataset.to_iterable_dataset())
                else:
                    if total_num_rows is not None and step.output.num_rows is not None:
                        total_num_rows += step.output.num_rows
                    elif step.output.num_rows is None:
                        total_num_rows = None
                    datasets.append(step.output.dataset)
            return LazyRows(
                cast(IterableDataset, concatenate_datasets(datasets, axis=axis)),
                total_num_rows=max_total_num_rows if axis == 1 else total_num_rows,
            )

    kwargs["step"] = steps[0]
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [[step.fingerprint for step in steps], axis]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_select_step(indices: Iterable, **kwargs):
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        dataset: Dataset
        if isinstance(step.output, OutputIterableDataset):
            dataset = _iterable_dataset_to_dataset(
                self=self,
                step=step,
                iterable_dataset=step.output.dataset,
                writer_batch_size=kwargs["writer_batch_size"],
                save_num_proc=kwargs["save_num_proc"],
            )
        else:
            dataset = step.output.dataset
        return dataset.select(
            indices=indices, writer_batch_size=kwargs["writer_batch_size"]
        )

    from .step import SelectStep

    kwargs["op_cls"] = SelectStep
    kwargs["op_name"] = "select"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [step.fingerprint, list(indices)]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_select_columns_step(column_names: str | list[str], **kwargs):
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        for n in column_names:
            assert n in step.output.column_names, f"Column '{n}' not found."
        return step.output.dataset.select_columns(column_names=column_names)

    from .step import SelectColumnsStep

    kwargs["op_cls"] = SelectColumnsStep
    kwargs["op_name"] = "select_columns"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [step.fingerprint, column_names]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_take_step(n: int, **kwargs):
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        from .step import LazyRows

        dataset: Dataset | IterableDataset = step.output.dataset

        if isinstance(dataset, Dataset):
            return Dataset.from_dict(dataset[:n])
        elif isinstance(dataset, IterableDataset):
            total_num_rows = None
            if step.output.num_rows is not None:
                total_num_rows = min(n, step.output.num_rows)
            return LazyRows(dataset.take(n), total_num_rows=total_num_rows)

    from .step import TakeStep

    kwargs["op_cls"] = TakeStep
    kwargs["op_name"] = "take"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [step.fingerprint, n]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_skip_step(n: int, **kwargs):
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        dataset: Dataset | IterableDataset = step.output.dataset

        if isinstance(dataset, Dataset):
            return Dataset.from_dict(dataset[n:])
        elif isinstance(dataset, IterableDataset):
            from .step import LazyRows

            total_num_rows = None
            if step.output.num_rows is not None:
                total_num_rows = max(step.output.num_rows - n, 0)
            return LazyRows(dataset.skip(n), total_num_rows=total_num_rows)

    from .step import SkipStep

    kwargs["op_cls"] = SkipStep
    kwargs["op_name"] = "skip"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [step.fingerprint, n]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_shuffle_step(seed: None | int, buffer_size: int, **kwargs) -> "Step":
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        dataset: Dataset | IterableDataset
        if not lazy and isinstance(step.output, OutputIterableDataset):
            dataset = _iterable_dataset_to_dataset(
                self=self,
                step=step,
                iterable_dataset=step.output.dataset,
                writer_batch_size=kwargs["writer_batch_size"],
                save_num_proc=kwargs["save_num_proc"],
            )
        else:
            dataset = step.output.dataset
        if isinstance(dataset, Dataset):
            return dataset.shuffle(
                seed=seed, writer_batch_size=kwargs["writer_batch_size"]
            )
        else:
            return dataset.shuffle(seed=seed, buffer_size=buffer_size)

    from .step import ShuffleStep

    kwargs["op_cls"] = ShuffleStep
    kwargs["op_name"] = "shuffle"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [step.fingerprint, seed, buffer_size]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_sort_step(
    column_names: str | Sequence[str],
    reverse: bool | Sequence[bool],
    null_placement,
    **kwargs,
):
    step = kwargs["step"]

    def run(self):
        dataset: Dataset
        if isinstance(step.output, OutputIterableDataset):
            dataset = _iterable_dataset_to_dataset(
                self=self,
                step=step,
                iterable_dataset=step.output.dataset,
                writer_batch_size=kwargs["writer_batch_size"],
                save_num_proc=kwargs["save_num_proc"],
            )
        else:
            dataset = step.output.dataset
        return dataset.sort(
            column_names=column_names, reverse=reverse, null_placement=null_placement
        )

    from .step import SortStep

    kwargs["op_cls"] = SortStep
    kwargs["op_name"] = "sort"
    kwargs["no_save"] = False
    kwargs["args"] = {
        "fingerprint": [step.fingerprint, column_names, reverse, null_placement]
    }
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_add_item_step(item: dict, **kwargs):
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        from .step import LazyRows

        dataset: Dataset | IterableDataset = step.output.dataset

        if isinstance(dataset, Dataset):
            return dataset.add_item(item)
        elif isinstance(dataset, IterableDataset):

            def add_item_generator(dataset):
                for row in dataset:
                    yield row
                yield item

            total_num_rows = None
            if step.output.num_rows is not None:
                total_num_rows = step.output.num_rows + 1

            return LazyRows(
                IterableDataset.from_generator(
                    partial(add_item_generator, dataset), features=step.output._features
                ),
                total_num_rows=total_num_rows,
            )

    from .step import AddItemStep

    kwargs["op_cls"] = AddItemStep
    kwargs["op_name"] = "add_item"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [step.fingerprint, item]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_map_step(
    function: Callable,
    with_indices: bool,
    input_columns: None | str | list[str],
    batched: bool,
    batch_size: int,
    remove_columns: None | str | list[str],
    total_num_rows: None | int,
    **kwargs,
) -> "Step":
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        from .step import LazyRows

        dataset: Dataset | IterableDataset
        if isinstance(step.output, OutputDataset) and lazy:
            dataset = step.output.dataset.to_iterable_dataset()
        else:
            dataset = step.output.dataset
        if isinstance(dataset, Dataset):
            DataDreamer._enable_hf_datasets_logging()
            map_results = dataset.map(
                partial(_user_transform, self, step, function, with_indices, batched),
                with_indices=True,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                remove_columns=remove_columns,
                writer_batch_size=kwargs["writer_batch_size"],
                num_proc=min(default_to(kwargs["save_num_proc"], 1), len(dataset)),
                desc=f"'{self.name}'",
            )
            DataDreamer._disable_hf_datasets_logging()
            return map_results
        else:
            return LazyRows(
                dataset.map(
                    partial(
                        _user_transform, self, step, function, with_indices, batched
                    ),
                    with_indices=True,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                    remove_columns=remove_columns,
                ),
                total_num_rows=total_num_rows,
                auto_progress=False if total_num_rows is None else True,
            )

    from .step import MapStep

    kwargs["op_cls"] = MapStep
    kwargs["op_name"] = "map"
    kwargs["no_save"] = lazy
    kwargs["args"] = {
        "fingerprint": [
            step.fingerprint,
            stable_fingerprint(function),
            with_indices,
            input_columns,
            batched,
            batch_size,
            remove_columns,
        ]
    }
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_filter_step(
    function: Callable,
    with_indices: bool,
    input_columns: None | str | list[str],
    batched: bool,
    batch_size: int,
    total_num_rows: None | int,
    **kwargs,
) -> "Step":
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        from .step import LazyRows

        dataset: Dataset | IterableDataset
        if isinstance(step.output, OutputDataset) and lazy:
            dataset = step.output.dataset.to_iterable_dataset()
        else:
            dataset = step.output.dataset
        if isinstance(dataset, Dataset):
            DataDreamer._enable_hf_datasets_logging()
            filter_results = dataset.filter(
                partial(_user_transform, self, step, function, with_indices, batched),
                with_indices=True,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                writer_batch_size=kwargs["writer_batch_size"],
                num_proc=min(default_to(kwargs["save_num_proc"], 1), len(dataset)),
                desc=f"'{self.name}'",
            )
            DataDreamer._disable_hf_datasets_logging()
            return filter_results
        else:
            return LazyRows(
                dataset.filter(
                    partial(
                        _user_transform, self, step, function, with_indices, batched
                    ),
                    with_indices=True,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                ),
                total_num_rows=total_num_rows,
                auto_progress=False if total_num_rows is None else True,
            )

    from .step import FilterStep

    kwargs["op_cls"] = FilterStep
    kwargs["op_name"] = "filter"
    kwargs["no_save"] = lazy
    kwargs["args"] = {
        "fingerprint": [
            step.fingerprint,
            stable_fingerprint(function),
            with_indices,
            input_columns,
            batched,
            batch_size,
        ]
    }
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_rename_column_step(
    original_column_name: str, new_column_name: str, **kwargs
):
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        if original_column_name == new_column_name:
            return step.output.dataset
        else:
            return step.output.dataset.rename_column(
                original_column_name=original_column_name,
                new_column_name=new_column_name,
            )

    from .step import RenameColumnStep

    kwargs["op_cls"] = RenameColumnStep
    kwargs["op_name"] = "rename_column"
    kwargs["no_save"] = lazy
    kwargs["args"] = {
        "fingerprint": [step.fingerprint, original_column_name, new_column_name]
    }
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_rename_columns_step(column_mapping: dict[str, str], **kwargs):
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        new_column_mapping = {k: v for k, v in column_mapping.items() if k != v}
        if len(new_column_mapping) > 0:
            return step.output.dataset.rename_columns(column_mapping=new_column_mapping)
        else:
            return step.output.dataset

    from .step import RenameColumnsStep

    kwargs["op_cls"] = RenameColumnsStep
    kwargs["op_name"] = "rename_columns"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [step.fingerprint, column_mapping]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_remove_columns_step(column_names: str | list[str], **kwargs):
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        return step.output.dataset.remove_columns(column_names=column_names)

    from .step import RemoveColumnsStep

    kwargs["op_cls"] = RemoveColumnsStep
    kwargs["op_name"] = "remove_columns"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [step.fingerprint, column_names]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_splits_step(
    train_size: None | float | int,
    validation_size: None | float | int,
    test_size: None | float | int,
    stratify_by_column: None | str,
    **kwargs,
):
    step = kwargs["step"]
    _, dataset_dict = _step_to_dataset_dict(
        step=step,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        stratify_by_column=stratify_by_column,
        writer_batch_size=kwargs["writer_batch_size"],
        save_num_proc=kwargs["save_num_proc"],
        save_num_shards=kwargs["save_num_shards"],
    )

    def run(dataset_dict, split, self):
        return dataset_dict[split]

    from .step import SplitStep

    kwargs["op_cls"] = SplitStep
    kwargs["no_save"] = False
    kwargs["args"] = {
        "fingerprint": [
            step.fingerprint,
            train_size,
            validation_size,
            test_size,
            stratify_by_column,
        ]
    }
    name = kwargs["name"]
    del kwargs["name"]
    return {
        split: partial(
            __create_step_operation_step,
            run=partial(run, dataset_dict, split),
            op_name=f"{split} split",
            name=f"{name} ({split})" if name else None,
            **kwargs,
        )()
        for split in dataset_dict.keys()
    }


def _create_shard_step(num_shards: int, index: int, contiguous: bool, **kwargs):
    step = kwargs["step"]

    def run(self):
        dataset: Dataset
        if isinstance(step.output, OutputIterableDataset):
            dataset = _iterable_dataset_to_dataset(
                self=self,
                step=step,
                iterable_dataset=step.output.dataset,
                writer_batch_size=kwargs["writer_batch_size"],
                save_num_proc=kwargs["save_num_proc"],
            )
        else:
            dataset = step.output.dataset
        DataDreamer._enable_hf_datasets_logging()
        shard_results = dataset.shard(
            num_shards=num_shards,
            index=index,
            contiguous=contiguous,
            writer_batch_size=kwargs["writer_batch_size"],
        )
        DataDreamer._disable_hf_datasets_logging()
        return shard_results

    from .step import ShardStep

    kwargs["op_cls"] = ShardStep
    kwargs["op_name"] = "shard"
    kwargs["no_save"] = False
    kwargs["args"] = {"fingerprint": [step.fingerprint, num_shards, index, contiguous]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_reverse_step(**kwargs):
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]

    def run(self):
        dataset: Dataset
        if isinstance(step.output, OutputIterableDataset):
            dataset = _iterable_dataset_to_dataset(
                self=self,
                step=step,
                iterable_dataset=step.output.dataset,
                writer_batch_size=kwargs["writer_batch_size"],
                save_num_proc=kwargs["save_num_proc"],
            )
        else:
            dataset = step.output.dataset

        def reverse_generator(dataset):
            for row in reversed(dataset):
                yield row

        return IterableDataset.from_generator(
            partial(reverse_generator, dataset), features=step.output._features
        )

    from .step import ReverseStep

    kwargs["op_cls"] = ReverseStep
    kwargs["op_name"] = "reverse"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": [step.fingerprint]}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_save_step(**kwargs) -> "Step":
    step = kwargs["step"]

    def run(self):
        if isinstance(step.output, OutputDataset):
            return step.output
        else:
            return step.output.dataset

    from .step import SaveStep

    kwargs["op_cls"] = SaveStep
    kwargs["op_name"] = "save"
    kwargs["no_save"] = False
    kwargs["args"] = {"fingerprint": step.fingerprint}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


def _create_copy_step(**kwargs) -> "Step":
    lazy, step = kwargs["lazy"], kwargs["step"]
    del kwargs["lazy"]
    wait(step)

    def run(self):
        if lazy:
            if isinstance(step.output, OutputDataset):
                return step.output.dataset.to_iterable_dataset()
            else:
                return step.output.dataset
        else:
            if isinstance(step.output, OutputDataset):
                return step.output
            else:
                dataset = _iterable_dataset_to_dataset(
                    self=self,
                    step=step,
                    iterable_dataset=step.output.dataset,
                    writer_batch_size=kwargs["writer_batch_size"],
                    save_num_proc=kwargs["save_num_proc"],
                )
                return dataset

    from .step import CopyStep

    kwargs["op_cls"] = CopyStep
    kwargs["op_name"] = "copy"
    kwargs["no_save"] = lazy
    kwargs["args"] = {"fingerprint": step.fingerprint}
    kwargs["run"] = run
    return partial(__create_step_operation_step, **kwargs)()


__all__ = [
    "_INTERNAL_STEP_OPERATION_KEY",
    "_step_to_dataset_dict",
    "_create_select_step",
    "_create_select_columns_step",
    "_create_take_step",
    "_create_skip_step",
    "_create_shuffle_step",
    "_create_sort_step",
    "_create_add_item_step",
    "_create_map_step",
    "_create_filter_step",
    "_create_rename_column_step",
    "_create_rename_columns_step",
    "_create_remove_columns_step",
    "_create_splits_step",
    "_create_shard_step",
    "_create_reverse_step",
    "_create_save_step",
    "_create_copy_step",
]
