import os
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Callable, Type, cast

from pyarrow.lib import ArrowInvalid, ArrowTypeError

from datasets import Dataset, IterableDataset, concatenate_datasets
from datasets.builder import DatasetGenerationError
from datasets.fingerprint import Hasher

from .. import DataDreamer
from ..datasets import OutputDataset, OutputIterableDataset
from ..errors import StepOutputTypeError
from ..logging import logger
from ..pickling import unpickle_transform
from .step_background import wait

if TYPE_CHECKING:  # pragma: no cover
    from .step import Step

_INTERNAL_STEP_OPERATION_KEY = "__DataDreamer__step_operation__"
_INTERNAL_STEP_OPERATION_NO_SAVE_KEY = "__DataDreamer__step_operation_no_save__"


##################################
# Helpers step operations
##################################


def _iterable_dataset_to_dataset(
    self,
    step: "Step",
    iterable_dataset: IterableDataset,
    writer_batch_size: None | int,
    save_num_proc: None | int,
) -> Dataset:
    def dataset_generator(iterable_dataset):
        i = None
        for i, row in enumerate(iterable_dataset):
            if step.output.num_rows is not None:
                self.progress = (i + 1) / step.output.num_rows
            else:
                self._set_progress_rows(i + 1)
            yield row
        if i is not None:
            self._set_progress_rows(i + 1)
        self.progress = 1.0

    logger.debug(
        f"Iterating through all of '{step.name}''s lazy results in"
        " preparation for saving."
    )

    try:
        if step._output_folder_path:
            cache_path = os.path.join(step._output_folder_path, "cache")
            dataset = Dataset.from_generator(
                partial(dataset_generator, iterable_dataset),
                features=step.output._features,
                cache_dir=cache_path,
                writer_batch_size=writer_batch_size,
                num_proc=save_num_proc,
            )
        else:
            dataset = Dataset.from_list(list(dataset_generator(iterable_dataset)))
    except DatasetGenerationError as e:
        raise e.__cause__
    self._pickled = step.output._pickled
    return dataset


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
    op_cls: Type["Step"],
    op_name: str,
    run: Callable,
    no_save: bool = False,
    setup: None | Callable = None,
    **kwargs,
) -> "Step":
    from .step import LazyRows

    if kwargs.get("progress_interval", None) is None:
        kwargs["progress_interval"] = step.progress_interval
    writer_batch_size = 1000
    if kwargs.get("writer_batch_size", None) is not None:
        writer_batch_size = kwargs["writer_batch_size"]
    del kwargs["writer_batch_size"]
    if kwargs.get("save_num_proc", None) is None:
        kwargs["save_num_proc"] = step.save_num_proc
    if kwargs.get("save_num_shards", None) is None:
        kwargs["save_num_shards"] = step.save_num_shards

    # Hash the fingerprint
    kwargs["args"]["fingerprint"] = Hasher.hash(kwargs["args"]["fingerprint"])

    class _StepOpStep(op_cls):  # type:ignore[valid-type,misc]
        def setup(self):
            self.register_arg("fingerprint")
            if callable(setup):
                setup(self)  # pragma: no cover

        def run(self):
            try:
                run_output = run(self)
            except (ArrowInvalid, ArrowTypeError, ValueError, TypeError) as e:
                raise StepOutputTypeError(str(e))

            if isinstance(run_output, LazyRows):
                return run_output

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
                return LazyRows(run_output, total_num_rows=step.output.num_rows)
            else:
                return run_output

    setattr(_StepOpStep, _INTERNAL_STEP_OPERATION_KEY, True)
    if no_save:
        setattr(_StepOpStep, _INTERNAL_STEP_OPERATION_NO_SAVE_KEY, True)
    _StepOpStep.__name__ = op_cls.__name__
    _StepOpStep.__qualname__ = op_cls.__name__
    _StepOpStep.__module__ = op_cls.__module__
    final_name: str = name or DataDreamer._new_step_name(step.name, transform=op_name)
    wait(step)
    return _StepOpStep(
        name=final_name,
        verbose=step.verbose,
        log_level=step.log_level,
        **kwargs,
    )


def __concatenate(  # noqa: C901
    *steps: "Step",
    name: None | str,
    lazy: bool,
    progress_interval: None | int,
    force: bool,
    writer_batch_size: None | int,
    save_num_proc: None | int,
    save_num_shards: None | int,
    background: bool,
    op_cls: Type["Step"],
    op_name: str,
    axis: int,
):
    from .step import LazyRows, Step

    if len(steps) == 0:
        raise ValueError(f"You must provide at least one step to {op_name}().")
    if not all([isinstance(step, Step) for step in steps]):
        raise TypeError(f"All arguments to {op_name}() must be of type Step.")

    if name is None:
        step_names = ", ".join([step.name for step in steps])
        name = DataDreamer._new_step_name(op_name + f"({step_names})")

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
                            writer_batch_size=writer_batch_size,
                            save_num_proc=save_num_proc,
                        )
                    )
            return concatenate_datasets(datasets, axis=axis)
        else:
            total_num_rows: None | int = 0
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
                total_num_rows=total_num_rows,
            )

    return partial(
        __create_step_operation_step,
        step=steps[0],
        name=name,
        op_cls=op_cls,
        op_name=op_name,
        run=run,
        no_save=lazy,
        args={"fingerprint": [step.fingerprint for step in steps]},
        progress_interval=progress_interval,
        force=force,
        writer_batch_size=writer_batch_size,
        save_num_proc=save_num_proc,
        save_num_shards=save_num_shards,
        background=background,
    )()


def _create_save_step(
    name: None | str,
    progress_interval: None | int,
    force: bool,
    writer_batch_size: None | int,
    save_num_proc: None | int,
    save_num_shards: None | int,
    background: bool,
    step: "Step",
) -> "Step":
    from .step import SaveStep

    def run(self):
        if isinstance(step.output, OutputDataset):
            return step.output
        else:
            return step.output.dataset

    return partial(
        __create_step_operation_step,
        step=step,
        name=name,
        op_cls=SaveStep,
        op_name="save",
        run=run,
        no_save=False,
        args={"fingerprint": step.fingerprint},
        progress_interval=progress_interval,
        force=force,
        writer_batch_size=writer_batch_size,
        save_num_proc=save_num_proc,
        save_num_shards=save_num_shards,
        background=background,
    )()


def _create_map_step(
    function: Callable,
    with_indices: bool,
    input_columns: None | str | list[str],
    batched: bool,
    batch_size: int,
    remove_columns: None | str | list[str],
    name: None | str,
    lazy: bool,
    progress_interval: None | int,
    force: bool,
    writer_batch_size: None | int,
    save_num_proc: None | int,
    save_num_shards: None | int,
    background: bool,
    step: "Step",
) -> "Step":
    from .step import MapStep

    def run(self):
        dataset: Dataset | IterableDataset
        if isinstance(step.output, OutputDataset) and lazy:
            dataset = step.output.dataset.to_iterable_dataset()
        else:
            dataset = step.output.dataset
        if isinstance(dataset, Dataset):
            return dataset.map(
                partial(
                    _user_transform,
                    self,
                    step,
                    function,
                    with_indices,
                    batched,
                ),
                with_indices=True,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                remove_columns=remove_columns,
                writer_batch_size=writer_batch_size,
                num_proc=save_num_proc,
                desc=self.name,
            )
        else:
            return dataset.map(
                partial(
                    _user_transform,
                    self,
                    step,
                    function,
                    with_indices,
                    batched,
                ),
                with_indices=True,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                remove_columns=remove_columns,
            )

    return partial(
        __create_step_operation_step,
        step=step,
        name=name,
        op_cls=MapStep,
        op_name="map",
        run=run,
        no_save=lazy,
        args={
            "fingerprint": [
                step.fingerprint,
                function,
                with_indices,
                input_columns,
                batched,
                batch_size,
                remove_columns,
            ]
        },
        progress_interval=progress_interval,
        force=force,
        writer_batch_size=writer_batch_size,
        save_num_proc=save_num_proc,
        save_num_shards=save_num_shards,
        background=background,
    )()


def _create_shuffle_step(
    seed: None | int,
    buffer_size: int,
    name: None | str,
    lazy: bool,
    progress_interval: None | int,
    force: bool,
    writer_batch_size: None | int,
    save_num_proc: None | int,
    save_num_shards: None | int,
    background: bool,
    step: "Step",
) -> "Step":
    from .step import ShuffleStep

    def run(self):
        dataset: Dataset | IterableDataset
        if not lazy and isinstance(step.output, OutputIterableDataset):
            dataset = _iterable_dataset_to_dataset(
                self=self,
                step=step,
                iterable_dataset=step.output.dataset,
                writer_batch_size=writer_batch_size,
                save_num_proc=save_num_proc,
            )
        else:
            dataset = step.output.dataset
        if isinstance(dataset, Dataset):
            return dataset.shuffle(seed=seed, writer_batch_size=writer_batch_size)
        else:
            return dataset.shuffle(seed=seed, buffer_size=buffer_size)

    return partial(
        __create_step_operation_step,
        step=step,
        name=name,
        op_cls=ShuffleStep,
        op_name="shuffle",
        run=run,
        no_save=lazy,
        args={
            "fingerprint": [
                step.fingerprint,
                seed,
                buffer_size,
            ]
        },
        progress_interval=progress_interval,
        force=force,
        writer_batch_size=writer_batch_size,
        save_num_proc=save_num_proc,
        save_num_shards=save_num_shards,
        background=background,
    )()


__all__ = [
    "_INTERNAL_STEP_OPERATION_KEY",
    "_create_save_step",
    "_create_map_step",
    "_create_shuffle_step",
]
