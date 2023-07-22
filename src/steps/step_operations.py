import os
from functools import partial
from typing import TYPE_CHECKING, Callable, Type

from datasets import Dataset

from .. import DataDreamer
from ..datasets import OutputDataset
from ..logging import logger
from ..pickling import unpickle_transform

if TYPE_CHECKING:  # pragma: no cover
    from .step import Step

_INTERNAL_STEP_OPERATION_KEY = "__DataDreamer__step_operation__"


def _create_step_operation_step(
    step: "Step",
    name: None | str,
    op_cls: Type["Step"],
    op_name: str,
    run: Callable,
    setup: None | Callable = None,
    **kwargs,
) -> "Step":
    class _StepOpStep(op_cls):  # type:ignore[valid-type,misc]
        def setup(self):
            self.register_arg("fingerprint")
            if callable(setup):
                setup(self)  # pragma: no cover

        def run(self):
            run_output = run(self)
            if step._pickled:
                self.pickle(True)
            return run_output

    setattr(_StepOpStep, _INTERNAL_STEP_OPERATION_KEY, True)
    _StepOpStep.__name__ = op_cls.__name__
    _StepOpStep.__qualname__ = op_cls.__name__
    _StepOpStep.__module__ = op_cls.__module__
    final_name: str = name or DataDreamer._new_step_name(step.name, op_name)
    if kwargs.get("save_num_proc", None) is None:
        kwargs["save_num_proc"] = step.save_num_proc
    if kwargs.get("save_num_shards", None) is None:
        kwargs["save_num_shards"] = step.save_num_shards
    return _StepOpStep(
        name=final_name,
        **kwargs,
    )


##################################
# Constructors for step operations
##################################


def _create_save_step(
    writer_batch_size: None | int,
    name: None | str,
    save_num_proc: None | int,
    save_num_shards: None | int,
    step: "Step",
) -> "Step":
    from .step import SaveStep

    def run(self):
        output_folder_path = step._output_folder_path
        if not output_folder_path:  # pragma: no cover
            raise RuntimeError("You must run the Step in a DataDreamer() context.")

        if isinstance(step.output, OutputDataset):
            return step.output
        else:

            def dataset_generator(dataset):
                yield from dataset

            logger.debug(
                f"Iterating through all of '{step.name}''s lazy results in"
                " preparation for saving."
            )
            cache_path = os.path.join(output_folder_path, "cache")
            dataset = Dataset.from_generator(
                partial(dataset_generator, step.output.dataset),
                features=step.output._features,
                cache_dir=cache_path,
                writer_batch_size=writer_batch_size,
                num_proc=save_num_proc,
            )
            self._pickled = step.output._pickled
            return dataset

    return partial(
        _create_step_operation_step,
        step=step,
        name=name,
        op_cls=SaveStep,
        op_name="save",
        run=run,
        args={"fingerprint": step.fingerprint},
        save_num_proc=save_num_proc,
        save_num_shards=save_num_shards,
    )()


def _create_map_step(
    function: Callable,
    with_indices: bool,
    input_columns: None | str | list[str],
    batched: bool,
    batch_size: int,
    remove_columns: None | str | list[str],
    writer_batch_size: None | int,
    name: None | str,
    save_num_proc: None | int,
    save_num_shards: None | int,
    step: "Step",
) -> "Step":
    from .step import LazyRows, MapStep

    def map_transform(x, *args, **kwargs):
        if step.output._pickled or step.output._pickled_inferred:
            x = unpickle_transform(x, features=step.output._features, batched=batched)
        return function(x, *args, **kwargs)

    def run(self):
        if isinstance(step.output, OutputDataset):
            return step.output.dataset.map(
                map_transform,
                with_indices=with_indices,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                remove_columns=remove_columns,
                writer_batch_size=writer_batch_size,
                num_proc=save_num_proc,
                desc=self.name,
            )
        else:
            return LazyRows(
                step.output.dataset.map(
                    map_transform,
                    with_indices=with_indices,
                    input_columns=input_columns,
                    batched=batched,
                    batch_size=batch_size,
                    remove_columns=remove_columns,
                ),
                total_num_rows=step.output.total_num_rows,
            )

    return partial(
        _create_step_operation_step,
        step=step,
        name=name,
        op_cls=MapStep,
        op_name="map",
        run=run,
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
        save_num_proc=save_num_proc,
        save_num_shards=save_num_shards,
    )()


__all__ = ["_INTERNAL_STEP_OPERATION_KEY", "_create_save_step", "_create_map_step"]
