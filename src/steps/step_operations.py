import os
from functools import partial
from typing import TYPE_CHECKING, Callable, Type

from datasets import Dataset

from .. import DataDreamer
from ..datasets import OutputDataset
from ..logging import logger

if TYPE_CHECKING:  # pragma: no cover
    from .step import Step

_INTERNAL_STEP_OPERATION_KEY = "__DataDreamer__step_operation__"


def _create_step_operation_step(
    step: "Step",
    name: None | str,
    op_cls: Type["Step"],
    op_name: str,
    setup: Callable,
    run: Callable,
    **kwargs,
) -> "Step":
    class _StepOpStep(op_cls):  # type:ignore[valid-type,misc]
        def setup(self):
            return setup(self)

        def run(self):
            return run(self)

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
    def setup(self):
        self.register_arg("fingerprint")

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

    from .step import SaveStep

    return partial(
        _create_step_operation_step,
        step=step,
        name=name,
        op_cls=SaveStep,
        op_name="save",
        setup=setup,
        run=run,
        args={"fingerprint": step.fingerprint},
        save_num_proc=save_num_proc,
        save_num_shards=save_num_shards,
    )()


__all__ = ["_INTERNAL_STEP_OPERATION_KEY", "_create_save_step"]
