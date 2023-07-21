import os
from functools import partial
from typing import TYPE_CHECKING

from datasets import Dataset

from .. import DataDreamer
from ..datasets import OutputDataset
from ..logging import logger

if TYPE_CHECKING:  # pragma: no cover
    from .step import Step

_INTERNAL_STEP_OPERATION_KEY = "__DataDreamer__step_operation__"

##################################
# Constructors for step operations
##################################


def _create_save_step(
    step: "Step",
    name: None | str,
    writer_batch_size: None | int,
    num_proc: None | int,
) -> "Step":
    from .step import SaveStep

    class _SaveStep(SaveStep):
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
                    num_proc=num_proc,
                )
                self._pickled = step.output._pickled
                return dataset

    setattr(_SaveStep, _INTERNAL_STEP_OPERATION_KEY, True)
    _SaveStep.__name__ = SaveStep.__name__
    _SaveStep.__qualname__ = SaveStep.__name__
    _SaveStep.__module__ = SaveStep.__module__
    final_name: str = name or DataDreamer._new_step_name(step.name, "save")
    save_step = _SaveStep(name=final_name, args={"fingerprint": step.fingerprint})
    return save_step


__all__ = ["_INTERNAL_STEP_OPERATION_KEY", "_create_save_step"]
