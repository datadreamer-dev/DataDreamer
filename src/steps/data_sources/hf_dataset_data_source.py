from functools import cached_property

from datasets import Dataset
from datasets.fingerprint import Hasher

from ..step_operations import _INTERNAL_STEP_OPERATION_KEY
from .data_source import DataSource


class HFDatasetDataSource(DataSource):
    """Loads a Hugging Face :py:class:`~datasets.Dataset` from a local path. See
    :py:func:`datasets.load_from_disk` for more details.

    Args:
        name: The name of the step.
        dataset_path: The path to the :py:class:`datasets.Dataset` folder.
        progress_interval: How often to log progress in seconds.
        force: Whether to force run the step (ignore saved results).
        verbose: Whether or not to print verbose logs.
        log_level: The logging level to use (:py:data:`~logging.DEBUG`, :py:data:`~logging.INFO`, etc.).
        save_num_proc: The number of processes to use if saving to disk.
        save_num_shards: The number of shards on disk to save the dataset into.
        background: Whether to run the operation in the background.
    """

    def __init__(
        self,
        name: str,
        dataset_path: str,
        progress_interval: None | int = None,
        force: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ):
        self.path_to_dataset = dataset_path
        super().__init__(
            name,
            data=None,  # type: ignore[arg-type]
            progress_interval=progress_interval,
            force=force,
            verbose=verbose,
            log_level=log_level,
            save_num_proc=save_num_proc,
            save_num_shards=save_num_shards,
            background=background,
        )

    def setup(self):
        pass

    def run(self):
        return Dataset.load_from_disk(self.path_to_dataset)

    @cached_property
    def fingerprint(self) -> str:
        return Hasher.hash([super().fingerprint, self.path_to_dataset])


setattr(HFDatasetDataSource, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["HFDatasetDataSource"]
