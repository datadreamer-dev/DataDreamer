from functools import cached_property
from typing import Sequence

from datasets import DatasetDict, load_dataset
from datasets.fingerprint import Hasher

from ..step_operations import _INTERNAL_STEP_OPERATION_KEY
from .data_source import DataSource


class CSVDataSource(DataSource):
    """Loads a CSV dataset from a local path. See :py:func:`datasets.load_dataset` for
    more details.

    Args:
        name: The name of the step.
        data_folder: The path to the dataset folder.
        data_files: The name of files from the folder to load.
        progress_interval: How often to log progress in seconds.
        force: Whether to force run the step (ignore saved results).
        verbose: Whether or not to print verbose logs.
        log_level: The logging level to use (:py:data:`~logging.DEBUG`, :py:data:`~logging.INFO`, etc.).
        save_num_proc: The number of processes to use if saving to disk.
        save_num_shards: The number of shards on disk to save the dataset into.
        background: Whether to run the operation in the background.
        **config_kwargs: Additional keyword arguments to pass to
            :py:func:`datasets.load_dataset`.
    """

    def __init__(
        self,
        name: str,
        data_folder: None | str = None,
        data_files: None | str | Sequence[str] = None,
        sep: str = ",",
        progress_interval: None | int = None,
        force: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
        **config_kwargs,
    ):
        self.data_folder = data_folder
        self.data_files = data_files
        self.sep = sep
        self.config_kwargs = config_kwargs
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
        if isinstance(self.data_files, dict):
            raise ValueError(
                "You supplied a dict to data_files, multiple splits are not supported."
            )
        result = load_dataset(
            "csv",
            data_dir=self.data_folder,
            data_files=self.data_files,
            num_proc=self.save_num_proc,
            sep=self.sep,
            **self.config_kwargs,
        )
        if isinstance(result, DatasetDict):
            result = result["train"]
        return result

    @cached_property
    def fingerprint(self) -> str:
        return Hasher.hash(
            [super().fingerprint, self.data_folder, self.data_files, self.config_kwargs]
        )


setattr(CSVDataSource, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["CSVDataSource"]
