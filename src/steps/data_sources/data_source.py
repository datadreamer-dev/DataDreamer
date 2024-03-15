import warnings
from functools import cached_property
from typing import Callable, Generator

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from datasets.fingerprint import Hasher

from ..step import Any, LazyRows, Step
from ..step_operations import _INTERNAL_STEP_OPERATION_KEY


class DataSource(Step):
    def __init__(
        self,
        name: str,
        data: dict
        | list
        | Callable[[], Generator[dict[str, Any], None, None]]
        | Dataset
        | IterableDataset,
        total_num_rows: None | int = None,
        auto_progress: bool = True,
        progress_interval: None | int = None,
        force: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
        **kwargs,
    ):
        """Loads a dataset from a in-memory Python object.

        Args:
            name: The name of the step.
            data (Any): The data to load as a dataset. See the
                `valid return formats <./pages/advanced_usage/creating_a_new_datadreamer_.../step.html#returning-outputs>`_
                for more details on acceptable data formats.
            total_num_rows: The total number of rows being processed (helps with
                displaying progress, if the data is being lazily loaded).
            auto_progress: Whether to automatically update the progress % for this step.
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
        self.dataset = data
        if (
            auto_progress
            and (isinstance(data, IterableDataset) or callable(data))
            and total_num_rows is None
        ):
            warnings.warn(
                "You did not specify `total_num_rows`, so we cannot"
                " automatically update the progress % for this step. Either"
                " specify DataSource(..., total_num_rows=#) or, to disable"
                " this warning, specify DataSource(.., auto_progress = False)",
                stacklevel=2,
            )
        self.total_num_rows = total_num_rows
        self.kwargs = kwargs
        super().__init__(
            name,
            inputs={},
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
        dataset = self.dataset
        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            raise ValueError("You supplied a DatasetDict, supply a Dataset instead.")
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset, IterableDataset):
            return LazyRows(
                dataset,
                total_num_rows=self.total_num_rows,
                auto_progress=False,
                **self.kwargs,
            )
        elif isinstance(dataset, dict):
            return Dataset.from_dict(dataset)
        elif isinstance(dataset, list):
            return Dataset.from_list(dataset)
        elif callable(dataset):
            return LazyRows(
                dataset,
                total_num_rows=self.total_num_rows,
                auto_progress=False,
                **self.kwargs,
            )

    @cached_property
    def fingerprint(self) -> str:
        return Hasher.hash([super().fingerprint, self.dataset])


setattr(DataSource, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["DataSource"]
