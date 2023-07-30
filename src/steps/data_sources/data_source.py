from functools import cached_property
from typing import Callable, Generator

from datasets import Dataset, DatasetDict, IterableDataset
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
        verbose: bool = False,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ):
        self.dataset = data
        self.total_num_rows = total_num_rows
        self.auto_progress = auto_progress
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
        if isinstance(dataset, DatasetDict):
            raise ValueError("You supplied a DatasetDict, supply a Dataset instead.")
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset, IterableDataset):
            return LazyRows(
                dataset,
                total_num_rows=self.total_num_rows,
                auto_progress=self.auto_progress,
            )
        elif isinstance(dataset, dict):
            return Dataset.from_dict(dataset)
        elif isinstance(dataset, list):
            return Dataset.from_list(dataset)
        elif callable(dataset):
            return LazyRows(
                dataset,
                total_num_rows=self.total_num_rows,
                auto_progress=self.auto_progress,
            )

    @cached_property
    def fingerprint(self) -> str:
        return Hasher.hash([super().fingerprint, self.dataset])


setattr(DataSource, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["DataSource"]
