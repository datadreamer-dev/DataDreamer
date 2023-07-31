from functools import cached_property

from datasets import Dataset, DatasetDict
from datasets.fingerprint import Hasher

from ..step_operations import _INTERNAL_STEP_OPERATION_KEY
from .data_source import DataSource


class HFDatasetDataSource(DataSource):
    def __init__(
        self,
        name: str,
        dataset_path: str,
        split: None | str = None,
        progress_interval: None | int = None,
        force: bool = False,
        verbose: bool = False,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
    ):
        self.dataset_path = dataset_path
        self.split = split
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
        result = Dataset.load_from_disk(self.dataset_path)
        if isinstance(result, DatasetDict) and self.split is None:
            raise ValueError(
                "You supplied a path to a DatasetDict, without specifying a split."
            )
        elif isinstance(result, DatasetDict):
            result = result[self.split]
        return result

    @cached_property
    def fingerprint(self) -> str:
        return Hasher.hash([super().fingerprint, self.dataset_path, self.split])


setattr(HFDatasetDataSource, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["HFDatasetDataSource"]
