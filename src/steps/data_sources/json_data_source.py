from functools import cached_property
from typing import Sequence

from datasets import DatasetDict, load_dataset
from datasets.fingerprint import Hasher

from ..step_operations import _INTERNAL_STEP_OPERATION_KEY
from .data_source import DataSource


class JSONDataSource(DataSource):
    def __init__(
        self,
        name: str,
        data_dir: None | str = None,
        data_files: None | str | Sequence[str] = None,
        progress_interval: None | int = None,
        force: bool = False,
        verbose: bool = False,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
        **config_kwargs
    ):
        self.data_dir = data_dir
        self.data_files = data_files
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
                "You supplied a dict to data_files, supply just a single"
                " list, multiple splits are not supported."
            )
        result = load_dataset(
            "json",
            data_dir=self.data_dir,
            data_files=self.data_files,
            num_proc=self.save_num_proc,
            **self.config_kwargs
        )
        if isinstance(result, DatasetDict):
            result = result["train"]
        return result

    @cached_property
    def fingerprint(self) -> str:
        return Hasher.hash(
            [super().fingerprint, self.data_dir, self.data_files, self.config_kwargs]
        )


setattr(JSONDataSource, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["JSONDataSource"]
