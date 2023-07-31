from functools import cached_property

from datasets import DatasetDict, IterableDatasetDict, load_dataset
from datasets.fingerprint import Hasher
from datasets.splits import Split
from datasets.utils.version import Version

from ..step_operations import _INTERNAL_STEP_OPERATION_KEY
from .data_source import DataSource


class HFHubDataSource(DataSource):
    def __init__(
        self,
        name: str,
        path: str,
        config_name: None | str = None,
        split: None | str | Split = None,
        revision: None | str | Version = None,
        streaming: bool = False,
        progress_interval: None | int = None,
        force: bool = False,
        verbose: bool = False,
        log_level: None | int = None,
        save_num_proc: None | int = None,
        save_num_shards: None | int = None,
        background: bool = False,
        **config_kwargs
    ):
        self.path = path
        self.config_name = config_name
        self.split = split
        self.revision = revision
        self.streaming = streaming
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
        result = load_dataset(
            path=self.path,
            name=self.config_name,
            split=self.split,
            revision=self.revision,
            streaming=self.streaming,
            **self.config_kwargs
        )
        if isinstance(result, (DatasetDict, IterableDatasetDict)):
            raise ValueError(
                "Choose a split with the `split=...` parameter,"
                " multiple splits are not supported."
            )
        return result

    @cached_property
    def fingerprint(self) -> str:
        return Hasher.hash(
            [
                super().fingerprint,
                self.path,
                self.config_name,
                self.split,
                self.revision,
                self.streaming,
                self.config_kwargs,
            ]
        )


setattr(HFHubDataSource, _INTERNAL_STEP_OPERATION_KEY, True)

__all__ = ["HFHubDataSource"]
