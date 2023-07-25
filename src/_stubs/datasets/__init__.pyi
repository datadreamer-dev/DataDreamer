from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Sequence

from datasets.features.features import Features

class Dataset:
    @classmethod
    def from_dict(cls, v: dict) -> Dataset: ...
    @classmethod
    def from_list(cls, v: list) -> Dataset: ...
    @classmethod
    def from_generator(
        cls,
        generator: Callable,
        features: None | Features,
        cache_dir: None | str,
        writer_batch_size: None | int = 1000,
        num_proc: None | int = None,
    ) -> Dataset: ...
    @property
    def column_names(self) -> None | list[str]: ...
    @property
    def info(self) -> Any: ...
    def list_indexes(self) -> list[str]: ...
    def drop_index(self, index_name: str): ...
    def reset_format(self) -> None: ...
    def select(
        self, indices: Iterable, writer_batch_size: None | int = 1000
    ) -> Dataset: ...
    def select_columns(self, column_names: str | list[str]) -> Dataset: ...
    def shuffle(
        self,
        seed=None,
        writer_batch_size: None | int = 1000,
    ) -> Dataset: ...
    def sort(
        self,
        column_names: str | Sequence[str],
        reverse: bool | Sequence[bool] = False,
        null_placement: str = "at_end",
    ) -> Dataset: ...
    def add_item(self, item: dict) -> Dataset: ...
    def map(
        self,
        function: None | Callable = None,
        with_indices: bool = False,
        input_columns: None | str | list[str] = None,
        batched: bool = False,
        batch_size: int = 1000,
        remove_columns: None | str | list[str] = None,
        writer_batch_size: None | int = 1000,
        num_proc: None | int = None,
        desc: None | str = None,
    ) -> Dataset: ...
    def filter(
        self,
        function: None | Callable = None,
        with_indices: bool = False,
        input_columns: None | str | list[str] = None,
        batched: bool = False,
        batch_size: int = 1000,
        writer_batch_size: None | int = 1000,
        num_proc: None | int = None,
        desc: None | str = None,
    ) -> Dataset: ...
    def rename_column(
        self, original_column_name: str, new_column_name: str
    ) -> Dataset: ...
    def rename_columns(self, column_mapping: dict[str, str]) -> Dataset: ...
    def remove_columns(self, column_names: str | list[str]) -> Dataset: ...
    def shard(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = False,
        writer_batch_size: None | int = 1000,
    ) -> Dataset: ...
    def save_to_disk(
        self, path: str, num_proc: None | int = None, num_shards: None | int = None
    ) -> None: ...
    @classmethod
    def load_from_disk(cls, path) -> Dataset: ...
    def to_iterable_dataset(self, num_shards: None | int = 1) -> IterableDataset: ...
    def __iter__(self): ...
    def __getitem__(self, index: int | slice | str | Iterable[int]) -> Any: ...
    def __len__(self): ...

class IterableDataset:
    @classmethod
    def from_generator(
        cls, v: Callable, features: None | Any = None
    ) -> IterableDataset: ...
    @property
    def column_names(self) -> None | list[str]: ...
    @property
    def info(self) -> Any: ...
    def select_columns(self, column_names: str | list[str]) -> IterableDataset: ...
    def take(self, n: int) -> IterableDataset: ...
    def skip(self, n: int) -> IterableDataset: ...
    def shuffle(
        self,
        seed=None,
        buffer_size: int = 1000,
    ) -> IterableDataset: ...
    def map(
        self,
        function: None | Callable = None,
        with_indices: bool = False,
        input_columns: None | str | list[str] = None,
        batched: bool = False,
        batch_size: int = 1000,
        remove_columns: None | str | list[str] = None,
    ) -> IterableDataset: ...
    def filter(
        self,
        function: None | Callable = None,
        with_indices: bool = False,
        input_columns: None | str | list[str] = None,
        batched: bool = False,
        batch_size: int = 1000,
    ) -> IterableDataset: ...
    def rename_column(
        self, original_column_name: str, new_column_name: str
    ) -> IterableDataset: ...
    def rename_columns(self, column_mapping: dict[str, str]) -> IterableDataset: ...
    def remove_columns(self, column_names: str | list[str]) -> IterableDataset: ...
    def __iter__(self): ...

def concatenate_datasets(
    dsets: list[Dataset | IterableDataset], axis: int = 0
) -> Dataset | IterableDataset: ...
