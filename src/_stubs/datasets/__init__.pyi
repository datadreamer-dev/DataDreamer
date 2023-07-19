from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import numpy

class Dataset:
    @classmethod
    def from_dict(cls, v: dict) -> Dataset: ...
    @classmethod
    def from_list(cls, v: list) -> Dataset: ...
    @property
    def column_names(self) -> None | list[str]: ...
    @property
    def info(self) -> Any: ...
    def list_indexes(self) -> list[str]: ...
    def drop_index(self, index_name: str): ...
    def reset_format(self) -> None: ...
    def shuffle(
        self,
        seed=None,
        generator: None | numpy.random._generator.Generator = None,
        buffer_size: int = 1000,
    ) -> Dataset: ...
    def take(self, n: int) -> Dataset: ...
    def select_columns(
        self, column_names: str | list[str], new_fingerprint: None | str = None
    ) -> Dataset: ...
    def rename_columns(self, column_mapping: dict[str, str]): ...
    def to_iterable_dataset(self, num_shards: None | int = 1) -> IterableDataset: ...
    def save_to_disk(self, path: str) -> None: ...
    @classmethod
    def load_from_disk(cls, path) -> Dataset: ...
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
    def shuffle(
        self,
        seed=None,
        generator: None | numpy.random._generator.Generator = None,
        buffer_size: int = 1000,
    ) -> IterableDataset: ...
    def take(self, n: int) -> IterableDataset: ...
    def select_columns(
        self, column_names: str | list[str], new_fingerprint: None | str = None
    ) -> IterableDataset: ...
    def rename_columns(self, column_mapping: dict[str, str]): ...
    def __iter__(self): ...
