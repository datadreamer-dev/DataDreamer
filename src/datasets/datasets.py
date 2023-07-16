from typing import Any

from pandas import DataFrame

from datasets import Dataset, IterableDataset


class OutputDatasetMixin:
    @property
    def dataset(self) -> Dataset | IterableDataset:
        if hasattr(self, "_OutputIterableDataset__dataset"):
            return self._OutputIterableDataset__dataset
        else:
            return self._OutputDataset__dataset  # type:ignore[attr-defined]

    @property
    def column_names(self) -> None | list[str]:
        return self.dataset.column_names

    @property
    def info(self) -> Any:
        return self.dataset.info

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, key: Any) -> Any:
        return self.dataset[key]

    def head(self, n=5, shuffle=False, seed=None, buffer_size=1000) -> DataFrame:
        if isinstance(self.dataset, Dataset):
            iterable_dataset = self.dataset.to_iterable_dataset()
        else:
            iterable_dataset = self.dataset
        if shuffle:
            iterable_dataset = iterable_dataset.shuffle(
                seed=seed, buffer_size=buffer_size
            )
        return DataFrame.from_records(list(iterable_dataset.take(n)))


class OutputIterableDataset(OutputDatasetMixin):
    def __init__(self, dataset: IterableDataset):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            raise ValueError(f"Expected IterableDataset, got {type(dataset)}.")
        self.__dataset: IterableDataset = dataset


class OutputDataset(OutputDatasetMixin):
    def __init__(self, dataset: Dataset):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            raise ValueError(f"Expected Dataset, got {type(dataset)}.")
        self.__dataset: Dataset = dataset
