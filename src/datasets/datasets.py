from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pandas import DataFrame

from datasets import Dataset, IterableDataset
from datasets.features.features import Features, Value

from ..datasets.utils import get_column_names
from ..pickling import unpickle_transform

if TYPE_CHECKING:
    from ..steps import Step


class OutputDatasetMixin:
    @property
    def dataset(self) -> Dataset | IterableDataset:
        return self._dataset  # type:ignore[attr-defined]

    @property
    def column_names(self) -> list[str]:
        return get_column_names(self.dataset)

    @property
    def info(self) -> Any:
        return self.dataset.info

    def __iter__(self):
        if self.info and self.info.features:
            features = self.info.features
        else:
            features = Features()
        if self._pickled or self._pickled_inferred:  # type:ignore[attr-defined]
            for row in iter(self.dataset):
                yield unpickle_transform(row, features=features, batched=False)
        else:
            yield from iter(self.dataset)

    def __getitem__(self, key: int | slice | str | Iterable[int]) -> Any:
        if isinstance(key, str):
            if isinstance(self.dataset, Dataset):
                return OutputDatasetColumn(
                    self._step,  # type:ignore[attr-defined]
                    self.dataset.select_columns([key]),
                    pickled=self._pickled,  # type:ignore[attr-defined]
                )
            else:
                return OutputIterableDatasetColumn(
                    self._step,  # type:ignore[attr-defined]
                    self.dataset.select_columns([key]),
                    pickled=self._pickled,  # type:ignore[attr-defined]
                )
        if self._pickled or self._pickled_inferred:  # type:ignore[attr-defined]
            if self.info and self.info.features:
                features = self.info.features
            else:
                features = Features()
            if isinstance(key, int):
                return unpickle_transform(
                    self.dataset[key],  # type:ignore[index]
                    features=features,
                    batched=False,
                )
            else:
                return unpickle_transform(
                    self.dataset[key],  # type:ignore[index]
                    features=features,
                    batched=True,
                )
        else:
            return self.dataset[key]  # type:ignore[index]

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


class OutputDatasetColumnMixin:
    def __iter__(self):
        column_name = self.column_names[0]  # type:ignore[attr-defined]
        for row in super().__iter__():  # type:ignore[misc]
            yield row[column_name]

    def __getitem__(self, key: int | slice | str | Iterable[int]) -> Any:
        column_name = self.column_names[0]  # type:ignore[attr-defined]
        if (
            self._pickled or self._pickled_inferred  # type:ignore[attr-defined]
        ) and isinstance(key, str):
            info = self.info  # type:ignore[attr-defined]
            dataset = self.dataset  # type:ignore[attr-defined]
            if info and info.features:
                features = info.features
            else:
                features = Features()
            return (
                unpickle_transform({key: dataset[key]}, features=features, batched=True)
            )[key]
        else:
            return super().__getitem__(key)[column_name]  # type:ignore[misc]


class OutputIterableDataset(OutputDatasetMixin):
    def __init__(self, step: "Step", dataset: IterableDataset, pickled: bool = False):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError("Expected Step, got {type(step)}.")
        if not isinstance(dataset, IterableDataset):
            raise ValueError(f"Expected IterableDataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: IterableDataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False
        if self.info and self.info.features:
            features = self.info.features
        else:
            features = Features()
        for f in features.values():
            if isinstance(f, Value) and f.dtype == "binary":
                self._pickled_inferred = True
                break


class OutputDataset(OutputDatasetMixin):
    def __init__(self, step: "Step", dataset: Dataset, pickled: bool = False):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError("Expected Step, got {type(step)}.")
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Expected Dataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: Dataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False

    def __len__(self):
        return len(self.dataset)  # type:ignore[arg-type]


class OutputIterableDatasetColumn(OutputDatasetColumnMixin, OutputIterableDataset):
    def __init__(self, step: "Step", dataset: IterableDataset, pickled: bool = False):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError("Expected Step, got {type(step)}.")
        if not isinstance(dataset, IterableDataset):
            raise ValueError(f"Expected IterableDataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: IterableDataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False
        if self.info and self.info.features:
            features = self.info.features
        else:
            features = Features()
        for f in features.values():
            if isinstance(f, Value) and f.dtype == "binary":
                self._pickled_inferred = True
                break
        if len(self.column_names) != 1:
            raise ValueError(f"Expected single column only, got {self.column_names}")


class OutputDatasetColumn(OutputDatasetColumnMixin, OutputDataset):
    def __init__(self, step: "Step", dataset: Dataset, pickled: bool = False):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError("Expected Step, got {type(step)}.")
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Expected Dataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: Dataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False
        if len(self.column_names) != 1:
            raise ValueError(f"Expected single column only, got {self.column_names}")
