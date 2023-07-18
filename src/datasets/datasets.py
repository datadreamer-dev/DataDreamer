from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pandas import DataFrame

from datasets import Dataset, IterableDataset
from datasets.features.features import Features, Value

from ..datasets.utils import get_column_names
from ..pickling import unpickle_transform

if TYPE_CHECKING:  # pragma: no cover
    from ..steps import Step


class OutputDatasetMixin:
    @property
    def step(self) -> "Step":
        return self._step  # type:ignore[attr-defined]

    @property
    def column_names(self) -> list[str]:
        return get_column_names(self.dataset)  # type:ignore[attr-defined]

    @property
    def info(self) -> Any:
        return self.dataset.info  # type:ignore[attr-defined]

    @property
    def _features(self) -> Features:
        if self.info and self.info.features:
            return self.info.features
        else:
            return Features()

    def __iter__(self):
        if self._pickled or self._pickled_inferred:  # type:ignore[attr-defined]
            for row in iter(self.dataset):  # type:ignore[attr-defined]
                yield unpickle_transform(row, features=self._features, batched=False)
        else:
            yield from iter(self.dataset)  # type:ignore[attr-defined]

    def __getitem__(self, key: int | slice | str | Iterable[int]) -> Any:
        if isinstance(key, str):
            feature = self._features.get(key, None)
            feature_is_pickled = False
            if isinstance(feature, Value) and feature.dtype == "binary":
                feature_is_pickled = True
            if isinstance(self.dataset, Dataset):  # type:ignore[attr-defined]
                return OutputDatasetColumn(
                    self._step,  # type:ignore[attr-defined]
                    self.dataset.select_columns([key]),  # type:ignore[attr-defined]
                    pickled=self._pickled  # type:ignore[attr-defined]
                    and feature_is_pickled,
                )
            else:
                return OutputIterableDatasetColumn(
                    self._step,  # type:ignore[attr-defined]
                    self.dataset.select_columns([key]),  # type:ignore[attr-defined]
                    pickled=self._pickled  # type:ignore[attr-defined]
                    and feature_is_pickled,
                )
        if self._pickled or self._pickled_inferred:  # type:ignore[attr-defined]
            if isinstance(key, int):
                return unpickle_transform(
                    self.dataset[key],  # type:ignore[attr-defined]
                    features=self._features,
                    batched=False,
                )
            else:
                return unpickle_transform(
                    self.dataset[key],  # type:ignore[attr-defined]
                    features=self._features,
                    batched=True,
                )
        else:
            return self.dataset[key]  # type:ignore[attr-defined]

    def head(self, n=5, shuffle=False, seed=None, buffer_size=1000) -> DataFrame:
        if isinstance(self.dataset, Dataset):  # type:ignore[attr-defined]
            iterable_dataset = (
                self.dataset.to_iterable_dataset()  # type:ignore[attr-defined]
            )
        else:
            iterable_dataset = self.dataset  # type:ignore[attr-defined]
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
        if isinstance(key, str):
            if isinstance(self, OutputIterableDatasetColumn):
                return iter(self)
            else:
                dataset = self.dataset  # type:ignore[attr-defined]
                if self._pickled or self._pickled_inferred:  # type:ignore[attr-defined]
                    return (
                        unpickle_transform(
                            {key: dataset[key]},
                            features=self._features,  # type:ignore[attr-defined]
                            batched=True,
                        )
                    )[key]
                else:
                    return dataset[key]
        else:
            return super().__getitem__(key)[column_name]  # type:ignore[misc]


class OutputIterableDataset(OutputDatasetMixin):
    def __init__(self, step: "Step", dataset: IterableDataset, pickled: bool = False):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError(f"Expected Step, got {type(step)}.")
        if not isinstance(dataset, IterableDataset):
            raise ValueError(f"Expected IterableDataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: IterableDataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False
        for f in self._features.values():
            if isinstance(f, Value) and f.dtype == "binary":
                self._pickled_inferred = True
                break

    @property
    def dataset(self) -> IterableDataset:
        return self._dataset


class OutputDataset(OutputDatasetMixin):
    def __init__(self, step: "Step", dataset: Dataset, pickled: bool = False):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError(f"Expected Step, got {type(step)}.")
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Expected Dataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: Dataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    def save_to_disk(self, path: str) -> None:
        self._dataset.save_to_disk(path)
        self._dataset = Dataset.load_from_disk(path)

    def __len__(self):
        return len(self.dataset)


class OutputIterableDatasetColumn(OutputDatasetColumnMixin, OutputIterableDataset):
    def __init__(self, step: "Step", dataset: IterableDataset, pickled: bool = False):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError(f"Expected Step, got {type(step)}.")
        if not isinstance(dataset, IterableDataset):
            raise ValueError(f"Expected IterableDataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: IterableDataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False
        for f in self._features.values():
            if isinstance(f, Value) and f.dtype == "binary":
                self._pickled_inferred = True
                break
        if len(self.column_names) != 1:
            raise ValueError(f"Expected single column only, got {self.column_names}")


class OutputDatasetColumn(OutputDatasetColumnMixin, OutputDataset):
    def __init__(self, step: "Step", dataset: Dataset, pickled: bool = False):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError(f"Expected Step, got {type(step)}.")
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Expected Dataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: Dataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False
        if len(self.column_names) != 1:
            raise ValueError(f"Expected single column only, got {self.column_names}")
