from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from datasets import Dataset, IterableDataset
from datasets.features.features import Features, Value
from datasets.fingerprint import Hasher
from pandas import DataFrame

from ..datasets.utils import get_column_names
from ..pickling import unpickle_transform

if TYPE_CHECKING:  # pragma: no cover
    from ..steps import Step


class OutputDatasetMixin:
    @property
    def step(self) -> "Step":
        """The step that produced the dataset."""
        return self._step  # type:ignore[attr-defined]

    @property
    def column_names(self) -> list[str]:
        """The column names in the dataset."""
        return get_column_names(self.dataset)  # type:ignore[attr-defined]

    @property
    def num_columns(self) -> int:
        """The number of columns in the dataset."""
        return len(self.column_names)

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
        """Get a row or column from the dataset.

        Args:
            key: The index or name of the column to get.

        Returns:
            The row or column from the dataset.
        """
        if isinstance(key, str):
            feature = self._features.get(key, None)
            feature_is_pickle_type = False
            if isinstance(feature, Value) and feature.dtype == "binary":
                feature_is_pickle_type = True
            if isinstance(self.dataset, Dataset):  # type:ignore[attr-defined]
                return OutputDatasetColumn(
                    self._step,  # type:ignore[attr-defined]
                    self.dataset.select_columns([key]),  # type:ignore[attr-defined]
                    pickled=self._pickled  # type:ignore[attr-defined]
                    and feature_is_pickle_type,
                )
            else:
                return OutputIterableDatasetColumn(
                    self._step,  # type:ignore[attr-defined]
                    self.dataset.select_columns([key]),  # type:ignore[attr-defined]
                    pickled=self._pickled  # type:ignore[attr-defined]
                    and feature_is_pickle_type,
                    total_num_rows=self.total_num_rows,  # type:ignore[attr-defined]
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

    @property
    def fingerprint(self) -> Any:
        return Hasher.hash((self.step.fingerprint, self.column_names))

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

    def __repr__(self) -> str:
        if isinstance(self, OutputDataset):
            return (
                f"{type(self).__name__}("
                f"column_names={str(self.column_names)}, "
                f"num_rows={len(self)}, "
                f"dataset=<{type(self.dataset).__name__} @ {id(self.dataset)}>"
                ")"
            )
        elif isinstance(self, OutputIterableDataset):
            return (
                f"{type(self).__name__}("
                f"column_names={str(self.column_names)}, "
                f"num_rows={str(self.total_num_rows).replace('None', 'Unknown')}, "
                f"dataset=<{type(self.dataset).__name__} @ {id(self.dataset)}>"
                ")"
            )
        else:
            return super().__repr__()  # pragma: no cover


class OutputDatasetColumnMixin:
    def __iter__(self):
        column_name = self.column_names[0]  # type:ignore[attr-defined]
        for row in super().__iter__():  # type:ignore[misc]
            yield row[column_name]

    def __getitem__(self, key: int | slice | str | Iterable[int]) -> Any:
        column_name = self.column_names[0]  # type:ignore[attr-defined]
        if isinstance(key, str):
            if isinstance(self, OutputIterableDatasetColumn):
                if key != column_name:
                    raise KeyError(
                        f"Column '{key}' is not valid. This OutputIterableDatasetColumn"
                        f" object only has a single column named '{column_name}'."
                    )
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

    def __repr__(self) -> str:
        if isinstance(self, OutputDatasetColumn):
            return (
                f"{type(self).__name__}("
                f"column_name={repr(self.column_names[0])}, "
                f"num_rows={len(self)}, "
                f"dataset=<{type(self.dataset).__name__} @ {id(self.dataset)}>"
                ")"
            )
        elif isinstance(self, OutputIterableDatasetColumn):
            return (
                f"{type(self).__name__}("
                f"column_name={repr(self.column_names[0])}, "
                f"num_rows={str(self.total_num_rows).replace('None', 'Unknown')}, "
                f"dataset=<{type(self.dataset).__name__} @ {id(self.dataset)}>"
                ")"
            )
        else:
            return super().__repr__()  # pragma: no cover


class OutputIterableDataset(OutputDatasetMixin):
    def __init__(
        self,
        step: "Step",
        dataset: IterableDataset,
        pickled: bool = False,
        total_num_rows: None | int = None,
    ):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError(f"Expected Step, got {type(step)}.")
        if not isinstance(dataset, IterableDataset):
            raise ValueError(f"Expected IterableDataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: IterableDataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False
        self.total_num_rows: None | int = total_num_rows
        for f in self._features.values():
            if isinstance(f, Value) and f.dtype == "binary":
                self._pickled_inferred = True
                break

    @property
    def dataset(self) -> IterableDataset:
        """The underlying Hugging Face :py:class:`~datasets.IterableDataset`."""
        return self._dataset

    @property
    def num_rows(self) -> None | int:
        """The number of rows in the dataset."""
        return self.total_num_rows


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
        """The underlying Hugging Face :py:class:`~datasets.Dataset`."""
        return self._dataset

    def save_to_disk(
        self, path: str, num_proc: None | int, num_shards: None | int
    ) -> None:
        self._dataset.save_to_disk(
            path,
            num_proc=min(num_proc if num_proc is not None else 1, len(self._dataset)),
            num_shards=num_shards,
        )
        self._dataset = Dataset.load_from_disk(path)

    @property
    def num_rows(self) -> int:
        """The number of rows in the dataset."""
        return len(self)

    def __len__(self):
        return len(self.dataset)


class OutputIterableDatasetColumn(OutputDatasetColumnMixin, OutputIterableDataset):
    def __init__(
        self,
        step: "Step",
        dataset: IterableDataset,
        pickled: bool = False,
        total_num_rows: None | int = None,
    ):
        from ..steps import Step

        if not isinstance(step, Step):
            raise ValueError(f"Expected Step, got {type(step)}.")
        if not isinstance(dataset, IterableDataset):
            raise ValueError(f"Expected IterableDataset, got {type(dataset)}.")
        self._step: "Step" = step
        self._dataset: IterableDataset = dataset
        self._pickled: bool = pickled
        self._pickled_inferred: bool = False
        self.total_num_rows: None | int = total_num_rows
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


class _SizedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: IterableDataset, total_num_rows: int):
        self.dataset = dataset
        self.total_num_rows = total_num_rows

    @property
    def features(self):  # pragma: no cover
        return self.dataset.features  # type:ignore[attr-defined]

    def cast_column(
        self, *args, **kwargs
    ) -> "_SizedIterableDataset":  # pragma: no cover
        return _SizedIterableDataset(
            dataset=self.dataset.cast_column(*args, **kwargs),
            total_num_rows=self.total_num_rows,
        )

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self) -> int:
        return self.total_num_rows


def get_sized_dataset(
    dataset: Dataset | IterableDataset, total_num_rows: None | int
) -> Dataset | _SizedIterableDataset | IterableDataset:
    if isinstance(dataset, IterableDataset) and total_num_rows is not None:
        return _SizedIterableDataset(dataset=dataset, total_num_rows=total_num_rows)
    else:
        return dataset
