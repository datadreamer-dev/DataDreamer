from functools import partial
from itertools import chain
from typing import Any, Iterator, cast

from datasets import Dataset, IterableDataset
from datasets.features.features import Features


def get_column_names(dataset: Dataset | IterableDataset) -> list[str]:
    column_names = cast(None | list[str], dataset.column_names)
    if column_names:
        return column_names
    else:
        try:
            first_row = next(iter(dataset))
        except StopIteration:
            return []
        return list(first_row.keys())


def dataset_zip(*datasets: Dataset) -> Dataset:
    if len(datasets) == 0:
        raise ValueError("You must provide at least one dataset to zip.")
    dataset_dicts: list[dict[str, list[Any]]] = [
        {n: list(d[n]) for n in get_column_names(d)} for d in datasets
    ]
    merged_dataset: dict[str, list[Any]] = {}
    for d in dataset_dicts:
        for k, v in d.items():
            merged_dataset[k] = v
    return Dataset.from_dict(merged_dataset)


def iterable_dataset_zip(*datasets: Dataset | IterableDataset) -> IterableDataset:
    if len(datasets) == 0:
        raise ValueError("You must provide at least one dataset to zip.")

    def merged_generator(datasets):
        iters: list[Iterator[dict[str, Any]]] = [iter(d) for d in datasets]
        for row_dicts in zip(*iters):
            row = {}
            for d in row_dicts:
                for k, v in d.items():
                    row[k] = v
            yield row

    column_names: list[str] = list(
        chain.from_iterable([get_column_names(d) for d in datasets])
    )
    features = Features([(n, None) for n in column_names])
    return IterableDataset.from_generator(
        partial(merged_generator, datasets), features=features
    )
