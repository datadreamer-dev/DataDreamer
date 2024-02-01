from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Any, Iterator, cast

from datasets import Dataset, IterableDataset
from datasets.features.features import Features

from .. import DataDreamer


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


def drop_unsupported_features(dataset: Dataset | IterableDataset):
    if isinstance(dataset, Dataset):
        dataset.reset_format()
        for index_name in dataset.list_indexes():
            dataset.drop_index(index_name)  # pragma: no cover


def dataset_zip(
    *datasets: Dataset,
    writer_batch_size: None | int = 1000,
    num_proc: None | int = None,
) -> Dataset:
    if len(datasets) == 0:
        raise ValueError("You must provide at least one dataset to zip.")
    datasets = tuple([deepcopy(d) for d in datasets])
    for d in datasets:
        drop_unsupported_features(d)
    smallest_dataset = min(datasets, key=lambda d: len(d))

    def merge_rows(datasets, x, idx):
        result_row = {}
        for d in datasets:
            result_row.update(d[idx])
        return result_row

    DataDreamer._enable_hf_datasets_logging()
    zip_results = smallest_dataset.map(
        partial(merge_rows, datasets),
        with_indices=True,
        desc="Zipping datasets together",
        writer_batch_size=writer_batch_size,
        num_proc=num_proc,
    )
    DataDreamer._disable_hf_datasets_logging()
    return zip_results


def iterable_dataset_zip(*datasets: Dataset | IterableDataset) -> IterableDataset:
    if len(datasets) == 0:
        raise ValueError("You must provide at least one dataset to zip.")
    datasets = tuple([deepcopy(d) for d in datasets])
    for d in datasets:
        drop_unsupported_features(d)

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
