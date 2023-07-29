import os
from collections import OrderedDict
from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset, DatasetDict

from ..datasets import OutputDataset, OutputIterableDataset
from ..pickling import unpickle_transform
from .step_background import wait

if TYPE_CHECKING:  # pragma: no cover
    from .step import Step


def _step_to_dataset_dict(
    step: "Step",
    train_size: None | float | int = None,
    validation_size: None | float | int = None,
    test_size: None | float | int = None,
    stratify_by_column: None | str = None,
    writer_batch_size: None | int = 1000,
    save_num_proc: None | int = None,
    save_num_shards: None | int = None,
) -> tuple[OutputDataset, DatasetDict]:
    # Wait for a lazy step to complete
    wait(step)
    if isinstance(step.output, OutputIterableDataset):
        step = step.save(
            writer_batch_size=writer_batch_size,
            save_num_proc=save_num_proc,
            save_num_shards=save_num_shards,
        )
    dataset: Dataset = cast(Dataset, step.output.dataset)
    splits: OrderedDict[str, Any] = OrderedDict(
        [
            (
                split_name,
                Decimal(str(proportion))
                if isinstance(proportion, float)
                else proportion,
            )
            for split_name, proportion in [
                ("train", train_size),
                ("validation", validation_size),
                ("test", test_size),
            ]
            if proportion is not None and proportion != 0
        ]
    )
    if len(splits) == 0:
        splits = OrderedDict({"train": Decimal("1.0")})
    split_names = list(splits.keys())
    proportions = list(splits.values())
    total = sum(proportions)
    if total != Decimal("1.0") and total != len(dataset):
        raise ValueError(
            "If not None, train_size, validation_size, test_size must sum up to 1.0 or"
            f" the number of rows ({len(dataset)}) in the dataset. Instead, got a total"
            f" of : {total}."
        )

    # Split the dataset
    total_proportion = Decimal("1.0")
    remainder_dataset = dataset
    while len(proportions) > 1:
        split_name = split_names.pop()
        proportion = proportions.pop()
        new_dataset_dict = remainder_dataset.train_test_split(
            test_size=float(proportion / total_proportion)
            if total == Decimal("1.0")
            else int(proportion),
            shuffle=False,
            stratify_by_column=stratify_by_column,
            writer_batch_size=writer_batch_size,
        )
        splits[split_name] = new_dataset_dict["test"]
        remainder_dataset = new_dataset_dict["train"]
        if total == Decimal("1.0"):
            total_proportion = total_proportion - proportion

    # Finally, assign the remainder of the splits
    split_name = split_names.pop()
    splits[split_name] = remainder_dataset

    return cast(OutputDataset, step.output), DatasetDict(splits)


def _path_to_split_paths(path: str, dataset_dict: DatasetDict) -> dict[str, str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base, extension = os.path.splitext(path)
    paths: dict[str, str] = {}
    for split_name in dataset_dict:
        if split_name == "validation":
            path_split_name = "val"
        else:
            path_split_name = split_name
        split_path = f"{base}.{path_split_name}{extension}"
        paths[split_name] = split_path
    return paths


def _unpickle_export(export: DatasetDict | list | dict, output_dataset: OutputDataset):
    if output_dataset._pickled:
        if isinstance(export, DatasetDict):
            export.set_transform(
                partial(
                    unpickle_transform,
                    features=output_dataset._features,
                    batched=True,
                )
            )
            return export
        elif isinstance(export, list):
            return [
                unpickle_transform(
                    row, features=output_dataset._features, batched=False
                )
                for row in export
            ]
        else:
            return unpickle_transform(
                export, features=output_dataset._features, batched=True
            )
    else:
        return export


__all__ = [
    "_step_to_dataset_dict",
    "_path_to_split_paths",
    "_unpickle_export",
]
