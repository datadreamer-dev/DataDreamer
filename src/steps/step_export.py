import os
from functools import partial

from datasets import DatasetDict

from ..datasets import OutputDataset
from ..pickling import unpickle_transform


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
                    unpickle_transform, features=output_dataset._features, batched=True
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


__all__ = ["_path_to_split_paths", "_unpickle_export"]
