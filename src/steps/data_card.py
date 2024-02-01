from typing import Any

from ..utils.collection_utils import sort_keys


class DataCardType:
    """The types of data card entries."""

    DATETIME = "Date & Time"
    MODEL_NAME = "Model Name"
    DATASET_NAME = "Dataset Name"
    LICENSE = "License Information"
    CITATION = "Citation Information"
    DATASET_CARD = "Dataset Card"
    MODEL_CARD = "Model Card"
    URL = "URL"


def sort_data_card(data_card: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return sort_keys(
        data_card,
        key_order=[
            DataCardType.DATETIME,
            DataCardType.DATASET_NAME,
            DataCardType.MODEL_NAME,
            DataCardType.URL,
            DataCardType.DATASET_CARD,
            DataCardType.MODEL_CARD,
            DataCardType.LICENSE,
            DataCardType.CITATION,
        ],
    )
