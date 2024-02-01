import pytest
from datasets import Dataset, IterableDataset
from datasets.features.features import Features

from ...datasets.utils import dataset_zip, get_column_names, iterable_dataset_zip


class TestErrors:
    def test_dataset_zip_no_args(self):
        with pytest.raises(ValueError):
            dataset_zip()

    def test_iterable_dataset_zip_no_args(self):
        with pytest.raises(ValueError):
            iterable_dataset_zip()


class TestFunctionality:
    def test_get_column_names(self):
        dataset_dict = {"foo": [1, 2, 3], "bar": ["a", "b", "c"]}

        def dataset_dict_generator():
            yield {"foo": 1, "bar": "a"}
            yield {"foo": 2, "bar": "b"}
            yield {"foo": 3, "bar": "c"}

        def empty_generator():
            return iter(())

        dataset = Dataset.from_dict(dataset_dict)
        iterable_dataset = IterableDataset.from_generator(dataset_dict_generator)
        iterable_dataset_empty = IterableDataset.from_generator(empty_generator)
        features = Features([("foo", None), ("bar", None)])
        iterable_dataset_features = IterableDataset.from_generator(
            empty_generator, features=features
        )
        assert set(get_column_names(dataset)) == set(["foo", "bar"])
        assert set(get_column_names(iterable_dataset)) == set(["foo", "bar"])
        assert set(get_column_names(iterable_dataset_empty)) == set([])
        assert set(get_column_names(iterable_dataset_features)) == set(["foo", "bar"])

    def test_dataset_zip(self):
        dataset_dict_1 = {"foo": [1, 2, 3], "bar": ["a", "b", "c"]}
        dataset_dict_2 = {"zoo": [1, 2, 3], "car": ["a", "b", "c"]}
        dataset_dict_3 = {"moo": [1, 2, 3], "far": ["a", "b", "c"]}
        dataset_1 = Dataset.from_dict(dataset_dict_1)
        dataset_2 = Dataset.from_dict(dataset_dict_2)
        dataset_3 = Dataset.from_dict(dataset_dict_3)
        zipped_dataset = dataset_zip(dataset_1, dataset_2, dataset_3)
        assert set(zipped_dataset.column_names) == set(  # type: ignore[arg-type]
            ["foo", "bar", "zoo", "car", "moo", "far"]
        )
        assert len(zipped_dataset) == 3
        assert zipped_dataset[0] == {
            "foo": 1,
            "bar": "a",
            "zoo": 1,
            "car": "a",
            "moo": 1,
            "far": "a",
        }

    def test_iterable_dataset_zip(self):
        dataset_dict_1 = {"foo": [1, 2, 3], "bar": ["a", "b", "c"]}

        def dataset_dict_2_generator():
            yield {"zoo": 1, "car": "a"}
            yield {"zoo": 2, "car": "b"}
            yield {"zoo": 3, "car": "c"}

        dataset_dict_3 = {"moo": [1, 2, 3], "far": ["a", "b", "c"]}
        dataset_1 = Dataset.from_dict(dataset_dict_1)
        iterable_dataset_2 = IterableDataset.from_generator(dataset_dict_2_generator)
        dataset_3 = Dataset.from_dict(dataset_dict_3)
        zipped_dataset = iterable_dataset_zip(dataset_1, iterable_dataset_2, dataset_3)
        assert set(zipped_dataset.column_names) == set(  # type: ignore[arg-type]
            ["foo", "bar", "zoo", "car", "moo", "far"]
        )
        assert isinstance(zipped_dataset, IterableDataset)
        assert next(iter(zipped_dataset)) == {
            "foo": 1,
            "bar": "a",
            "zoo": 1,
            "car": "a",
            "moo": 1,
            "far": "a",
        }
