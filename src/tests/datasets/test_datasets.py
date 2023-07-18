import pytest

from datasets import Dataset, IterableDataset

from ...datasets import (
    OutputDataset,
    OutputDatasetColumn,
    OutputIterableDataset,
    OutputIterableDatasetColumn,
)
from ...steps import Step


class TestErrors:
    @pytest.mark.parametrize(
        "cls,iterable",
        [
            (OutputDataset, False),
            (OutputDatasetColumn, False),
            (OutputIterableDataset, True),
            (OutputIterableDatasetColumn, True),
        ],
    )
    def test_invalid_args_step(self, cls, iterable):
        dataset = Dataset.from_dict({"out1": ["a", "b", "c"]})
        if iterable:
            dataset = dataset.to_iterable_dataset()  # type:ignore[assignment]
        with pytest.raises(ValueError):
            cls(None, dataset, pickled=False)

    @pytest.mark.parametrize(
        "cls,iterable",
        [
            (OutputDataset, False),
            (OutputDatasetColumn, False),
            (OutputIterableDataset, True),
            (OutputIterableDatasetColumn, True),
        ],
    )
    def test_invalid_args_dataset(self, cls, iterable):
        step = Step("my-step", None, "out1")
        dataset = Dataset.from_dict({"out1": ["a", "b", "c"]})
        if not iterable:
            dataset = dataset.to_iterable_dataset()  # type:ignore[assignment]
        with pytest.raises(ValueError):
            cls(step, dataset, pickled=False)

    @pytest.mark.parametrize(
        "cls,iterable",
        [
            (OutputDatasetColumn, False),
            (OutputIterableDatasetColumn, True),
        ],
    )
    def test_too_few_columns(self, cls, iterable):
        step = Step("my-step", None, "out1")
        dataset = Dataset.from_dict({})
        if iterable:

            def empty_generator():
                return iter(())

            dataset = IterableDataset.from_generator(
                empty_generator
            )  # type:ignore[assignment]
        with pytest.raises(ValueError):
            cls(step, dataset, pickled=False)

    @pytest.mark.parametrize(
        "cls,iterable",
        [
            (OutputDatasetColumn, False),
            (OutputIterableDatasetColumn, True),
        ],
    )
    def test_too_many_columns(self, cls, iterable):
        step = Step("my-step", None, "out1")
        dataset = Dataset.from_dict({"out1": ["a", "b", "c"], "out2": [1, 2, 3]})
        if iterable:
            dataset = dataset.to_iterable_dataset()  # type:ignore[assignment]
        with pytest.raises(ValueError):
            cls(step, dataset, pickled=False)


class TestFunctionality:
    def test_dataset(self):
        step = Step("my-step", None, ["out1"])
        dataset = Dataset.from_dict({"out1": ["a", "b", "c"]})
        output = OutputDataset(step, dataset, pickled=False)
        assert len(output) == 3
        rows = list(output)
        assert [row["out1"] for row in rows] == ["a", "b", "c"]

    def test_dataset_pickled(self):
        step = Step("my-step", None, ["out1"])
        dataset = Dataset.from_dict(
            {
                "out1": [
                    step.pickle(set(["a"])),
                    step.pickle(set(["b"])),
                    step.pickle(set(["c"])),
                ]
            }
        )
        output = OutputDataset(step, dataset, pickled=True)
        assert len(output) == 3
        rows = list(output)
        assert [row["out1"] for row in rows] == [set(["a"]), set(["b"]), set(["c"])]

    def test_iterable_dataset(self):
        step = Step("my-step", None, ["out1"])
        dataset = Dataset.from_dict({"out1": ["a", "b", "c"]}).to_iterable_dataset()
        output = OutputIterableDataset(step, dataset, pickled=False)
        rows = list(output)
        assert [row["out1"] for row in rows] == ["a", "b", "c"]

    def test_iterable_dataset_pickled(self):
        step = Step("my-step", None, ["out1"])
        dataset = Dataset.from_dict(
            {
                "out1": [
                    step.pickle(set(["a"])),
                    step.pickle(set(["b"])),
                    step.pickle(set(["c"])),
                ]
            }
        ).to_iterable_dataset()
        output = OutputIterableDataset(step, dataset, pickled=True)
        rows = list(output)
        assert [row["out1"] for row in rows] == [set(["a"]), set(["b"]), set(["c"])]

    def test_dataset_indexing(self):
        step = Step("my-step", None, ["out1"])
        dataset = Dataset.from_dict(
            {
                "out1": ["a", "b", "c"],
                "out2": [
                    step.pickle(set(["a"])),
                    step.pickle(set(["b"])),
                    step.pickle(set(["c"])),
                ],
            }
        )
        output = OutputDataset(step, dataset, pickled=True)
        assert output[1] == {"out1": "b", "out2": set(["b"])}
        assert output[0:2] == {"out1": ["a", "b"], "out2": [set(["a"]), set(["b"])]}
        assert output[[0, 2]] == {"out1": ["a", "c"], "out2": [set(["a"]), set(["c"])]}
        assert isinstance(output["out1"], OutputDatasetColumn)
        assert output["out1"]._pickled is False
        assert len(output["out1"]) == 3
        assert list(output["out1"]) == ["a", "b", "c"]
        assert output["out1"][1] == "b"
        assert output["out1"][0:2] == ["a", "b"]
        assert output["out1"][[0, 2]] == ["a", "c"]
        assert output["out1"]["out1"] == ["a", "b", "c"]
        assert isinstance(output["out2"], OutputDatasetColumn)
        assert output["out2"]._pickled
        assert len(output["out2"]) == 3
        assert list(output["out2"]) == [set(["a"]), set(["b"]), set(["c"])]
        assert output["out2"][1] == set(["b"])
        assert output["out2"][0:2] == [set(["a"]), set(["b"])]
        assert output["out2"][[0, 2]] == [set(["a"]), set(["c"])]
        assert output["out2"]["out2"] == [set(["a"]), set(["b"]), set(["c"])]

    def test_iterable_dataset_indexing(self):
        step = Step("my-step", None, ["out1"])
        dataset = Dataset.from_dict(
            {
                "out1": ["a", "b", "c"],
                "out2": [
                    step.pickle(set(["a"])),
                    step.pickle(set(["b"])),
                    step.pickle(set(["c"])),
                ],
            }
        ).to_iterable_dataset()
        output = OutputIterableDataset(step, dataset, pickled=False)
        with pytest.raises(NotImplementedError):
            assert output[1] == {"out1": "b", "out2": set(["b"])}
        with pytest.raises(NotImplementedError):
            assert output[0:2] == {"out1": ["a", "b"], "out2": [set(["a"]), set(["b"])]}
        with pytest.raises(NotImplementedError):
            assert output[[0, 2]] == {
                "out1": ["a", "c"],
                "out2": [set(["a"]), set(["c"])],
            }
        assert isinstance(output["out1"], OutputIterableDatasetColumn)
        assert output["out1"]._pickled is False
        assert output["out1"]._pickled_inferred is False
        with pytest.raises(TypeError):
            assert len(output["out1"]) == 3  # type:ignore[arg-type]
        assert list(output["out1"]) == ["a", "b", "c"]
        with pytest.raises(NotImplementedError):
            assert output["out1"][1] == "b"
        with pytest.raises(NotImplementedError):
            assert output["out1"][0:2] == ["a", "b"]
        with pytest.raises(NotImplementedError):
            assert output["out1"][[0, 2]] == ["a", "c"]
        assert list(output["out1"]["out1"]) == ["a", "b", "c"]
        assert isinstance(output["out2"], OutputIterableDatasetColumn)
        assert output["out2"]._pickled is False
        assert output["out2"]._pickled_inferred
        with pytest.raises(TypeError):
            assert len(output["out2"]) == 3  # type:ignore[arg-type]
        assert list(output["out2"]) == [set(["a"]), set(["b"]), set(["c"])]
        with pytest.raises(NotImplementedError):
            assert output["out2"][1] == set(["b"])
        with pytest.raises(NotImplementedError):
            assert output["out2"][0:2] == [set(["a"]), set(["b"])]
        with pytest.raises(NotImplementedError):
            assert output["out2"][[0, 2]] == [set(["a"]), set(["c"])]
        assert list(output["out2"]["out2"]) == [set(["a"]), set(["b"]), set(["c"])]
