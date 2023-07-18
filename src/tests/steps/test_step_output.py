from datetime import datetime

import pytest

from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset

from ...datasets import OutputDataset, OutputIterableDataset
from ...errors import StepOutputError, StepOutputTypeError
from ...pickling.pickle import _pickle
from ...steps import LazyRowBatches, LazyRows, Step


class TestErrors:
    def test_access_output_before_step_is_run(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputError):
            step.output

    def test_access_output_while_step_is_running(self):
        step = Step("my-step", None, "out1")
        step.progress = 0.565
        assert step.progress == 0.565
        with pytest.raises(StepOutputError):
            step.output

    def test_output_invalid_type(self):
        step_single = Step("my-step", None, "out1")
        with pytest.raises(StepOutputError):
            step_single._set_output({"out2": 5})
        with pytest.raises(StepOutputError):
            step_single._set_output({"out2": "a"})

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = HFIterableDataset.from_generator(dataset_generator)
        with pytest.raises(StepOutputError):
            step_single._set_output(iterable_dataset)  # type: ignore[arg-type]

        step_multiple = Step("my-step", None, ["out1", "out2"])
        with pytest.raises(StepOutputError):
            step_multiple._set_output(5)  # type: ignore[arg-type]
        with pytest.raises(StepOutputError):
            step_multiple._set_output(
                LazyRows([iterable_dataset, [1, 2, 3]], total_num_rows=3)
            )

    def test_output_twice(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": ["a", "b", "c"]})
        with pytest.raises(StepOutputError):
            step._set_output({"out1": ["a", "b", "c"]})

    def test_output_dict_with_wrong_keys(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputError):
            step._set_output({"out2": ["a", "b", "c"]})

    def test_output_list_with_wrong_number_of_outputs(self):
        step = Step("my-step", None, ["out1", "out2", "out3", "out4"])
        with pytest.raises(StepOutputError):
            step._set_output([("a", 1), ("b", 2), ("c", 3)])

    def test_output_list_of_dicts_with_wrong_keys(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputError):
            step._set_output([{"out1": "a"}, {"out2": "b"}, {"out2": "c"}])

    def test_output_tuple_with_wrong_number_of_outputs(self):
        step = Step("my-step", None, ["out1"])
        with pytest.raises(StepOutputError):
            step._set_output((["a", "b", "c"], [1, 2, 3]))

    def test_output_dataset_with_wrong_number_of_columns(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None
        dataset_dict = {"out1": ["a", "b", "c"], "out2": [1, 2, 3]}
        dataset = HFDataset.from_dict(dataset_dict)
        with pytest.raises(StepOutputError):
            step._set_output(dataset)

        def dataset_generator():
            yield {"out1": "a", "out2": 1}
            yield {"out1": "b", "out2": 2}
            yield {"out1": "c", "out2": 3}

        iterable_dataset = HFIterableDataset.from_generator(dataset_generator)
        with pytest.raises(StepOutputError):
            step._set_output(LazyRows(iterable_dataset, total_num_rows=3))

    def test_output_generator_function_of_dict_with_wrong_keys(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out2": "a"}
            yield {"out2": "b"}
            yield {"out2": "c"}

        with pytest.raises(StepOutputError):
            step._set_output(LazyRows(dataset_generator, total_num_rows=3))

    def test_output_generator_function_of_list_of_dict_batched_with_wrong_keys(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [{"out1": "a"}, {"out2": "b"}]
            yield [{"out2": "c"}]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        with pytest.raises(StepOutputError):
            list(step.output)

    def test_output_generator_function_of_list_with_wrong_number_of_outputs(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ["a"]
            yield ["b"]
            yield ["c"]

        with pytest.raises(StepOutputError):
            step._set_output(LazyRows(dataset_generator, total_num_rows=3))

    def test_output_generator_function_of_tuple_with_wrong_number_of_outputs(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ("a",)
            yield ("b",)
            yield ("c",)

        with pytest.raises(StepOutputError):
            step._set_output(LazyRows(dataset_generator, total_num_rows=3))

    def test_output_generator_function_of_tuple_batched_column_with_wrong_number_of_outputs(
        self,
    ):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield (["a", "b"],)
            yield (["c"],)

        with pytest.raises(StepOutputError):
            step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))

    def test_total_num_rows_warnings(self):
        def dataset_generator():
            yield ("a",)
            yield ("b",)
            yield ("c",)

        with pytest.warns(UserWarning):
            LazyRows(dataset_generator)

        def dataset_batch_generator():
            yield (["a", "b"],)
            yield (["c"],)

        with pytest.warns(UserWarning):
            LazyRowBatches(dataset_batch_generator)

    def test_pickle_warning(self):
        with pytest.warns(UserWarning):
            _pickle(5)


class TestProgress:
    def test_before_output(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None
        assert step._Step__get_progress_string() == "0%"  # type: ignore[attr-defined]

    def test_set_progress(self):
        step = Step("my-step", None, "out1")
        step.progress = 0.565
        assert step.progress == 0.565

    def test_get_progress_string(self):
        step = Step("my-step", None, "out1")
        step.progress = 0.565
        assert step._Step__get_progress_string() == "56%"  # type: ignore[attr-defined]

    def test_progress_range_lower(self):
        step = Step("my-step", None, "out1")
        step.progress = -0.5
        assert step.progress == 0.0

    def test_progress_range_higher(self):
        step = Step("my-step", None, "out1")
        step.progress = 1.5
        assert step.progress == 1.0

    def test_progress_monotonic_increases(self):
        step = Step("my-step", None, "out1")
        step.progress = 0.1
        assert step.progress == 0.1
        step.progress = 0.2
        assert step.progress == 0.2
        step.progress = 0.1
        assert step.progress == 0.2

    def test_progress_after_output_single(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None
        step._set_output("a")  # type: ignore[arg-type]
        assert step.progress == 1.0

    def test_progress_after_output_list(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None
        step._set_output(["a", "b", "c"])
        assert step.progress == 1.0

    def test_progress_after_output_tuple_of_list(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None
        step._set_output((["a", "b", "c"],))
        assert step.progress == 1.0

    def test_progress_after_output_dataset(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = HFDataset.from_dict(dataset_dict)
        step._set_output(dataset)
        assert step.progress == 1.0

    def test_progress_after_output_iterable_dataset(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = HFIterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows(iterable_dataset, total_num_rows=3))
        assert step.progress == 0.0

    def test_progress_is_1_after__output_is_dataset(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = HFDataset.from_dict(dataset_dict)
        step._Step__output = dataset  # type: ignore[attr-defined]
        assert step.progress is None
        step.progress = 0.5
        assert step.progress == 1.0

    def test_auto_progress(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        next(iter(step.output))
        assert step.progress == 1.0 / 3.0

    def test_auto_progress_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [{"out1": "a"}, {"out1": "b"}, {"out1": "c"}]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        next(iter(step.output))
        assert step.progress == 1.0 / 3.0


class TestEmptyOutput:
    def test_output_single_empty(self):
        step = Step("my-step", None, "out1")
        step._set_output(None)
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 0
        assert set(step.output.column_names) == set(["out1"])

    def test_output_list_empty(self):
        step = Step("my-step", None, "out1")
        step._set_output([])
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 0
        assert set(step.output.column_names) == set(["out1"])

    def test_output_iterator_empty(self):
        step = Step("my-step", None, "out1")
        step._set_output(map(lambda x: x, []))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 0
        assert set(step.output.column_names) == set(["out1"])

    def test_output_tuple_of_list_empty(self):
        step = Step("my-step", None, "out1")
        step._set_output(([],))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 0
        assert set(step.output.column_names) == set(["out1"])

    def test_output_dict_of_list_empty(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": []})
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 0
        assert set(step.output.column_names) == set(["out1"])

    def test_output_generator_function_empty(self):
        step = Step("my-step", None, "out1")

        def empty_generator():
            return iter(())

        step._set_output(LazyRows(empty_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert len(list(step.output)) == 0
        assert set(step.output.column_names) == set(["out1"])

    def test_output_generator_function_of_tuple_batched_column_empty(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ([], [])
            yield ([], [])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert len(list(step.output)) == 0
        assert set(step.output.column_names) == set(["out1", "out2"])

    def test_output_generator_function_of_dict_batched_empty(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": []}
            yield {"out1": []}

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert len(list(step.output)) == 0
        assert set(step.output.column_names) == set(["out1"])

    def test_output_generator_function_of_tuple_batched_row_empty(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield []
            yield []

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert len(list(step.output)) == 0
        assert set(step.output.column_names) == set(["out1", "out2"])


class TestSingleOutput:
    def test_output_single(self):
        step = Step("my-step", None, "out1")
        step._set_output("a")  # type: ignore[arg-type]
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_single_list(self):
        step = Step("my-step", None, "out1")
        step._set_output(["a"])
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_single_tuple(self):
        step = Step("my-step", None, "out1")
        step._set_output(("a",))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_single_dict(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": "a"})
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_list(self):
        step = Step("my-step", None, "out1")
        step._set_output(["a", "b", "c"])
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_list_of_dicts_matching(self):
        step = Step("my-step", None, "out1")
        step._set_output([{"out1": "a"}, {"out1": "b"}, {"out1": "c"}])
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_list_of_dicts(self):
        step = Step("my-step", None, "out1")
        step._set_output([{"foo": "a"}, {"foo": "b"}, {"foo": "c"}])
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == {"foo": "a"}
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == {"foo": "a"}

    def test_output_list_of_tuple(self):
        step = Step("my-step", None, "out1")
        step._set_output([("a",), ("b",), ("c",)])
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_list_of_list_row(self):
        step = Step("my-step", None, "out1")
        step._set_output([["a"], ["b"], ["c"]])
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == ["a"]
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == ["a"]

    def test_output_list_of_list_column(self):
        step = Step("my-step", None, "out1")
        step._set_output([["a", "b", "c"]])
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == ["a", "b", "c"]
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == ["a", "b", "c"]

    def test_output_iterator(self):
        step = Step("my-step", None, "out1")
        step._set_output(map(lambda x: x, ["a", "b", "c"]))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_tuple_of_list(self):
        step = Step("my-step", None, "out1")
        step._set_output((["a", "b", "c"],))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_tuple_of_iterator(self):
        step = Step("my-step", None, "out1")
        step._set_output((iter(range(3)),))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == 0

    def test_output_dict_of_list(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": ["a", "b", "c"]})
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_dict_of_iterator(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": iter(range(3))})
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == 0

    def test_output_dataset(self):
        step = Step("my-step", None, "out1")
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = HFDataset.from_dict(dataset_dict)
        step._set_output(dataset)
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_iterable_dataset(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = HFIterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows(iterable_dataset, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_iterable_dataset_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = HFIterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRowBatches(iterable_dataset, total_num_rows=3))  # type: ignore[arg-type]
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_list_of_datasets(self):
        step = Step("my-step", None, "out1")
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = HFDataset.from_dict(dataset_dict)
        step._set_output([dataset])
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_tuple_of_datasets(self):
        step = Step("my-step", None, "out1")
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = HFDataset.from_dict(dataset_dict)
        step._set_output((dataset,))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_generator_function_of_dict(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_dict_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": ["a", "b"]}
            yield {"out1": ["c"]}

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_dict_batched_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": iter(["a", "b"])}
            yield {"out1": iter(["c"])}

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_list_of_dict_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [{"out1": "a"}, {"out1": "b"}]
            yield [{"out1": "c"}]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_list_of_dict_batched_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter([{"out1": "a"}, {"out1": "b"}])
            yield iter([{"out1": "c"}])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ("a",)
            yield ("b",)
            yield ("c",)

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple_batched_row(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [("a",), ("b",)]
            yield [("c",)]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple_batched_column(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield (["a", "b"],)
            yield (["c"],)

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple_batched_row_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter([("a",), ("b",)])
            yield iter([("c",)])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple_batched_column_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield (iter(["a", "b"]),)
            yield (iter(["c"]),)

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_list_row(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ["a"]
            yield ["b"]
            yield ["c"]

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == ["a"]

    def test_output_generator_function_of_list_row_batched_row(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [["a"], ["b"]]
            yield [["c"]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == ["a"]

    def test_output_generator_function_of_list_row_batched_column(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [["a", "b"]]
            yield [["c"]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_list_row_batched_row_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter([["a"], ["b"]])
            yield iter([["c"]])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == ["a"]

    def test_output_generator_function_of_list_row_batched_column_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [iter(["a", "b"])]
            yield [iter(["c"])]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_dict_of_generator_function(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows({"out1": dataset_generator}, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_dict_of_generator_function_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ["a", "b"]
            yield ["c"]

        step._set_output(LazyRowBatches({"out1": dataset_generator}, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_dict_of_generator_function_batched_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter(["a", "b"])
            yield iter(["c"])

        step._set_output(LazyRowBatches({"out1": dataset_generator}, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_list_of_generator_function(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows([dataset_generator], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_list_of_generator_function_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ["a", "b"]
            yield ["c"]

        step._set_output(LazyRowBatches([dataset_generator], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_list_of_generator_function_batched_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter(["a", "b"])
            yield iter(["c"])

        step._set_output(LazyRowBatches([dataset_generator], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_tuple_of_generator_function(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows((dataset_generator,), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_tuple_of_generator_function_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ["a", "b"]
            yield ["c"]

        step._set_output(LazyRowBatches((dataset_generator,), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_tuple_of_generator_function_batched_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter(["a", "b"])
            yield iter(["c"])

        step._set_output(LazyRowBatches((dataset_generator,), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"


class TestMultipleOutput:
    def test_output_single_list(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output(["a", 1])
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"
        assert step.output[0]["out2"] == 1

    def test_output_single_tuple(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output(("a", 1))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"
        assert step.output[0]["out2"] == 1

    def test_output_single_dict(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output({"out1": "a", "out2": 1})
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"
        assert step.output[0]["out2"] == 1

    def test_output_list_of_tuple(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output([("a", 1), ("b", 2), ("c", 3)])
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_list_of_list_row(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output([["a", 1], ["b", 2], ["c", 3]])
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_list_of_list_column(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output([["a", "b", "c"], [1, 2, 3]])
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_tuple_of_list(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output((["a", "b", "c"], [1, 2, 3]))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_tuple_of_iterator(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output((iter(range(3)), ["a", "b", "c"]))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert list(step.output["out1"]) == [0, 1, 2]
        assert list(step.output["out2"]) == ["a", "b", "c"]

    def test_output_dict_of_list(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output({"out1": ["a", "b", "c"], "out2": [1, 2, 3]})
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_dict_of_iterator(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output({"out1": iter(range(3)), "out2": ["a", "b", "c"]})
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert list(step.output["out1"]) == [0, 1, 2]
        assert list(step.output["out2"]) == ["a", "b", "c"]

    def test_output_dataset(self):
        step = Step("my-step", None, ["out1", "out2"])
        dataset_dict = {"out1": ["a", "b", "c"], "out2": [1, 2, 3]}
        dataset = HFDataset.from_dict(dataset_dict)
        step._set_output(dataset)
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_iterable_dataset(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield {"out1": "a", "out2": 1}
            yield {"out1": "b", "out2": 2}
            yield {"out1": "c", "out2": 3}

        iterable_dataset = HFIterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows(iterable_dataset, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_iterable_dataset_batched(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield {"out1": "a", "out2": 1}
            yield {"out1": "b", "out2": 2}
            yield {"out1": "c", "out2": 3}

        iterable_dataset = HFIterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRowBatches(iterable_dataset, total_num_rows=3))  # type: ignore[arg-type]
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_list_of_datasets(self):
        step = Step("my-step", None, ["out1", "out2"])
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = HFDataset.from_dict(dataset_dict)

        def dataset_generator():
            yield {"out2": 1}
            yield {"out2": 2}
            yield {"out2": 3}

        iterable_dataset = HFIterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows([dataset, iterable_dataset], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_tuple_of_datasets(self):
        step = Step("my-step", None, ["out1", "out2"])
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = HFDataset.from_dict(dataset_dict)

        def dataset_generator():
            yield {"out2": 1}
            yield {"out2": 2}
            yield {"out2": 3}

        iterable_dataset = HFIterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows((dataset, iterable_dataset), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_dict(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield {"out1": "a", "out2": 1}
            yield {"out1": "b", "out2": 2}
            yield {"out1": "c", "out2": 3}

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_dict_batched(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield {"out1": ["a", "b"], "out2": [1, 2]}
            yield {"out1": ["c"], "out2": [3]}

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_tuple(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ("a", 1)
            yield ("b", 2)
            yield ("c", 3)

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_tuple_batched_row(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [("a", 1), ("b", 2)]
            yield [("c", 3)]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_tuple_batched_column(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield (["a", "b"], [1, 2])
            yield (["c"], [3])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_iterator_row(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield iter(["a", 1])
            yield iter(["b", 2])
            yield iter(["c", 3])

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_list_row(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ["a", 1]
            yield ["b", 2]
            yield ["c", 3]

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_list_row_batched_row(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [["a", 1], ["b", 2]]
            yield [["c", 3]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_list_row_batched_column_ambiguous(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [["a", "b"], [1, 2]]
            yield [["c"], [3]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        with pytest.raises(StepOutputTypeError):
            assert len(list(step.output)) == 3

    def test_output_generator_function_of_list_row_batched_column(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [["a", "b", "c"], [1, 2, 3]]
            yield [["d"], [4]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=4))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 4
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c", "d"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3, 4]

    def test_output_generator_function_of_list_row_batched_row_iterator(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield iter([["a", 1], ["b", 2]])
            yield iter([["c", 3]])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_list_row_batched_column_iterator(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [iter(["a", "b"]), [1, 2]]
            yield [iter(["c"]), [3]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_dict_of_generator_function(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(
            LazyRows({"out1": dataset_generator, "out2": [1, 2, 3]}, total_num_rows=3)
        )
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_dict_of_generator_function_batched(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ["a", "b"]
            yield ["c"]

        step._set_output(
            LazyRowBatches(
                {"out1": dataset_generator, "out2": [[1, 2], [3]]},
                total_num_rows=3,
            )
        )
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_list_of_generator_function(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows([dataset_generator, [1, 2, 3]], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_list_of_generator_function_batched(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ["a", "b"]
            yield ["c"]

        step._set_output(
            LazyRowBatches([dataset_generator, [[1, 2], [3]]], total_num_rows=3)
        )
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_tuple_of_generator_function(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows((dataset_generator, [1, 2, 3]), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_tuple_of_generator_function_batched(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ["a", "b"]
            yield ["c"]

        step._set_output(
            LazyRowBatches((dataset_generator, [[1, 2], [3]]), total_num_rows=3)
        )
        assert set(step.output.column_names) == set(["out1", "out2"])
        assert isinstance(step.output, OutputIterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]


class TestTypes:
    def test_features(self):
        from datetime import datetime

        step = Step("my-step", None, ["out1", "out2", "out3", "out4", "out5", "out6"])
        step._set_output(
            {
                "out1": [{"a": "foo"}],
                "out2": [set(["a"])],
                "out3": [("a",)],
                "out4": [["a"]],
                "out5": [datetime.now()],
                "out6": [None],
            }
        )
        assert set(step.output.column_names) == set(
            ["out1", "out2", "out3", "out4", "out5", "out6"]
        )
        assert isinstance(step.output, OutputDataset)
        assert len(step.output["out1"]) == 1
        assert str(step.output.info.features) == (
            "{'out1': {'a': Value(dtype='string', id=None)},"
            " 'out2': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),"
            " 'out3': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),"
            " 'out4': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),"
            " 'out5': Value(dtype='timestamp[us]', id=None),"
            " 'out6': Value(dtype='null', id=None)}"
        )

    def test_iterable_dataset_features(self):
        step = Step("my-step", None, ["out1"])

        def dataset_generator():
            yield {"out1": {"foo": [5]}}

        step._set_output(LazyRows(dataset_generator, total_num_rows=1))
        assert isinstance(step.output, OutputIterableDataset)
        assert str(step.output.info.features) == (
            "{'out1': {'foo': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None)}}"  # noqa: B950
        )

    def test_iterable_dataset_features_empty(self):
        step = Step("my-step", None, ["out1"])

        def empty_generator():
            return iter(())

        step._set_output(LazyRows(empty_generator, total_num_rows=0))
        assert isinstance(step.output, OutputIterableDataset)
        assert str(step.output.info.features) == ("{'out1': None}")

    def test_func(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputTypeError):
            step._set_output({"out1": [lambda x: x]})

    def test_pickle(self):
        step = Step("my-step", None, ["out1", "out2", "out3", "out4"])
        datetime_now = datetime.now()
        step._set_output(
            {
                "out1": [step.pickle(lambda x: x)],
                "out2": [step.pickle(set("a"))],
                "out3": [b"foo"],
                "out4": [datetime_now],
            }
        )
        assert step.output["out1"][0].__code__.co_code == (lambda x: x).__code__.co_code
        assert step.output["out2"][0] == set("a")
        assert step.output["out3"][0] == b"foo"
        assert step.output["out4"][0] == datetime_now
        assert str(step.output.info.features) == (
            "{'out1': Value(dtype='binary', id=None),"
            " 'out2': Value(dtype='binary', id=None),"
            " 'out3': Value(dtype='binary', id=None),"
            " 'out4': Value(dtype='timestamp[us]', id=None)}"
        )

    def test_pickle_iterable_dataset(self):
        step = Step("my-step", None, ["out1", "out2", "out3", "out4"])
        datetime_now = datetime.now()

        def dataset_generator():
            yield {
                "out1": step.pickle(lambda x: x),
                "out2": step.pickle(set("a")),
                "out3": b"foo",
                "out4": datetime_now,
            }

        step._set_output(LazyRows(dataset_generator, total_num_rows=1))
        first_row = next(iter(step.output))
        assert first_row["out1"].__code__.co_code == (lambda x: x).__code__.co_code
        assert first_row["out2"] == set("a")
        assert first_row["out3"] == b"foo"
        assert first_row["out4"] == datetime_now
        assert str(step.output.info.features) == (
            "{'out1': Value(dtype='binary', id=None),"
            " 'out2': Value(dtype='binary', id=None),"
            " 'out3': Value(dtype='binary', id=None),"
            " 'out4': Value(dtype='timestamp[us]', id=None)}"
        )

    def test_unpickle(self):
        step = Step("my-step", None, ["out1", "out2", "out3", "out4"])
        datetime_now = datetime.now()
        step._set_output(
            {
                "out1": [step.pickle(lambda x: x)],
                "out2": [step.pickle(set("a"))],
                "out3": [b"foo"],
                "out4": [datetime_now],
            }
        )
        assert isinstance(step.output, OutputDataset)
        assert (
            step.unpickle(step.output.dataset["out1"][0]).__code__.co_code
            == (lambda x: x).__code__.co_code
        )
        assert step.unpickle(step.output.dataset["out2"][0]) == set("a")

    def test_dict_with_no_keys(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": [{}]})
        assert step.output["out1"][0] == {}

    def test_int_and_none(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": [5, None]})
        assert step.output["out1"][0] == 5
        assert step.output["out1"][1] is None

    def test_int_and_float(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": [5, 0.5]})
        assert step.output["out1"][0] == 5
        assert step.output["out1"][1] == 0.5

    def test_int_and_str(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputTypeError):
            step._set_output({"out1": [5, "a"]})

    def test_str_and_int(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputTypeError):
            step._set_output({"out1": ["a", 5]})

    def test_str_and_datetime(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputTypeError):
            step._set_output({"out1": ["a", datetime.now()]})

    def test_str_and_func(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputTypeError):
            step._set_output({"out1": ["a", lambda x: x]})

    def test_dict_and_none(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": [None, {"foo": 5}]})
        assert step.output["out1"][0] is None
        assert step.output["out1"][1] == {"foo": 5}

    def test_dict_with_different_keys(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": [{"foo": 5}, {"foo": 5, "bar": 5}]})
        assert step.output["out1"][0] == {"foo": 5, "bar": None}
        assert step.output["out1"][1] == {"foo": 5, "bar": 5}

    def test_non_dict_and_dict(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputTypeError):
            step._set_output({"out1": [5, {"foo": 5}]})

    def test_list_with_no_elements(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": [[]]})
        assert step.output["out1"][0] == []

    def test_list_with_different_lengths(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": [[5], [1, 2]]})
        assert step.output["out1"][0] == [5]
        assert step.output["out1"][1] == [1, 2]

    def test_list_with_int_and_str(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputTypeError):
            step._set_output({"out1": [[5, "a"]]})

    def test_list_with_str_and_int(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputTypeError):
            step._set_output({"out1": [["a", 5]]})

    def test_list_and_non_list(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(StepOutputTypeError):
            step._set_output({"out1": [5, [5]]})

    def test_iterable_dataset_int_and_str(self):
        step = Step("my-step", None, ["out1"])

        def dataset_generator():
            yield {"out1": 5}
            yield {"out1": "a"}

        step._set_output(LazyRows(dataset_generator, total_num_rows=2))
        with pytest.raises(StepOutputTypeError):
            assert [row["out1"] for row in list(step.output)] == [5, "a"]

    def test_iterable_dataset_str_and_int(self):
        step = Step("my-step", None, ["out1"])

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": 5}

        step._set_output(LazyRows(dataset_generator, total_num_rows=2))
        with pytest.raises(StepOutputTypeError):
            assert [row["out1"] for row in list(step.output)] == ["a", 5]

    def test_iterable_dataset_non_dict_and_dict(self):
        step = Step("my-step", None, ["out1"])

        def dataset_generator():
            yield {"out1": 5}
            yield {"out1": {"foo": 5}}

        step._set_output(LazyRows(dataset_generator, total_num_rows=2))

        with pytest.raises(StepOutputTypeError):
            assert [row["out1"] for row in list(step.output)] == [5, {"foo": 5}]

    def test_iterable_dataset_list_with_int_and_str(self):
        step = Step("my-step", None, ["out1"])

        def dataset_generator():
            yield {"out1": [5]}
            yield {"out1": ["a"]}

        step._set_output(LazyRows(dataset_generator, total_num_rows=2))
        with pytest.raises(StepOutputTypeError):
            assert [row["out1"] for row in list(step.output)] == [[5], ["a"]]

    def test_iterable_dataset_dict_different_shape(self):
        step = Step("my-step", None, ["out1"])

        def dataset_generator():
            yield {"out1": {"foo": 5}}
            yield {"out1": {"foo": {"baz": 6}, "bar": 7}}

        step._set_output(LazyRows(dataset_generator, total_num_rows=2))

        with pytest.raises(StepOutputTypeError):
            assert [row["out1"] for row in list(step.output)] == [
                {"foo": 5},
                {"foo": {"baz": 6}, "bar": 7},
            ]

    def test_iterable_dataset_list_with_str_and_int(self):
        step = Step("my-step", None, ["out1"])

        def dataset_generator():
            yield {"out1": ["a"]}
            yield {"out1": [5]}

        step._set_output(LazyRows(dataset_generator, total_num_rows=2))
        assert [row["out1"] for row in list(step.output)] == [["a"], ["5"]]

    def test_iterable_dataset_list_with_different_lengths(self):
        step = Step("my-step", None, ["out1"])

        def dataset_generator():
            yield {"out1": [5]}
            yield {"out1": [1, 2]}

        step._set_output(LazyRows(dataset_generator, total_num_rows=2))
        assert [row["out1"] for row in list(step.output)] == [[5], [1, 2]]
