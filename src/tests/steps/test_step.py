import pytest

from datasets import Dataset, IterableDataset

from ...steps import LazyRowBatches, LazyRows, Step


class TestErrors:
    def test_no_outputs_named(self):
        with pytest.raises(ValueError):
            Step("my-step", None, [])

    def test_access_output_before_step_is_run(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(AttributeError):
            step.output

    def test_access_output_while_step_is_running(self):
        step = Step("my-step", None, "out1")
        step.progress = 0.565
        assert step.progress == 0.565
        with pytest.raises(AttributeError):
            step.output

    def test_output_invalid_type(self):
        step_single = Step("my-step", None, "out1")
        with pytest.raises(AttributeError):
            step_single._set_output({"out2": 5})
        with pytest.raises(AttributeError):
            step_single._set_output({"out2": "a"})

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = IterableDataset.from_generator(dataset_generator)
        with pytest.raises(AttributeError):
            step_single._set_output(iterable_dataset)  # type: ignore[arg-type]

        step_multiple = Step("my-step", None, ["out1", "out2"])
        with pytest.raises(AttributeError):
            step_multiple._set_output(
                LazyRows([iterable_dataset, [1, 2, 3]], total_num_rows=3)
            )

    def test_output_dict_with_wrong_keys(self):
        step = Step("my-step", None, "out1")
        with pytest.raises(AttributeError):
            step._set_output({"out2": ["a", "b", "c"]})

    def test_output_list_with_wrong_number_of_outputs(self):
        step = Step("my-step", None, ["out1", "out2", "out3", "out4"])
        with pytest.raises(AttributeError):
            step._set_output([("a", 1), ("b", 2), ("c", 3)])

    def test_output_tuple_with_wrong_number_of_outputs(self):
        step = Step("my-step", None, ["out1"])
        with pytest.raises(AttributeError):
            step._set_output((["a", "b", "c"], [1, 2, 3]))

    def test_output_dataset_with_wrong_number_of_columns(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None
        dataset_dict = {"out1": ["a", "b", "c"], "out2": [1, 2, 3]}
        dataset = Dataset.from_dict(dataset_dict)
        with pytest.raises(AttributeError):
            step._set_output(dataset)

        def dataset_generator():
            yield {"out1": "a", "out2": 1}
            yield {"out1": "b", "out2": 2}
            yield {"out1": "c", "out2": 3}

        iterable_dataset = IterableDataset.from_generator(dataset_generator)
        with pytest.raises(AttributeError):
            step._set_output(LazyRows(iterable_dataset, total_num_rows=3))

    def test_output_generator_function_of_dict_with_wrong_keys(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out2": "a"}
            yield {"out2": "b"}
            yield {"out2": "c"}

        with pytest.raises(AttributeError):
            step._set_output(LazyRows(dataset_generator, total_num_rows=3))

    def test_output_generator_function_of_list_with_wrong_number_of_outputs(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ["a"]
            yield ["b"]
            yield ["c"]

        with pytest.raises(AttributeError):
            step._set_output(LazyRows(dataset_generator, total_num_rows=3))

    def test_output_generator_function_of_tuple_with_wrong_number_of_outputs(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ("a",)
            yield ("b",)
            yield ("c",)

        with pytest.raises(AttributeError):
            step._set_output(LazyRows(dataset_generator, total_num_rows=3))

    def test_output_generator_function_of_tuple_batched_column_with_wrong_number_of_outputs(
        self,
    ):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield (["a", "b"],)
            yield (["c"],)

        with pytest.raises(AttributeError):
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
        dataset = Dataset.from_dict(dataset_dict)
        step._set_output(dataset)
        assert step.progress == 1.0

    def test_progress_after_output_iterable_dataset(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = IterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows(iterable_dataset, total_num_rows=3))
        assert step.progress == 0.0

    def test_progress_is_1_after__output_is_dataset(self):
        step = Step("my-step", None, "out1")
        assert step.progress is None
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = Dataset.from_dict(dataset_dict)
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
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        next(iter(step.output))
        assert step.progress == 1.0 / 3.0

    def test_auto_progress_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [{"out1": "a"}, {"out1": "b"}, {"out1": "c"}]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        next(iter(step.output))
        assert step.progress == 1.0 / 3.0

class TestEmptyOutput():
    def test_output_single_empty(self):
        step = Step("my-step", None, "out1")
        step._set_output(None)
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 0
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]

    def test_output_list_empty(self):
        step = Step("my-step", None, "out1")
        step._set_output([])
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 0
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]

    def test_output_tuple_of_list_empty(self):
        step = Step("my-step", None, "out1")
        step._set_output(([],))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 0
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]

    def test_output_dict_of_list_empty(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": []})
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 0
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]

    def test_output_generator_function_empty(self):
        step = Step("my-step", None, "out1")

        def empty_generator():
            return iter(())

        step._set_output(LazyRows(empty_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(list(step.output)) == 0
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]

    def test_output_generator_function_of_tuple_batched_column_empty(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ([], [])
            yield ([], [])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(list(step.output)) == 0
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]

    def test_output_generator_function_of_dict_batched_empty(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": []}
            yield {"out1": []}

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(list(step.output)) == 0
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]

    def test_output_generator_function_of_tuple_batched_row_empty(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield []
            yield []

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(list(step.output)) == 0
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]


class TestSingleOutput:
    def test_output_single(self):
        step = Step("my-step", None, "out1")
        step._set_output("a")  # type: ignore[arg-type]
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_single_list(self):
        step = Step("my-step", None, "out1")
        step._set_output(["a"])
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_single_tuple(self):
        step = Step("my-step", None, "out1")
        step._set_output(("a",))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_single_dict(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": "a"})
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_list(self):
        step = Step("my-step", None, "out1")
        step._set_output(["a", "b", "c"])
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_list_of_tuple(self):
        step = Step("my-step", None, "out1")
        step._set_output([("a",), ("b",), ("c",)])
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_list_of_list_row(self):
        step = Step("my-step", None, "out1")
        step._set_output([["a"], ["b"], ["c"]])
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == ["a"]
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == ["a"]

    def test_output_list_of_list_column(self):
        step = Step("my-step", None, "out1")
        step._set_output([["a", "b", "c"]])
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == ["a", "b", "c"]
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == ["a", "b", "c"]

    def test_output_tuple_of_list(self):
        step = Step("my-step", None, "out1")
        step._set_output((["a", "b", "c"],))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_tuple_of_iterator(self):
        step = Step("my-step", None, "out1")
        step._set_output((range(3),))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == 0

    def test_output_dict_of_list(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": ["a", "b", "c"]})
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_dict_of_iterator(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": range(3)})
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == 0

    def test_output_dataset(self):
        step = Step("my-step", None, "out1")
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = Dataset.from_dict(dataset_dict)
        step._set_output(dataset)
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_iterable_dataset(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = IterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows(iterable_dataset, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_iterable_dataset_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = IterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRowBatches(iterable_dataset, total_num_rows=3))  # type: ignore[arg-type]
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_list_of_datasets(self):
        step = Step("my-step", None, "out1")
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = Dataset.from_dict(dataset_dict)
        step._set_output([dataset])
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_tuple_of_datasets(self):
        step = Step("my-step", None, "out1")
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = Dataset.from_dict(dataset_dict)
        step._set_output((dataset,))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"

    def test_output_generator_function_of_dict(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"


    def test_output_generator_function_of_dict_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": ["a", "b"]}
            yield {"out1": ["c"]}

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_dict_batched_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield {"out1": iter(["a", "b"])}
            yield {"out1": iter(["c"])}

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ("a",)
            yield ("b",)
            yield ("c",)

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple_batched_row(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [("a",), ("b",)]
            yield [("c",)]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple_batched_column(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield (["a", "b"],)
            yield (["c"],)

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple_batched_row_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter([("a",), ("b",)])
            yield iter([("c",)])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_tuple_batched_column_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield (iter(["a", "b"]),)
            yield (iter(["c"]),)

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_list_row(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ["a"]
            yield ["b"]
            yield ["c"]

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == ["a"]

    def test_output_generator_function_of_list_row_batched_row(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [["a"], ["b"]]
            yield [["c"]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == ["a"]

    def test_output_generator_function_of_list_row_batched_column(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [["a", "b"]]
            yield [["c"]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_generator_function_of_list_row_batched_row_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter([["a"], ["b"]])
            yield iter([["c"]])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == ["a"]

    def test_output_generator_function_of_list_row_batched_column_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield [iter(["a", "b"])]
            yield [iter(["c"])]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_dict_of_generator_function(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows({"out1": dataset_generator}, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_dict_of_generator_function_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ["a", "b"]
            yield ["c"]

        step._set_output(LazyRowBatches({"out1": dataset_generator}, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_dict_of_generator_function_batched_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter(["a", "b"])
            yield iter(["c"])

        step._set_output(LazyRowBatches({"out1": dataset_generator}, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_list_of_generator_function(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows([dataset_generator], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_list_of_generator_function_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ["a", "b"]
            yield ["c"]

        step._set_output(LazyRowBatches([dataset_generator], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_list_of_generator_function_batched_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter(["a", "b"])
            yield iter(["c"])

        step._set_output(LazyRowBatches([dataset_generator], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_tuple_of_generator_function(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows((dataset_generator,), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_tuple_of_generator_function_batched(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield ["a", "b"]
            yield ["c"]

        step._set_output(LazyRowBatches((dataset_generator,), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"

    def test_output_tuple_of_generator_function_batched_iterator(self):
        step = Step("my-step", None, "out1")

        def dataset_generator():
            yield iter(["a", "b"])
            yield iter(["c"])

        step._set_output(LazyRowBatches((dataset_generator,), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert first_row["out1"] == "a"


class TestMultipleOutput:
    def test_output_single_list(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output(["a", 1])
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"
        assert step.output[0]["out2"] == 1

    def test_output_single_tuple(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output(("a", 1))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"
        assert step.output[0]["out2"] == 1

    def test_output_single_dict(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output({"out1": "a", "out2": 1})
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert step.output[0]["out1"] == "a"
        assert step.output[0]["out2"] == 1

    def test_output_list_of_tuple(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output([("a", 1), ("b", 2), ("c", 3)])
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_list_of_list_row(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output([["a", 1], ["b", 2], ["c", 3]])
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_list_of_list_column(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output([["a", "b", "c"], [1, 2, 3]])
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_tuple_of_list(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output((["a", "b", "c"], [1, 2, 3]))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_tuple_of_iterator(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output((range(3), ["a", "b", "c"]))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert list(step.output["out1"]) == [0, 1, 2]
        assert list(step.output["out2"]) == ["a", "b", "c"]

    def test_output_dict_of_list(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output({"out1": ["a", "b", "c"], "out2": [1, 2, 3]})
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_dict_of_iterator(self):
        step = Step("my-step", None, ["out1", "out2"])
        step._set_output({"out1": range(3), "out2": ["a", "b", "c"]})
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert list(step.output["out1"]) == [0, 1, 2]
        assert list(step.output["out2"]) == ["a", "b", "c"]

    def test_output_dataset(self):
        step = Step("my-step", None, ["out1", "out2"])
        dataset_dict = {"out1": ["a", "b", "c"], "out2": [1, 2, 3]}
        dataset = Dataset.from_dict(dataset_dict)
        step._set_output(dataset)
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert list(step.output["out1"]) == ["a", "b", "c"]
        assert list(step.output["out2"]) == [1, 2, 3]

    def test_output_iterable_dataset(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield {"out1": "a", "out2": 1}
            yield {"out1": "b", "out2": 2}
            yield {"out1": "c", "out2": 3}

        iterable_dataset = IterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows(iterable_dataset, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_iterable_dataset_batched(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield {"out1": "a", "out2": 1}
            yield {"out1": "b", "out2": 2}
            yield {"out1": "c", "out2": 3}

        iterable_dataset = IterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRowBatches(iterable_dataset, total_num_rows=3))  # type: ignore[arg-type]
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_list_of_datasets(self):
        step = Step("my-step", None, ["out1", "out2"])
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = Dataset.from_dict(dataset_dict)

        def dataset_generator():
            yield {"out2": 1}
            yield {"out2": 2}
            yield {"out2": 3}

        iterable_dataset = IterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows([dataset, iterable_dataset], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_tuple_of_datasets(self):
        step = Step("my-step", None, ["out1", "out2"])
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = Dataset.from_dict(dataset_dict)

        def dataset_generator():
            yield {"out2": 1}
            yield {"out2": 2}
            yield {"out2": 3}

        iterable_dataset = IterableDataset.from_generator(dataset_generator)
        step._set_output(LazyRows((dataset, iterable_dataset), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_dict(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield {"out1": "a", "out2": 1}
            yield {"out1": "b", "out2": 2}
            yield {"out1": "c", "out2": 3}

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_dict_batched(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield {"out1": ["a", "b"], "out2": [1, 2]}
            yield {"out1": ["c"], "out2": [3]}

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_tuple(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ("a", 1)
            yield ("b", 2)
            yield ("c", 3)

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_tuple_batched_row(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [("a", 1), ("b", 2)]
            yield [("c", 3)]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_tuple_batched_column(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield (["a", "b"], [1, 2])
            yield (["c"], [3])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_list_row(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield ["a", 1]
            yield ["b", 2]
            yield ["c", 3]

        step._set_output(LazyRows(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_list_row_batched_row(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [["a", 1], ["b", 2]]
            yield [["c", 3]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_list_row_batched_column_ambiguous(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [["a", "b"], [1, 2]]
            yield [["c"], [3]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", 1, "c"]
        assert [row["out2"] for row in list(step.output)] == ["b", 2, 3]

    def test_output_generator_function_of_list_row_batched_column(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [["a", "b", "c"], [1, 2, 3]]
            yield [["d"], [4]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 4
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c", "d"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3, 4]

    def test_output_generator_function_of_list_row_batched_row_iterator(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield iter([["a", 1], ["b", 2]])
            yield iter([["c", 3]])

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_generator_function_of_list_row_batched_column_iterator(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield [iter(["a", "b"]), [1, 2]]
            yield [iter(["c"]), [3]]

        step._set_output(LazyRowBatches(dataset_generator, total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
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
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
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
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_list_of_generator_function(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows([dataset_generator, [1, 2, 3]], total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
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
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]

    def test_output_tuple_of_generator_function(self):
        step = Step("my-step", None, ["out1", "out2"])

        def dataset_generator():
            yield "a"
            yield "b"
            yield "c"

        step._set_output(LazyRows((dataset_generator, [1, 2, 3]), total_num_rows=3))
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
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
        assert set(step.output.column_names) == set(["out1", "out2"])  # type: ignore[arg-type]
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)  # type: ignore[arg-type]
        assert [row["out1"] for row in list(step.output)] == ["a", "b", "c"]
        assert [row["out2"] for row in list(step.output)] == [1, 2, 3]
