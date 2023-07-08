from datasets import Dataset, IterableDataset

from ...steps import Step


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

        def dataset_dict_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = IterableDataset.from_generator(dataset_dict_generator)
        step._set_output(iterable_dataset)
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


class TestSingleOutput:
    def test_output_list(self):
        step = Step("my-step", None, "out1")
        step._set_output(["a", "b", "c"])
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_list_of_tuple(self):
        step = Step("my-step", None, "out1")
        step._set_output([("a",), ("b",), ("c",)])
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == ["a"]
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == ["a"]

    def test_output_list_of_list(self):
        step = Step("my-step", None, "out1")
        step._set_output([["a", "b", "c"]])
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 1
        assert step.output["out1"][0] == ["a", "b", "c"]
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == ["a", "b", "c"]

    def test_output_tuple_of_list(self):
        step = Step("my-step", None, "out1")
        step._set_output((["a", "b", "c"],))
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_tuple_of_iterator(self):
        step = Step("my-step", None, "out1")
        step._set_output((range(3),))
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == 0

    def test_output_dict_of_list(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": ["a", "b", "c"]})
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_dict_of_iterator(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": range(3)})
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == 0

    def test_output_dataset(self):
        step = Step("my-step", None, "out1")
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = Dataset.from_dict(dataset_dict)
        step._set_output(dataset)
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_iterable_dataset(self):
        step = Step("my-step", None, "out1")

        def dataset_dict_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        iterable_dataset = IterableDataset.from_generator(dataset_dict_generator)
        step._set_output(iterable_dataset)
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"

    def test_output_list_of_datasets(self):
        step = Step("my-step", None, "out1")
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = Dataset.from_dict(dataset_dict)
        step._set_output([dataset])
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_tuple_of_datasets(self):
        step = Step("my-step", None, "out1")
        dataset_dict = {"out1": ["a", "b", "c"]}
        dataset = Dataset.from_dict(dataset_dict)
        step._set_output((dataset,))
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_output_generator_function(self):
        step = Step("my-step", None, "out1")

        def dataset_dict_generator():
            yield {"out1": "a"}
            yield {"out1": "b"}
            yield {"out1": "c"}

        step._set_output(dataset_dict_generator)
        assert set(step.output.column_names) == set(["out1"])
        assert isinstance(step.output, IterableDataset)
        assert len(list(step.output)) == 3
        first_row = next(iter(step.output))
        assert set(first_row.keys()) == set(step.output.column_names)
        assert first_row["out1"] == "a"