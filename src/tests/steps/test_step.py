from ...steps.step import Step


class TestOutput:
    def test_step_single_output_list(self):
        step = Step("my-step", None, "out1")
        step._set_output(["a", "b", "c"])
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_step_single_output_tuple_of_list(self):
        step = Step("my-step", None, "out1")
        step._set_output((["a", "b", "c"],))
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_step_single_output_tuple_of_iterator(self):
        step = Step("my-step", None, "out1")
        step._set_output((range(3),))
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == 0

    def test_step_single_output_dict_of_list(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": ["a", "b", "c"]})
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == "a"
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == "a"

    def test_step_single_output_dict_of_iterator(self):
        step = Step("my-step", None, "out1")
        step._set_output({"out1": range(3)})
        assert set(step.output.column_names) == set(["out1"])
        assert len(step.output["out1"]) == 3
        assert step.output["out1"][0] == 0
        assert set(step.output[0].keys()) == set(step.output.column_names)
        assert step.output[0]["out1"] == 0
