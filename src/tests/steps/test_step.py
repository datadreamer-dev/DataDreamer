from typing import Callable

import pytest

from ...steps import LazyRows, Step, TraceInfoType


class TestErrors:
    def test_init_cannot_be_overridden(self):
        with pytest.raises(AttributeError):

            class OverrideStep(Step):
                def __init__(self):
                    pass

    def test_empty_name(self, create_test_step):
        with pytest.raises(ValueError):
            create_test_step(name="", inputs=None, output_names=["out1"])

    def test_step_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Step(name="my-step")

    def test_no_outputs_registered(self, create_test_step: Callable[..., Step]):
        with pytest.raises(ValueError):
            create_test_step(name="my-step", inputs=None, output_names=[])

    def test_invalid_arguments_to_register_functions(
        self, create_test_step: Callable[..., Step]
    ):
        def setup(self):
            with pytest.raises(TypeError):
                self.register_input(5)
            with pytest.raises(TypeError):
                self.register_arg(5)
            with pytest.raises(TypeError):
                self.register_output(5)
            with pytest.raises(TypeError):
                self.register_trace_info(5, "http://example.com")

        create_test_step(
            name="my-step", inputs=None, output_names=["out1"], setup=setup
        )

    def test_register_functions_run_after_initialization(
        self, create_test_step: Callable[..., Step]
    ):
        step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
        with pytest.raises(RuntimeError):
            step.register_input("in1")
        with pytest.raises(RuntimeError):
            step.register_arg("arg1")
        with pytest.raises(RuntimeError):
            step.register_output("out1")
        with pytest.raises(RuntimeError):
            step.register_trace_info(TraceInfoType.URL, "http://example.com")

    def test_invalid_initialization(self, create_test_step: Callable[..., Step]):
        def setup_inputs(self):
            self.register_output("foo1")
            self.register_input("in1")

        def setup_args(self):
            self.register_output("foo1")
            self.register_arg("arg1")

        def setup_output(self):
            self.register_output("foo1")

        with pytest.raises(ValueError):
            create_test_step(
                name="my-step",
                inputs=None,
                args=None,
                outputs=None,
                output_names=None,
                setup=setup_inputs,
            )
        with pytest.raises(TypeError):
            create_test_step(
                name="my-step",
                inputs={"in1": 5},
                args=None,
                outputs=None,
                output_names=None,
                setup=setup_inputs,
            )
        with pytest.raises(ValueError):
            create_test_step(
                name="my-step",
                inputs={"in1": 5, "in2": 6},
                args=None,
                outputs=None,
                output_names=None,
                setup=setup_inputs,
            )
        with pytest.raises(ValueError):
            create_test_step(
                name="my-step",
                inputs=None,
                args={"arg2": 5},
                outputs=None,
                output_names=None,
                setup=setup_args,
            )
        with pytest.raises(ValueError):
            create_test_step(
                name="my-step",
                inputs=None,
                args=None,
                outputs={"foo2": "out1"},
                output_names=None,
                setup=setup_output,
            )


class TestFunctionality:
    def test_new_step(self, create_test_step: Callable[..., Step]):
        step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
        assert step.name == "my-step"
        assert step.output_names == ("out1",)
        assert step.trace_info == {}

    def test_step_str_repr(self, create_test_step: Callable[..., Step]):
        def setup_1(self):
            self.register_arg("arg1")
            self.register_output("out1")

        step = create_test_step(
            name="my-step", args={"arg1": 5}, outputs={"out1": "foo"}, setup=setup_1
        )
        assert (
            str(step)
            == "TestStep(\n\tname='my-step',\n\targs={\n\t\t'arg1': 5\n\t},\n\tinputs={},\n\toutputs={\n\t\t'out1' => 'foo'\n\t},\n\toutput=None,\n)"  # noqa: B950
        )

        def setup_2(self):
            self.register_output("out1")
            self.register_output("out2")

        step = create_test_step(
            name="my-step", args=None, outputs={"out1": "foo"}, setup=setup_2
        )
        step._set_output({"out1": ["a", "b", "c"], "out2": [1, 2, 3]})
        assert str(step).startswith(
            "TestStep(\n\tname='my-step',\n\tinputs={},\n\toutputs={\n\t\t'out1' => 'foo',\n\t\t'out2' => 'out2'\n\t},\n\toutput=OutputDataset(column_names=['foo', 'out2'], num_rows=3, dataset=<Dataset @ "  # noqa: B950
        )
        assert str(step).endswith(">),\n)")

    def test_outputs_renames_columns(self, create_test_step: Callable[..., Step]):
        step = create_test_step(
            name="my-step",
            inputs=None,
            outputs={"out1": "out2", "out2": "out1"},
            output_names=["out1", "out2"],
        )
        step._set_output({"out1": ["a", "b", "c"], "out2": [1, 2, 3]})
        assert list(step.output["out2"]) == ["a", "b", "c"]
        assert list(step.output["out1"]) == [1, 2, 3]

    def test_trace_info(self, create_test_step: Callable[..., Step]):
        def setup(self):
            self.register_trace_info(TraceInfoType.CITATION, "citation")
            self.register_trace_info(TraceInfoType.URL, "http://example.com")

        trace_info = create_test_step(
            name="my-step", inputs=None, output_names=["out1"], setup=setup
        ).trace_info
        assert trace_info == {
            "my-step": {
                "Citation Information": ["citation"],
                "URL": ["http://example.com"],
            }
        }

    def test_head(self, create_test_step: Callable[..., Step]):
        step = create_test_step(
            name="my-step", inputs=None, output_names=["out1", "out2"]
        )
        step._set_output(
            {
                "out1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "out2": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            }
        )
        head_df = step.head()
        assert set(head_df.columns) == set(["out1", "out2"])
        assert len(head_df) == 5
        assert list(head_df["out1"]) == [1, 2, 3, 4, 5]
        assert list(head_df["out2"]) == ["a", "b", "c", "d", "e"]

    def test_head_shuffle(self, create_test_step: Callable[..., Step]):
        step = create_test_step(
            name="my-step", inputs=None, output_names=["out1", "out2"]
        )

        def dataset_generator():
            yield {"out1": 1, "out2": "a"}
            yield {"out1": 2, "out2": "b"}
            yield {"out1": 3, "out2": "c"}
            yield {"out1": 4, "out2": "d"}
            yield {"out1": 5, "out2": "e"}
            yield {"out1": 6, "out2": "f"}
            yield {"out1": 7, "out2": "g"}
            yield {"out1": 8, "out2": "h"}
            yield {"out1": 9, "out2": "i"}
            yield {"out1": 10, "out2": "j"}

        step._set_output(LazyRows(dataset_generator, total_num_rows=10))
        head_df = step.head(n=3, shuffle=True, buffer_size=7, seed=42)
        assert set(head_df.columns) == set(["out1", "out2"])
        assert len(head_df) == 3
        assert list(head_df["out1"]) == [1, 6, 5]
        assert list(head_df["out2"]) == ["a", "f", "e"]
