import os
from typing import Callable

import pytest

from ... import DataDreamer
from ...steps import (
    DataCardType,
    DataSource,
    LazyRows,
    Step,
    SuperStep,
    concurrent,
    wait,
)


class TestErrors:
    def test_init_cannot_be_overridden(self):
        with pytest.raises(AttributeError):

            class OverrideStep(Step):
                def __init__(self):
                    pass

    def test_empty_name(self, create_test_step):
        with pytest.raises(ValueError):
            create_test_step(name="", inputs=None, output_names=["out1"])

    def test_step_setup_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Step(name="my-step")

    def test_step_run_not_implemented(self, create_datadreamer):
        with pytest.raises(NotImplementedError):
            Step(name="my-step")

        class TestStep(Step):
            def setup(self):
                self.register_output("out1")

        # This will work fine
        TestStep(name="my-step")

        with create_datadreamer():
            # This will not work because now .run() will tried to be called
            with pytest.raises(NotImplementedError):
                TestStep(name="my-step")

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
                self.register_data_card(5, "http://example.com")

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
        step.register_data_card(DataCardType.URL, "http://example.com")

    def test_invalid_initialization(self, create_test_step: Callable[..., Step]):
        def setup_inputs(self):
            self.register_output("foo1")
            self.register_input("in1")

        def setup_non_required_inputs(self):
            self.register_output("foo1")
            self.register_input("in1", required=False)

        def setup_args(self):
            self.register_output("foo1")
            self.register_arg("arg1")

        def setup_args_with_kwargs(self):
            self.register_output("foo1")
            self.register_arg("arg1")
            self.register_arg("**kwargs")

        def setup_non_required_args(self):
            self.register_output("foo1")
            self.register_arg("arg1", required=False)

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
        create_test_step(
            name="my-step",
            inputs=None,
            args=None,
            outputs=None,
            output_names=None,
            setup=setup_non_required_inputs,
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
                args={"arg1": 4, "arg2": 5},
                outputs=None,
                output_names=None,
                setup=setup_args,
            )
        create_test_step(
            name="my-step",
            inputs=None,
            args={"arg1": 4, "arg2": 5},
            outputs=None,
            output_names=None,
            setup=setup_args_with_kwargs,
        )
        create_test_step(
            name="my-step",
            inputs=None,
            args={},
            outputs=None,
            output_names=None,
            setup=setup_non_required_args,
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
        assert set(step._data_card["my-step"].keys()) == {"Date & Time"}

    def test_step_help_string(self):
        class TestStep1(Step):
            def setup(self):
                pass

        assert (
            TestStep1.help
            == "TestStep1(\n\tname = 'The name of the step.',\n\toutputs = {},\n)"
        )

        class TestStep2(Step):
            def setup(self):
                self.register_arg("arg1", help="The first argument.")
                self.register_input("in1", help="The first input.")
                self.register_output("out1", help="The first output.")

        assert (
            TestStep2.help
            == "TestStep2(\n\tname = 'The name of the step.',\n\tinputs = {\n\t\t'in1': 'The first input.'\n\t},\n\targs = {\n\t\t'arg1': 'The first argument.'\n\t},\n\toutputs = {\n\t\t'out1': 'The first output.'\n\t},\n)"  # noqa: B950
        )

    def test_step_str_repr(self, create_test_step: Callable[..., Step]):
        def setup_1(self):
            self.register_arg("arg1", required=False, default=6)
            self.register_output("out1")

        step = create_test_step(
            name="my-step", args={}, outputs={"out1": "foo"}, setup=setup_1
        )
        assert (
            str(step)
            == "TestStep(\n\tname='my-step',\n\tinputs={},\n\targs={\n\t\t'arg1': 6\n\t},\n\toutputs={\n\t\t'out1' => 'foo'\n\t},\n\tprogress='0%',\n\toutput=None,\n)"  # noqa: B950
        )
        step = create_test_step(
            name="my-step", args={"arg1": 5}, outputs={"out1": "foo"}, setup=setup_1
        )
        assert (
            str(step)
            == "TestStep(\n\tname='my-step',\n\tinputs={},\n\targs={\n\t\t'arg1': 5\n\t},\n\toutputs={\n\t\t'out1' => 'foo'\n\t},\n\tprogress='0%',\n\toutput=None,\n)"  # noqa: B950
        )

        def setup_2(self):
            self.register_output("out1")
            self.register_output("out2")

        step = create_test_step(
            name="my-step", args=None, outputs={"out1": "foo"}, setup=setup_2
        )
        step._set_output({"out1": ["a", "b", "c"], "out2": [1, 2, 3]})
        assert str(step).startswith(
            "TestStep(\n\tname='my-step',\n\tinputs={},\n\toutputs={\n\t\t'out1' => 'foo',\n\t\t'out2' => 'out2'\n\t},\n\tprogress='100%',\n\toutput=OutputDataset(column_names=['foo', 'out2'], num_rows=3, dataset=<Dataset @ "  # noqa: B950
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

    def test_data_card(self, create_test_step: Callable[..., Step]):
        def setup(self):
            self.register_data_card(DataCardType.CITATION, "citation")
            self.register_data_card(DataCardType.URL, "http://example.com")

        data_card = create_test_step(
            name="my-step", inputs=None, output_names=["out1"], setup=setup
        )._data_card
        del data_card["my-step"]["Date & Time"]
        assert data_card == {
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


class ChildStep(Step):
    def setup(self):
        self.register_input("in1")
        self.register_output("out1")
        self.register_data_card(DataCardType.URL, "http://example.com")

    def run(self):
        return [x + 1 for x in self.inputs["in1"]]


class ParentStep(SuperStep):
    def setup(self):
        self.register_input("in1")
        self.register_output("out1")

    def run(self):
        child_step = ChildStep("child", inputs={"in1": self.inputs["in1"]})
        wait(child_step)
        return child_step.output.dataset


class NonSuperParentStep(Step):
    def setup(self):
        self.register_input("in1")
        self.register_output("out1")

    def run(self):
        child_step = ChildStep("child", inputs={"in1": self.inputs["in1"]})
        wait(child_step)
        return child_step.output.dataset


class ConcurrentParentStep(SuperStep):
    def setup(self):
        self.register_input("in1")
        self.register_output("out1")

    def run(self):
        child_step = concurrent(
            lambda: ChildStep("child", inputs={"in1": self.inputs["in1"]})
        )[0]
        wait(child_step)
        return child_step.output.dataset


class GrandParentStep(SuperStep):
    def setup(self):
        self.register_input("in1")
        self.register_output("out1")
        self.register_arg("parent_class")

    def run(self):
        parent_step = self.args["parent_class"](
            "parent", inputs={"in1": self.inputs["in1"]}
        )
        assert not parent_step.background
        return parent_step.output.dataset


def _assert_super_step_structure():
    grandparent_step_1_path = os.path.join(
        DataDreamer.get_output_folder_path(), "grandparent"
    )
    grandparent_step_2_path = os.path.join(
        DataDreamer.get_output_folder_path(), "grandparent2"
    )
    assert os.path.isfile(
        os.path.join(grandparent_step_1_path, "_dataset", "dataset_info.json")
    )
    assert os.path.isfile(
        os.path.join(grandparent_step_1_path, "parent", "_dataset", "dataset_info.json")
    )
    assert os.path.isfile(
        os.path.join(
            grandparent_step_1_path, "parent", "child", "_dataset", "dataset_info.json"
        )
    )
    assert os.path.isfile(
        os.path.join(grandparent_step_2_path, "_dataset", "dataset_info.json")
    )
    assert os.path.isfile(
        os.path.join(grandparent_step_2_path, "parent", "_dataset", "dataset_info.json")
    )
    assert os.path.isfile(
        os.path.join(
            grandparent_step_2_path, "parent", "child", "_dataset", "dataset_info.json"
        )
    )


class TestSuperStep:
    def test_non_superstep(self, create_datadreamer):
        with create_datadreamer():
            ds_step = DataSource("Initial", data={"out1": [1, 2, 3]})
            with pytest.raises(RuntimeError):
                NonSuperParentStep("parent", inputs={"in1": ds_step.output["out1"]})

    def test_superstep(self, create_datadreamer):
        with create_datadreamer():
            ds_step = DataSource("Initial", data={"out1": [1, 2, 3]})
            grandparent_step_1 = GrandParentStep(
                "grandparent",
                inputs={"in1": ds_step.output["out1"]},
                args={"parent_class": ParentStep},
            )
            grandparent_step_2 = GrandParentStep(
                "grandparent2",
                inputs={"in1": grandparent_step_1.output["out1"]},
                args={"parent_class": ParentStep},
            )
            assert list(grandparent_step_2.output["out1"]) == [3, 4, 5]
            _assert_super_step_structure()

    def test_superstep_concurrent(self, create_datadreamer):
        with create_datadreamer():
            ds_step = DataSource("Initial", data={"out1": [1, 2, 3]})
            grandparent_step_1 = GrandParentStep(
                "grandparent",
                inputs={"in1": ds_step.output["out1"]},
                args={"parent_class": ConcurrentParentStep},
            )
            grandparent_step_2 = GrandParentStep(
                "grandparent2",
                inputs={"in1": grandparent_step_1.output["out1"]},
                args={"parent_class": ConcurrentParentStep},
            )
            assert list(grandparent_step_2.output["out1"]) == [3, 4, 5]
            _assert_super_step_structure()

    def test_superstep_background(self, create_datadreamer):
        with create_datadreamer():
            ds_step = DataSource("Initial", data={"out1": [1, 2, 3]})
            grandparent_step_1 = GrandParentStep(
                "grandparent",
                inputs={"in1": ds_step.output["out1"]},
                args={"parent_class": ParentStep},
                background=True,
            )
            assert not grandparent_step_1.background
            grandparent_step_2 = GrandParentStep(
                "grandparent2",
                inputs={"in1": grandparent_step_1.output["out1"]},
                args={"parent_class": ParentStep},
                background=True,
            )
            assert not grandparent_step_2.background
            assert list(grandparent_step_2.output["out1"]) == [3, 4, 5]
            _assert_super_step_structure()

    def test_superstep_data_card_propagates(self, create_datadreamer):
        with create_datadreamer():
            ds_step = DataSource("Initial", data={"out1": [1, 2, 3]})
            grandparent_step = GrandParentStep(
                "grandparent",
                inputs={"in1": ds_step.output["out1"]},
                args={"parent_class": ParentStep},
            )
            child_data_card = grandparent_step._data_card[
                "grandparent / parent / child"
            ]
            del child_data_card["Date & Time"]
            assert child_data_card == {DataCardType.URL: ["http://example.com"]}
