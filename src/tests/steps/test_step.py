import pytest

from ...steps import Step


class TestErrors:
    def test_no_outputs_named(self):
        with pytest.raises(ValueError):
            Step("my-step", None, [])


class TestFunctionality:
    def test_new_step(self):
        step = Step("my-step", None, ["out1"])
        assert step.name == "my-step"
        assert step.output_names == ("out1",)
