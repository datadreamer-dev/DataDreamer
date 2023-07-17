import pytest

from ...steps import LazyRows, Step


class TestErrors:
    def test_no_outputs_named(self):
        with pytest.raises(ValueError):
            Step("my-step", None, [])


class TestFunctionality:
    def test_new_step(self):
        step = Step("my-step", None, ["out1"])
        assert step.name == "my-step"
        assert step.output_names == ("out1",)

    def test_head(self):
        step = Step("my-step", None, ["out1", "out2"])
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

    def test_head_shuffle(self):
        step = Step("my-step", None, ["out1", "out2"])

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
