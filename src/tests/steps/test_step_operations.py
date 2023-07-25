import os
from typing import Callable

import pytest

from datasets import Dataset

from ... import DataDreamer
from ...datasets import OutputDataset, OutputIterableDataset
from ...errors import StepOutputTypeError
from ...steps import LazyRows, Step, concat, zipped
from ...steps.step import (
    ConcatStep,
    FilterStep,
    MapStep,
    SaveStep,
    SelectColumnsStep,
    SelectStep,
    ShuffleStep,
    ZippedStep,
)


class TestSave:
    def test_save_step_naming(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                {
                    "out1": [
                        step.pickle(set(["a"])),
                        step.pickle(set(["b"])),
                        step.pickle(set(["c"])),
                    ]
                }
            )
            save_step_1 = step.save()
            save_step_2 = step.save()
            save_step_1_path = os.path.join(
                DataDreamer.get_output_folder_path(), "my-step-save"
            )
            save_step_2_path = os.path.join(
                DataDreamer.get_output_folder_path(), "my-step-save-2"
            )
            assert save_step_1.name == "my-step (save)"
            assert save_step_2.name == "my-step (save #2)"
            assert os.path.isdir(save_step_1_path)
            assert os.path.isdir(save_step_2_path)

    def test_save_on_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                {
                    "out1": [
                        step.pickle(set(["a"])),
                        step.pickle(set(["b"])),
                        step.pickle(set(["c"])),
                    ]
                }
            )
            save_step = step.save()
            assert type(save_step).__name__ == "SaveStep"
            assert isinstance(save_step, SaveStep)
            assert isinstance(save_step.output, OutputDataset)
            save_step_path = os.path.join(
                DataDreamer.get_output_folder_path(), "my-step-save"
            )
            assert os.path.isdir(save_step_path)
            assert os.path.isfile(
                os.path.join(
                    DataDreamer.get_output_folder_path(),
                    "my-step-save",
                    "dataset",
                    "dataset_info.json",
                )
            )
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            assert step._resumed
            save_step = step.save()
            assert save_step._resumed
            assert save_step.output._pickled
            assert save_step.output["out1"][0] == set(["a"])

    def test_save_on_iterable_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict(
                        {
                            "out1": [
                                step.pickle(set(["a"])),
                                step.pickle(set(["b"])),
                                step.pickle(set(["c"])),
                            ]
                        }
                    ).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            save_step = step.save(name="save-step", writer_batch_size=1000)
            step_path = os.path.join(DataDreamer.get_output_folder_path(), "my-step")
            assert os.path.isdir(os.path.join(step_path, "cache", "generator"))
            assert type(save_step).__name__ == "SaveStep"
            assert isinstance(save_step, SaveStep)
            assert isinstance(save_step.output, OutputDataset)
            save_step_path = os.path.join(
                DataDreamer.get_output_folder_path(), "save-step"
            )
            assert os.path.isdir(save_step_path)
            assert os.path.isfile(
                os.path.join(
                    DataDreamer.get_output_folder_path(),
                    "save-step",
                    "dataset",
                    "dataset_info.json",
                )
            )
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": ["a", "b", "c"]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            save_step = step.save(name="save-step", writer_batch_size=1000)
            assert save_step._resumed
            assert save_step.output._pickled
            assert save_step.output["out1"][0] == set(["a"])

    def test_save_num_shards(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step",
                inputs=None,
                output_names=["out1"],
                save_num_proc=3,
                save_num_shards=3,
            )
            step._set_output(
                {
                    "out1": [
                        step.pickle(set(["a"])),
                        step.pickle(set(["b"])),
                        step.pickle(set(["c"])),
                    ]
                }
            )
            step.save()
            save_step_path = os.path.join(
                DataDreamer.get_output_folder_path(), "my-step-save"
            )
            assert os.path.isfile(
                os.path.join(save_step_path, "dataset", "data-00000-of-00003.arrow")
            )

        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict(
                        {
                            "out1": [
                                step.pickle(set(["a"])),
                                step.pickle(set(["b"])),
                                step.pickle(set(["c"])),
                            ]
                        }
                    ).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            step.save(
                name="save-step",
                writer_batch_size=1000,
                save_num_proc=3,
                save_num_shards=3,
            )
            save_step_path = os.path.join(
                DataDreamer.get_output_folder_path(), "save-step"
            )
            assert os.path.isfile(
                os.path.join(save_step_path, "dataset", "data-00000-of-00003.arrow")
            )


class TestMap:
    def test_map_on_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output({"out1": [1, 2, 3]})
            with pytest.warns(UserWarning):
                map_step = step.map(lambda row: {"out1": row["out1"] * 2})
            assert isinstance(map_step, MapStep)
            assert isinstance(map_step.output, OutputIterableDataset)
            assert map_step.output.num_rows is None
            assert list(map_step.output["out1"])[2] == 6
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            with pytest.warns(UserWarning):
                map_step = step.map(lambda row: {"out1": row["out1"] * 2})
            assert list(map_step.output["out1"])[2] == 6

    def test_map_on_iterable_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict(
                        {
                            "out1": [
                                step.pickle(set(["a"])),
                                step.pickle(set(["b"])),
                                step.pickle(set(["c"])),
                            ]
                        }
                    ).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            with pytest.warns(UserWarning):
                map_step = step.map(
                    lambda batch, idxs: {
                        "out1": [
                            row.add(idx) or step.pickle(row)
                            for row, idx in zip(batch["out1"], idxs)
                        ]
                    },
                    with_indices=True,
                    batched=True,
                    batch_size=3,
                )
            assert isinstance(map_step, MapStep)
            assert isinstance(map_step.output, OutputIterableDataset)
            assert map_step.output.num_rows is None
            assert list(map_step.output["out1"])[2] == set(["c", 2])

    def test_map_with_total_num_rows(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output({"out1": [1, 2, 3]})
            map_step = step.map(lambda row: {"out1": row["out1"] * 2}, total_num_rows=3)
            assert isinstance(map_step, MapStep)
            assert isinstance(map_step.output, OutputIterableDataset)
            assert map_step.output.num_rows == 3

    def test_map_add_remove_columns(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output({"out1": [1, 2, 3]})
            map_step = step.map(
                lambda row: {"out1": row["out1"], "out2": row["out1"] * 2},
                lazy=False,
                total_num_rows=3,
            )
            assert isinstance(map_step, MapStep)
            assert isinstance(map_step.output, OutputDataset)
            assert map_step.output["out1"][2] == 3
            assert map_step.output["out2"][2] == 6

        with create_datadreamer():
            step = create_test_step(
                name="my-step", inputs=None, output_names=["out1", "out2"]
            )
            step._set_output({"out1": [1, 2, 3], "out2": ["a", "b", "c"]})
            map_step = step.map(
                lambda row: {"out1": row["out2"]},
                remove_columns=["out2"],
                lazy=False,
                total_num_rows=3,
            )
            assert isinstance(map_step, MapStep)
            assert isinstance(map_step.output, OutputDataset)
            assert set(map_step.output.column_names) == set(["out1"])
            assert map_step.output["out1"][2] == "c"

    def test_map_pickle_error(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                {
                    "out1": [
                        step.pickle(lambda x: x),
                        step.pickle(lambda x: x),
                        step.pickle(lambda x: x),
                    ]
                }
            )
            with pytest.raises(StepOutputTypeError):
                step.map(
                    lambda row: {"out1": row["out1"]}, lazy=False, total_num_rows=3
                )

    def test_map_lazy_pickle_error(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                {
                    "out1": [
                        step.pickle(lambda x: x),
                        step.pickle(lambda x: x),
                        step.pickle(lambda x: x),
                    ]
                }
            )
            with pytest.raises(StepOutputTypeError):
                list(
                    step.map(lambda row: {"out1": row["out1"]}, total_num_rows=3).output
                )


class TestFilter:
    def test_filter_on_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output({"out1": [1, 2, 3]})
            with pytest.warns(UserWarning):
                filter_step = step.filter(lambda row: row["out1"] in [1, 3])
            assert isinstance(filter_step, FilterStep)
            assert isinstance(filter_step.output, OutputIterableDataset)
            assert filter_step.output.num_rows is None
            assert list(filter_step.output["out1"]) == [1, 3]

    def test_filter_on_iterable_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": [1, 2, 3]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            with pytest.warns(UserWarning):
                filter_step = step.filter(lambda row: row["out1"] in [1, 3])
            assert isinstance(filter_step, FilterStep)
            assert isinstance(filter_step.output, OutputIterableDataset)
            assert filter_step.output.num_rows is None
            assert list(filter_step.output["out1"]) == [1, 3]

    def test_filter_with_total_num_rows(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output({"out1": [1, 2, 3]})
            filter_step = step.filter(
                lambda row: row["out1"] in [1, 3], total_num_rows=2
            )
            assert isinstance(filter_step, FilterStep)
            assert isinstance(filter_step.output, OutputIterableDataset)
            assert filter_step.output.num_rows == 2
            assert list(filter_step.output["out1"]) == [1, 3]


class TestShuffle:
    def test_dataset_lazy(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output({"out1": [1, 2, 3]})
            shuffle_step = step.shuffle(seed=42)
            assert isinstance(shuffle_step, ShuffleStep)
            assert (
                shuffle_step.output.dataset._indices  # type:ignore[union-attr]
                is not None
            )
            assert shuffle_step.progress == 1.0
            assert shuffle_step.output["out1"][0] == 3
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            shuffle_step = step.shuffle(seed=42)
            assert not shuffle_step._resumed
            assert (
                shuffle_step.output.dataset._indices  # type:ignore[union-attr]
                is not None
            )
            assert shuffle_step.output["out1"][0] == 3

    def test_iterable_dataset_lazy(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": [1, 2, 3]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            shuffle_step = step.shuffle(seed=42, buffer_size=100)
            assert isinstance(shuffle_step, ShuffleStep)
            assert isinstance(shuffle_step.output, OutputIterableDataset)
            assert list(shuffle_step.output["out1"]) == [3, 2, 1]
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": [1, 2, 3]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            shuffle_step = step.shuffle(seed=42)
            assert not shuffle_step._resumed
            assert isinstance(shuffle_step.output, OutputIterableDataset)
            assert list(shuffle_step.output["out1"]) == [3, 2, 1]

    def test_dataset(self, create_datadreamer, create_test_step: Callable[..., Step]):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output({"out1": [1, 2, 3]})
            shuffle_step = step.shuffle(seed=42, lazy=False)
            assert isinstance(shuffle_step, ShuffleStep)
            assert (
                shuffle_step.output.dataset._indices is None  # type:ignore[union-attr]
            )
            assert shuffle_step.output["out1"][0] == 3
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            shuffle_step = step.shuffle(seed=42, lazy=False)
            assert shuffle_step._resumed
            assert (
                shuffle_step.output.dataset._indices is None  # type:ignore[union-attr]
            )
            assert shuffle_step.output["out1"][0] == 3

    def test_iterable_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": [1, 2, 3]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            shuffle_step = step.shuffle(seed=42, lazy=False)
            assert isinstance(shuffle_step, ShuffleStep)
            assert (
                shuffle_step.output.dataset._indices is None  # type:ignore[union-attr]
            )
            assert shuffle_step.output["out1"][0] == 3
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": [1, 2, 3]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            shuffle_step = step.shuffle(seed=42, lazy=False)
            assert shuffle_step._resumed
            assert (
                shuffle_step.output.dataset._indices is None  # type:ignore[union-attr]
            )
            assert shuffle_step.output["out1"][0] == 3


class TestConcat:
    def test_concat_invalid_args(self):
        with pytest.raises(ValueError):
            concat()
        with pytest.raises(TypeError):
            concat(5)  # type: ignore[arg-type]

    def test_concat_lazy(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step._set_output(
                {
                    "out1": [
                        step.pickle(set(["a"])),
                        step.pickle(set(["b"])),
                        step.pickle(set(["c"])),
                    ]
                }
            )
            iterable_step = create_test_step(
                name="my-step-2", inputs=None, output_names=["out1"]
            )
            iterable_step._set_output(
                LazyRows(
                    Dataset.from_dict(
                        {
                            "out1": [
                                iterable_step.pickle(set(["d"])),
                                iterable_step.pickle(set(["e"])),
                                iterable_step.pickle(set(["f"])),
                                iterable_step.pickle(set(["g"])),
                            ]
                        }
                    ).to_iterable_dataset(),
                    total_num_rows=4,
                )
            )
            concat_step = concat(step, iterable_step)
            assert concat_step.name == "concat(my-step-1, my-step-2)"
            isinstance(concat_step, ConcatStep)
            isinstance(concat_step.output, OutputIterableDataset)
            assert concat_step._pickled is True
            assert concat_step.output.num_rows == 7
            assert list(concat_step.output["out1"])[0] == set(["a"])
            assert list(concat_step.output["out1"])[-1] == set(["g"])

    def test_concat(self, create_datadreamer, create_test_step: Callable[..., Step]):
        with create_datadreamer():
            iterable_step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            iterable_step._set_output(
                LazyRows(
                    Dataset.from_dict(
                        {
                            "out1": [
                                iterable_step.pickle(set(["a"])),
                                iterable_step.pickle(set(["b"])),
                                iterable_step.pickle(set(["c"])),
                                iterable_step.pickle(set(["d"])),
                            ]
                        }
                    ).to_iterable_dataset(),
                    total_num_rows=4,
                )
            )
            step = create_test_step(
                name="my-step-2", inputs=None, output_names=["out1"]
            )
            step._set_output(
                {
                    "out1": [
                        step.pickle(set(["e"])),
                        step.pickle(set(["f"])),
                        step.pickle(set(["g"])),
                    ]
                }
            )
            concat_step = concat(iterable_step, step, lazy=False)
            assert concat_step.name == "concat(my-step-1, my-step-2)"
            isinstance(concat_step, ConcatStep)
            isinstance(concat_step.output, OutputDataset)
            assert concat_step._pickled is True
            assert len(concat_step.output) == 7  # type:ignore[arg-type]
            assert concat_step.output["out1"][0] == set(["a"])
            assert concat_step.output["out1"][-1] == set(["g"])

    def test_concat_lazy_with_no_num_rows(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step._set_output(
                {
                    "out1": [
                        step.pickle(set(["a"])),
                        step.pickle(set(["b"])),
                        step.pickle(set(["c"])),
                    ]
                }
            )
            iterable_step = create_test_step(
                name="my-step-2", inputs=None, output_names=["out1"]
            )
            with pytest.warns(UserWarning):
                iterable_step._set_output(
                    LazyRows(
                        Dataset.from_dict(
                            {
                                "out1": [
                                    iterable_step.pickle(set(["d"])),
                                    iterable_step.pickle(set(["e"])),
                                    iterable_step.pickle(set(["f"])),
                                    iterable_step.pickle(set(["g"])),
                                ]
                            }
                        ).to_iterable_dataset(),
                    )
                )
            with pytest.warns(UserWarning):
                concat_step = concat(step, iterable_step)
            isinstance(concat_step.output, OutputIterableDataset)
            assert concat_step.output.num_rows is None
            assert list(concat_step.output["out1"])[-1] == set(["g"])

    def test_concat_with_different_column_names(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step_1 = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step_1._set_output({"out1": ["a", "b", "c"]})
            step_2 = create_test_step(
                name="my-step-2", inputs=None, output_names=["out2"]
            )
            step_2._set_output({"out2": ["d", "e", "f"]})
            concat_step = concat(step_1, step_2, lazy=False)
            assert len(concat_step.output) == 6  # type:ignore[arg-type]
            assert concat_step.output[0] == {"out1": "a", "out2": None}


class TestZipped:
    def test_zipped_invalid_args(self):
        with pytest.raises(ValueError):
            zipped()
        with pytest.raises(TypeError):
            zipped(5)  # type: ignore[arg-type]

    def test_zipped_lazy(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step._set_output({"out1": ["a", "b", "c"]})
            iterable_step = create_test_step(
                name="my-step-2", inputs=None, output_names=["out2"]
            )
            iterable_step._set_output(
                LazyRows(
                    Dataset.from_dict(
                        {
                            "out2": [
                                iterable_step.pickle(set(["d"])),
                                iterable_step.pickle(set(["e"])),
                                iterable_step.pickle(set(["f"])),
                            ]
                        }
                    ).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            zipped_step = zipped(step, iterable_step)
            assert zipped_step.name == "zipped(my-step-1, my-step-2)"
            isinstance(zipped_step, ZippedStep)
            isinstance(zipped_step.output, OutputIterableDataset)
            assert zipped_step._pickled is True
            assert zipped_step.output.num_rows == 3
            assert list(zipped_step.output)[0] == {"out1": "a", "out2": set(["d"])}

    def test_zipped(self, create_datadreamer, create_test_step: Callable[..., Step]):
        with create_datadreamer():
            iterable_step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            iterable_step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": ["a", "b", "c"]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            step = create_test_step(
                name="my-step-2", inputs=None, output_names=["out2"]
            )
            step._set_output(
                {
                    "out2": [
                        step.pickle(set(["d"])),
                        step.pickle(set(["e"])),
                        step.pickle(set(["f"])),
                    ]
                }
            )
            zipped_step = zipped(iterable_step, step, lazy=False)
            assert zipped_step.name == "zipped(my-step-1, my-step-2)"
            isinstance(zipped_step, ZippedStep)
            isinstance(zipped_step.output, OutputDataset)
            assert zipped_step._pickled is True
            assert len(zipped_step.output) == 3  # type:ignore[arg-type]
            assert zipped_step.output[0] == {"out1": "a", "out2": set(["d"])}

    def test_zipped_lazy_with_no_num_rows(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step._set_output({"out1": ["a", "b", "c"]})
            iterable_step = create_test_step(
                name="my-step-2", inputs=None, output_names=["out2"]
            )
            with pytest.warns(UserWarning):
                iterable_step._set_output(
                    LazyRows(
                        Dataset.from_dict(
                            {
                                "out2": [
                                    iterable_step.pickle(set(["d"])),
                                    iterable_step.pickle(set(["e"])),
                                    iterable_step.pickle(set(["f"])),
                                ]
                            }
                        ).to_iterable_dataset(),
                    )
                )
            with pytest.warns(UserWarning):
                zipped_step = zipped(step, iterable_step)
            isinstance(zipped_step.output, OutputIterableDataset)
            assert zipped_step.output.num_rows is None
            assert list(zipped_step.output)[0] == {"out1": "a", "out2": set(["d"])}

    def test_zipped_with_different_lengths_error(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step_1 = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step_1._set_output({"out1": ["a", "b", "c"]})
            step_2 = create_test_step(
                name="my-step-2", inputs=None, output_names=["out2"]
            )
            step_2._set_output({"out2": ["d", "e", "f", "g"]})

            with pytest.raises(StepOutputTypeError):
                zipped_step = zipped(step_1, step_2, lazy=False)

            zipped_step = zipped(step_1, step_2, lazy=True)
            assert list(zipped_step.output)[-1] == {"out1": None, "out2": "g"}
            assert zipped_step.output.num_rows == 4


class TestSelect:
    def test_select_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step._set_output({"out1": [1, 2, 3]})
            select_step = step.select([0, 2])
            isinstance(select_step, SelectStep)
            isinstance(select_step.output, OutputDataset)
            assert len(select_step.output) == 2  # type:ignore[arg-type]
            assert list(select_step.output["out1"]) == [1, 3]

    def test_select_iterable_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": [1, 2, 3]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            select_step = step.select([0, 2])
            isinstance(select_step, SelectStep)
            isinstance(select_step.output, OutputDataset)
            assert len(select_step.output) == 2  # type:ignore[arg-type]
            assert list(select_step.output["out1"]) == [1, 3]


class TestSelectColumns:
    def test_select_columns_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1", "out2"]
            )
            step._set_output({"out1": [1, 2, 3], "out2": [4, 5, 6]})
            select_columns_step = step.select_columns(["out2"])
            isinstance(select_columns_step, SelectColumnsStep)
            isinstance(select_columns_step.output, OutputDataset)
            assert len(select_columns_step.output) == 3  # type:ignore[arg-type]
            assert set(select_columns_step.output.column_names) == set(["out2"])

    def test_select_columns_iterable_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1", "out2"]
            )
            step._set_output(
                LazyRows(
                    Dataset.from_dict(
                        {"out1": [1, 2, 3], "out2": [4, 5, 6]}
                    ).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            select_columns_step = step.select_columns(["out2"])
            isinstance(select_columns_step, SelectColumnsStep)
            isinstance(select_columns_step.output, OutputIterableDataset)
            assert select_columns_step.output.num_rows == 3
            assert set(select_columns_step.output.column_names) == set(["out2"])


class TestShard:
    def test_shard_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step._set_output({"out1": [1, 2, 3, 4]})
            shard_step = step.shard(num_shards=2, index=1)
            isinstance(shard_step, SelectColumnsStep)
            isinstance(shard_step.output, OutputDataset)
            assert shard_step.output.num_rows == 2
            assert list(shard_step.output["out1"]) == [2, 4]

    def test_shard_iterable_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": [1, 2, 3, 4]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            shard_step = step.shard(num_shards=2, index=1)
            isinstance(shard_step, SelectColumnsStep)
            isinstance(shard_step.output, OutputDataset)
            assert shard_step.output.num_rows == 2
            assert list(shard_step.output["out1"]) == [2, 4]

    def test_shard_dataset_contiguous(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step._set_output({"out1": [1, 2, 3, 4]})
            shard_step = step.shard(
                num_shards=2, index=1, contiguous=True, writer_batch_size=1000
            )
            isinstance(shard_step, SelectColumnsStep)
            isinstance(shard_step.output, OutputDataset)
            assert shard_step.output.num_rows == 2
            assert list(shard_step.output["out1"]) == [3, 4]
