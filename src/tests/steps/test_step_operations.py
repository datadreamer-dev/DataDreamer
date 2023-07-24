import os
from typing import Callable

import pytest

from datasets import Dataset

from ... import DataDreamer
from ...datasets import OutputDataset, OutputIterableDataset
from ...errors import StepOutputTypeError
from ...steps import LazyRows, Step, concat, zipped
from ...steps.step import ConcatStep, MapStep, SaveStep, ShuffleStep, ZippedStep


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
            map_step = step.map(lambda row: {"out1": row["out1"] * 2})
            assert isinstance(map_step, MapStep)
            assert isinstance(map_step.output, OutputIterableDataset)
            assert list(map_step.output["out1"])[2] == 6
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
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
            assert list(map_step.output["out1"])[2] == set(["c", 2])

    def test_map_add_remove_columns(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output({"out1": [1, 2, 3]})
            map_step = step.map(
                lambda row: {"out1": row["out1"], "out2": row["out1"] * 2}, lazy=False
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
                step.map(lambda row: {"out1": row["out1"]}, lazy=False)

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
                list(step.map(lambda row: {"out1": row["out1"]}).output)


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
            assert shuffle_step.output.dataset._indices is not None
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
            assert shuffle_step.output.dataset._indices is None
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
            assert shuffle_step.output.dataset._indices is None
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
            assert len(concat_step.output) == 7
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
            assert len(zipped_step.output) == 3
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
