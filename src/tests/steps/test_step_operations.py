import os
from typing import Callable

from datasets import Dataset

from ... import DataDreamer
from ...datasets import OutputDataset, OutputIterableDataset
from ...steps import LazyRows, Step
from ...steps.step import MapStep, SaveStep


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
                DataDreamer.ctx.output_folder_path, "my-step-save"
            )
            save_step_2_path = os.path.join(
                DataDreamer.ctx.output_folder_path, "my-step-save-2"
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
                DataDreamer.ctx.output_folder_path, "my-step-save"
            )
            assert os.path.isdir(save_step_path)
            assert os.path.isfile(
                os.path.join(
                    DataDreamer.ctx.output_folder_path,
                    "my-step-save",
                    "dataset",
                    "dataset_info.json",
                )
            )
            resume_path = os.path.basename(DataDreamer.ctx.output_folder_path)

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
            step_path = os.path.join(DataDreamer.ctx.output_folder_path, "my-step")
            assert os.path.isdir(os.path.join(step_path, "cache", "generator"))
            assert type(save_step).__name__ == "SaveStep"
            assert isinstance(save_step, SaveStep)
            assert isinstance(save_step.output, OutputDataset)
            save_step_path = os.path.join(
                DataDreamer.ctx.output_folder_path, "save-step"
            )
            assert os.path.isdir(save_step_path)
            assert os.path.isfile(
                os.path.join(
                    DataDreamer.ctx.output_folder_path,
                    "save-step",
                    "dataset",
                    "dataset_info.json",
                )
            )
            resume_path = os.path.basename(DataDreamer.ctx.output_folder_path)

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
                DataDreamer.ctx.output_folder_path, "my-step-save"
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
                DataDreamer.ctx.output_folder_path, "save-step"
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
            assert isinstance(map_step.output, OutputDataset)
            assert map_step.output["out1"][2] == 6
            resume_path = os.path.basename(DataDreamer.ctx.output_folder_path)

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            map_step = step.map(lambda row: {"out1": row["out1"] * 2})
            assert map_step._resumed
            assert map_step.output["out1"][2] == 6

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
