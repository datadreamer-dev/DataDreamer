import os
from typing import Callable

from datasets import Dataset

from ... import DataDreamer
from ...datasets import OutputDataset
from ...steps import LazyRows, Step
from ...steps.step import SaveStep


class TestSave:
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
            save_step = step.save("save-step", writer_batch_size=1000, num_proc=2)
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
            save_step = step.save("save-step", writer_batch_size=1000, num_proc=2)
            assert save_step._resumed
            assert save_step.output._pickled
            assert save_step.output["out1"][0] == set(["a"])

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
