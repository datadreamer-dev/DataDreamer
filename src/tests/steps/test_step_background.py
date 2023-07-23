import os

import pytest

from ... import DataDreamer
from ...datasets import OutputDataset, OutputIterableDataset
from ...errors import StepOutputError
from ...steps import LazyRows, Step, wait


class TestBackground:
    def test_step_without_background_runs_in_same_process(self, create_datadreamer):
        class TestStep(Step):
            def setup(self):
                self.register_output("out1")

            def run(self):
                pid = self.pickle(set([os.getpid()]))
                return [pid, pid, pid]

        with create_datadreamer():
            step = TestStep(name="my-step", background=False)
            assert set(step.output.column_names) == set(["out1"])
            assert isinstance(step.output, OutputDataset)
            assert step.output._pickled is True
            assert len(step.output["out1"]) == 3
            step_path = os.path.join(DataDreamer.ctx.output_folder_path, "my-step")
            assert os.path.isfile(
                os.path.join(step_path, "dataset", "dataset_info.json")
            )
            assert step.output["out1"][0] == set([os.getpid()])

    def test_step_with_background_runs_in_background_process(
        self, create_datadreamer, caplog
    ):
        class TestStep(Step):
            def setup(self):
                self.register_output("out1")

            def run(self):
                pid = self.pickle(set([os.getpid()]))
                return [pid, pid, pid]

        with create_datadreamer():
            step = TestStep(name="my-step", background=True)
            with pytest.raises(StepOutputError):
                step.output
            wait(step)
            assert set(step.output.column_names) == set(["out1"])
            assert isinstance(step.output, OutputDataset)
            assert step.output._pickled is True
            assert len(step.output["out1"]) == 3
            step_path = os.path.join(DataDreamer.ctx.output_folder_path, "my-step")
            assert os.path.isfile(
                os.path.join(step_path, "dataset", "dataset_info.json")
            )
            assert step.output["out1"][0] != set([os.getpid()])
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert any(["Step 'my-step' is running. ⏳" in log for log in logs])
            assert any(
                [
                    "Step 'my-step' finished and is saved to disk. 🎉" in log
                    for log in logs
                ]
            )
            assert len(logs) == 3

    def test_generator_step_with_background_runs_in_background_process(
        self, create_datadreamer, caplog
    ):
        class TestStep(Step):
            def setup(self):
                self.register_output("out1")

            def run(self):
                def data_generator():
                    for _ in range(3):
                        yield self.pickle(set([os.getpid()]))

                return LazyRows(data_generator, total_num_rows=3)

        with create_datadreamer():
            step = TestStep(name="my-step", background=True)
            with pytest.raises(StepOutputError):
                step.output
            wait(step)
            assert set(step.output.column_names) == set(["out1"])
            assert isinstance(step.output, OutputIterableDataset)
            assert step.output._pickled is True
            assert step.output["out1"].num_rows == 3
            step_path = os.path.join(DataDreamer.ctx.output_folder_path, "my-step")
            assert not os.path.exists(
                os.path.join(step_path, "dataset", "dataset_info.json")
            )
            assert list(step.output["out1"])[0] != set([os.getpid()])
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert any(["Step 'my-step' will run lazily. 🥱" in log for log in logs])
            assert len(logs) == 3
