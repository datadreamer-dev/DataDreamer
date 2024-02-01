import json
import os
from time import sleep

import pytest
from flaky import flaky

from ... import DataDreamer
from ...datasets import OutputDataset, OutputIterableDataset
from ...errors import StepOutputError
from ...steps import LazyRows, Step, concurrent, wait


class TestErrors:
    def test_wait_invalid_args(self):
        with pytest.raises(TypeError):
            wait(5)  # type: ignore[arg-type]

    def test_concurrent_invalid_args(self):
        with pytest.raises(TypeError):
            concurrent(5)  # type: ignore[arg-type]


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
            step_path = os.path.join(DataDreamer.get_output_folder_path(), "my-step")
            assert os.path.isfile(
                os.path.join(step_path, "_dataset", "dataset_info.json")
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
            assert step.background
            with pytest.raises(StepOutputError):
                step.output  # noqa: B018
            wait(step)
            assert set(step.output.column_names) == set(["out1"])
            assert isinstance(step.output, OutputDataset)
            assert step.output._pickled is True
            assert len(step.output["out1"]) == 3
            step_path = os.path.join(DataDreamer.get_output_folder_path(), "my-step")
            assert os.path.isfile(
                os.path.join(step_path, "_dataset", "dataset_info.json")
            )
            assert step.output["out1"][0] != set([os.getpid()])
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert any(
                [
                    "Step 'my-step' is running in the background. ⏳" in log
                    for log in logs
                ]
            )
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
            assert step.background
            with pytest.raises(StepOutputError):
                step.output  # noqa: B018
            wait(step)
            assert set(step.output.column_names) == set(["out1"])
            assert isinstance(step.output, OutputIterableDataset)
            assert step.output._pickled is True
            assert step.output["out1"].num_rows == 3
            step_path = os.path.join(DataDreamer.get_output_folder_path(), "my-step")
            assert not os.path.exists(
                os.path.join(step_path, "_dataset", "dataset_info.json")
            )
            assert list(step.output["out1"])[0] != set([os.getpid()])
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert any(["Step 'my-step' will run lazily. 🥱" in log for log in logs])
            assert len(logs) == 3

    def test_step_operation_on_backgrounded_step(self, create_datadreamer, caplog):
        class TestStep(Step):
            def setup(self):
                self.register_output("out1")

            def run(self):
                return [1, 2, 3]

        with create_datadreamer():
            step = TestStep(name="my-step", background=True)
            assert step.background
            shuffle_step = step.shuffle(seed=42)
            assert isinstance(shuffle_step.output, OutputDataset)
            assert shuffle_step.output["out1"][0] == 3

    def test_concurrent_step_operations(self, create_datadreamer, caplog):
        class TestStep(Step):
            def setup(self):
                self.register_output("out1")

            def run(self):
                return [1, 2, 3]

        with create_datadreamer():

            def create_step_and_shuffle():
                step = TestStep(name="my-step", background=True)
                assert step.background
                shuffle_step = step.shuffle(seed=42)
                return shuffle_step

            shuffle_step_1, shuffle_step_2 = concurrent(
                create_step_and_shuffle, create_step_and_shuffle
            )
            assert isinstance(shuffle_step_1.output, OutputDataset)
            assert isinstance(shuffle_step_2.output, OutputDataset)
            assert shuffle_step_1.output["out1"][0] == 3
            assert shuffle_step_2.output["out1"][0] == 3

    @flaky(max_runs=3)
    def test_can_get_background_progress_in_foreground(self, create_datadreamer):
        class TestStep(Step):
            def setup(self):
                self.register_output("out1")

            def run(self):
                def data_generator():
                    for i in range(1000):
                        yield i

                return LazyRows(data_generator, total_num_rows=1000)

        with create_datadreamer():
            step = TestStep(name="my-step", background=True, progress_interval=0)
            assert step.background
            wait(step)
            assert next(iter(step.output)) == {"out1": 0}
            sleep(1)  # Wait for progress file to write
            assert step.progress > 0.0  # type:ignore[operator]
            assert step.progress < 1.0  # type:ignore[operator]
            step_path = os.path.join(DataDreamer.get_output_folder_path(), "my-step")
            assert os.path.isdir(step_path)
            assert os.path.isfile(os.path.join(step_path, ".background_progress"))
            with open(os.path.join(step_path, ".background_progress"), "r") as f:
                progress_data = json.load(f)
                assert step.progress == progress_data["progress"]

    @flaky(max_runs=3)
    def test_can_get_background_progress_rows_in_foreground(self, create_datadreamer):
        class TestStep(Step):
            def setup(self):
                self.register_output("out1")

            def run(self):
                def data_generator():
                    for i in range(1000):
                        yield i

                return LazyRows(data_generator, auto_progress=False)

        with create_datadreamer():
            step = TestStep(name="my-step", background=True, progress_interval=0)
            assert step.background
            wait(step)
            assert next(iter(step.output)) == {"out1": 0}
            sleep(1)  # Wait for progress file to write
            assert "row(s)" in step._Step__get_progress_string()  # type: ignore[attr-defined]
            assert "row(s)" in str(step)
            assert step._Step__progress_rows > 0  # type: ignore[attr-defined]
            step_path = os.path.join(DataDreamer.get_output_folder_path(), "my-step")
            assert os.path.isdir(step_path)
            assert os.path.isfile(os.path.join(step_path, ".background_progress"))
            with open(os.path.join(step_path, ".background_progress"), "r") as f:
                progress_data = json.load(f)
                assert step._Step__progress_rows == progress_data["progress_rows"]  # type: ignore[attr-defined]
