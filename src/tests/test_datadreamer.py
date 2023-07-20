import json
import logging
import os
from typing import Callable

import pytest

from datasets import Dataset

from .. import DataDreamer, __version__
from ..datasets import OutputDataset
from ..errors import StepOutputError
from ..steps import LazyRows, Step, TraceInfoType


class TestErrors:
    def test_path_is_to_file(self):
        with pytest.raises(ValueError):
            with DataDreamer("./README.md"):
                pass

    def test_nested(self, create_datadreamer):
        with pytest.raises(RuntimeError):
            with create_datadreamer():
                with create_datadreamer():
                    pass

    def test_steps_with_same_name(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            create_test_step(name="my-step", inputs=None, output_names=["out1"])
            with pytest.raises(ValueError):
                create_test_step(name="my-step", inputs=None, output_names=["out1"])
            with pytest.raises(ValueError):
                create_test_step(name="my-step:::", inputs=None, output_names=["out1"])


class TestFunctionality:
    def test_logging(self, create_datadreamer, caplog):
        with create_datadreamer():
            pass
        log_dates = [
            rec.asctime if hasattr(rec, "asctime") else False for rec in caplog.records
        ]
        logs = [rec.message for rec in caplog.records]
        caplog.clear()
        assert logs[0].startswith("Intialized. ✨ Dreaming to folder: ")
        assert logs[1].startswith("Done. 🥂 Results in folder:")
        assert not any(log_dates)

        with create_datadreamer(verbose=False):
            pass
        logs = [rec.message for rec in caplog.records]
        caplog.clear()
        assert len(logs) == 0

        with create_datadreamer(log_date=True):
            pass
        log_dates = [
            rec.asctime if hasattr(rec, "asctime") else False for rec in caplog.records
        ]
        logs = [rec.message for rec in caplog.records]
        caplog.clear()
        assert logs[0].startswith("Intialized. ✨ Dreaming to folder: ")
        assert logs[1].startswith("Done. 🥂 Results in folder:")
        assert all(log_dates)

    def test_step_logging(
        self, create_datadreamer, create_test_step: Callable[..., Step], caplog
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step.logger.info("Log from step.")
        logs = [rec.message for rec in caplog.records]
        caplog.clear()
        assert logs[2] != "Log from step."

        with create_datadreamer():
            step = create_test_step(
                name="my-step", inputs=None, output_names=["out1"], verbose=True
            )
            step.logger.info("Log from step.")
        logs = [rec.message for rec in caplog.records]
        caplog.clear()
        assert logs[2] == "Log from step."

        with create_datadreamer():
            step = create_test_step(
                name="my-step",
                inputs=None,
                output_names=["out1"],
                verbose=True,
                log_level=logging.WARNING,
            )
            step.logger.info("Log from step.")
        logs = [rec.message for rec in caplog.records]
        caplog.clear()
        assert logs[2] != "Log from step."

        with create_datadreamer(verbose=True):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step.logger.info("Log from step.")
        logs = [rec.message for rec in caplog.records]
        caplog.clear()
        assert logs[2] != "Log from step."

        with create_datadreamer(log_level=logging.WARNING):
            step = create_test_step(
                name="my-step", inputs=None, output_names=["out1"], verbose=True
            )
            step.logger.info("Log from step.")
        logs = [rec.message for rec in caplog.records]
        caplog.clear()
        assert len(logs) < 3

        with create_datadreamer(verbose=True, log_level=logging.INFO):
            step = create_test_step(
                name="my-step", inputs=None, output_names=["out1"], verbose=True
            )
            step.logger.info("Log from step.")
        logs = [rec.message for rec in caplog.records]
        caplog.clear()
        assert logs[2] == "Log from step."

    def test_creates_folder(self, create_datadreamer):
        with create_datadreamer():
            assert os.path.exists(DataDreamer.ctx.output_folder_path)
            assert os.path.isdir(DataDreamer.ctx.output_folder_path)
            assert DataDreamer.ctx.steps == []

    def test_ctx_clears(self, create_datadreamer):
        with create_datadreamer():
            DataDreamer.ctx.foo = 5
            assert hasattr(DataDreamer.ctx, "foo")
            assert DataDreamer.ctx.foo == 5

        with create_datadreamer():
            assert not hasattr(DataDreamer.ctx, "foo")

    def test_steps_added_to_ctx(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            assert DataDreamer.ctx.steps == []
            step_1 = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"]
            )
            step_2 = create_test_step(
                name="my-step-2", inputs=None, output_names=["out1"]
            )
            assert DataDreamer.ctx.steps == [step_1, step_2]

        with create_datadreamer():
            assert DataDreamer.ctx.steps == []

    def test_step_creates_folder_and_saves_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output({"out1": ["a", "b", "c"]})
            assert step.fingerprint == "2d5db4b5ff337633"
            del step
            assert os.path.isdir(
                os.path.join(DataDreamer.ctx.output_folder_path, "my-step")
            )
            assert os.path.isfile(
                os.path.join(DataDreamer.ctx.output_folder_path, "my-step", "step.json")
            )
            with open(
                os.path.join(
                    DataDreamer.ctx.output_folder_path, "my-step", "step.json"
                ),
                "r",
            ) as f:
                metadata = json.load(f)
                assert metadata["__version__"] == __version__
                assert metadata["fingerprint"] == "2d5db4b5ff337633"
                assert metadata["pickled"] is False
            assert os.path.isdir(
                os.path.join(DataDreamer.ctx.output_folder_path, "my-step", "dataset")
            )
            assert os.path.isfile(
                os.path.join(
                    DataDreamer.ctx.output_folder_path,
                    "my-step",
                    "dataset",
                    "dataset_info.json",
                )
            )
            resume_path = os.path.basename(DataDreamer.ctx.output_folder_path)

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            assert set(step.output.column_names) == set(["out1"])
            assert isinstance(step.output, OutputDataset)
            assert step.output.step == step
            assert step.output._pickled is False
            assert len(step.output["out1"]) == 3
            assert step.output["out1"][0] == "a"
            assert set(step.output[0].keys()) == set(step.output.column_names)
            assert list(step.output["out1"]) == ["a", "b", "c"]
            del step

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out2"])
            with pytest.raises(StepOutputError):
                step.output
            assert os.path.isdir(
                os.path.join(DataDreamer.ctx.output_folder_path, "my-step")
            )
            assert not os.path.exists(
                os.path.join(DataDreamer.ctx.output_folder_path, "my-step", "step.json")
            )
            assert not os.path.exists(
                os.path.join(
                    DataDreamer.ctx.output_folder_path,
                    "my-step",
                    "dataset",
                    "dataset_info.json",
                )
            )

    def test_step_does_not_save_iterable_dataset(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": ["a", "b", "c"]}).to_iterable_dataset(),
                    total_num_rows=3,
                )
            )
            assert step.fingerprint == "2d5db4b5ff337633"
            del step
            assert os.path.isdir(
                os.path.join(DataDreamer.ctx.output_folder_path, "my-step")
            )
            assert not os.path.exists(
                os.path.join(DataDreamer.ctx.output_folder_path, "my-step", "step.json")
            )
            assert not os.path.exists(
                os.path.join(DataDreamer.ctx.output_folder_path, "my-step", "dataset")
            )
            resume_path = os.path.basename(DataDreamer.ctx.output_folder_path)

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            with pytest.raises(StepOutputError):
                assert set(step.output.column_names) == set(["out1"])

    def test_saves_pickled_dataset(
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
            del step
            resume_path = os.path.basename(DataDreamer.ctx.output_folder_path)

        with create_datadreamer(resume_path):
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            assert set(step.output.column_names) == set(["out1"])
            assert isinstance(step.output, OutputDataset)
            assert step.output.step == step
            assert step.output._pickled is True
            assert len(step.output["out1"]) == 3
            assert step.output["out1"][0] == set(["a"])
            assert set(step.output[0].keys()) == set(step.output.column_names)
            assert list(step.output["out1"]) == [set(["a"]), set(["b"]), set(["c"])]

    def test_step_creates_run_output_folder(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            run_output_folder_path = step.get_run_output_folder_path()
            assert os.path.isdir(run_output_folder_path)
            assert os.path.join(DataDreamer.ctx.output_folder_path, "run_output")

    def test_trace_info_propogates(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        def setup_1(self):
            self.register_trace_info(TraceInfoType.CITATION, "citation")
            self.register_trace_info(TraceInfoType.URL, "http://example.com")

        def setup_2(self):
            self.register_input("out1")
            self.register_trace_info(TraceInfoType.CITATION, "citation2")
            self.register_trace_info(TraceInfoType.URL, "http://example2.com")

        with create_datadreamer():
            step_1 = create_test_step(
                name="my-step-1", inputs=None, output_names=["out1"], setup=setup_1
            )
            step_1._set_output({"out1": ["a", "b", "c"]})
            step_2 = create_test_step(
                name="my-step-2",
                inputs={"out1": step_1.output["out1"]},
                output_names=["out1"],
                setup=setup_2,
            )
            assert step_1.trace_info == {
                "my-step-1": {
                    "Citation Information": ["citation"],
                    "URL": ["http://example.com"],
                }
            }
            assert step_2.trace_info == {
                "my-step-1": {
                    "Citation Information": ["citation"],
                    "URL": ["http://example.com"],
                },
                "my-step-2": {
                    "Citation Information": ["citation2"],
                    "URL": ["http://example2.com"],
                },
            }
