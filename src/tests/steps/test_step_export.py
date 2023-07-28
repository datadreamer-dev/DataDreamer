import os
from typing import Callable

import pytest

from datasets import Dataset

from ... import DataDreamer
from ...steps import LazyRows, Step


class TestError:
    def test_export_dataset_to_dict_two_splits_wrong_proportions(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            with pytest.raises(ValueError):
                step.export_to_dict(validation_size=0.2, test_size=0.7)

    def test_export_dataset_to_dict_two_splits_by_row_wrong_proportions(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            with pytest.raises(ValueError):
                step.export_to_dict(validation_size=2, test_size=7)


class TestPickle:
    def test_export_dataset_to_dict_pickled(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                Dataset.from_dict(
                    {"out1": [step.pickle(set([i])) for i in range(1, 11)]}
                )
            )
            export = step.export_to_dict()
            assert isinstance(export, dict)
            assert step._pickled
            assert "out1" in export
            assert len(export["out1"]) == 10
            assert export["out1"][0] == set([1])

    def test_export_dataset_to_list_pickled(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                Dataset.from_dict(
                    {"out1": [step.pickle(set([i])) for i in range(1, 11)]}
                )
            )
            export = step.export_to_list()
            assert isinstance(export, list)
            assert step._pickled
            assert len(export) == 10
            assert export[0]["out1"] == set([1])

    def test_export_dataset_to_hf_dataset_pickled(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                Dataset.from_dict(
                    {"out1": [step.pickle(set([i])) for i in range(1, 11)]}
                )
            )
            export = step.export_to_hf_dataset(
                os.path.join(step._output_folder_path, "export")
            )
            assert isinstance(export, Dataset)
            assert step._pickled
            assert len(export) == 10
            assert export["out1"][0] == set([1])


class TestExport:
    def test_export_dataset_to_dict_no_split(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export = step.export_to_dict()
            assert isinstance(export, dict)
            assert "out1" in export
            assert len(export["out1"]) == 10
            assert export["out1"][0] == 1

    def test_export_dataset_to_dict_single_train_split(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export = step.export_to_dict(train_size=1.0)
            assert isinstance(export, dict)
            assert "out1" in export
            assert len(export["out1"]) == 10

    def test_export_dataset_to_dict_single_other_split(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export = step.export_to_dict(validation_size=1.0)
            assert isinstance(export, dict)
            assert "out1" in export
            assert len(export["out1"]) == 10

    def test_export_dataset_to_dict_two_splits(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export = step.export_to_dict(validation_size=0.3, test_size=0.7)
            assert isinstance(export, dict)
            assert "validation" in export
            assert "test" in export
            assert len(export["validation"]["out1"]) == 3
            assert len(export["test"]["out1"]) == 7

    def test_export_dataset_to_dict_two_splits_by_row(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export = step.export_to_dict(validation_size=3, test_size=7)
            assert isinstance(export, dict)
            assert "validation" in export
            assert "test" in export
            assert len(export["validation"]["out1"]) == 3
            assert len(export["test"]["out1"]) == 7

    def test_export_dataset_to_dict_three_splits(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 101)}))
            export = step.export_to_dict(
                train_size=0.7, validation_size=0.1, test_size=0.2
            )
            assert isinstance(export, dict)
            assert "train" in export
            assert "validation" in export
            assert "test" in export
            assert len(export["train"]["out1"]) == 70
            assert len(export["validation"]["out1"]) == 10
            assert len(export["test"]["out1"]) == 20

    def test_export_iterable_dataset_to_dict_three_splits(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(
                LazyRows(
                    Dataset.from_dict({"out1": range(1, 101)}).to_iterable_dataset(),
                    total_num_rows=100,
                )
            )
            export = step.export_to_dict(
                train_size=0.7, validation_size=0.1, test_size=0.2
            )
            assert isinstance(export, dict)
            assert "train" in export
            assert "validation" in export
            assert "test" in export
            assert len(export["train"]["out1"]) == 70
            assert len(export["validation"]["out1"]) == 10
            assert len(export["test"]["out1"]) == 20
