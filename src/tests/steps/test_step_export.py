import os
from typing import Callable, cast

import pytest
from datasets import Dataset, DatasetDict
from datasets.fingerprint import Hasher
from dill.source import getsource

from ...steps import DataCardType, LazyRows, Step


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
            export_path = os.path.join(cast(str, step._output_folder_path), "export")
            export = step.export_to_hf_dataset(export_path)
            assert isinstance(export, Dataset)
            assert step._pickled
            assert len(export) == 10
            assert export["out1"][0] == set([1])


class TestExportToDict:
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


class TestExportToList:
    def test_export_dataset_to_list_single_train_split(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export = step.export_to_list(train_size=1.0)
            assert isinstance(export, list)
            assert len(export) == 10

    def test_export_dataset_to_list_two_splits(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export = step.export_to_list(validation_size=0.3, test_size=0.7)
            assert isinstance(export, dict)
            assert "validation" in export
            assert "test" in export
            assert len(export["validation"]) == 3
            assert len(export["test"]) == 7
            assert isinstance(export["validation"], list)
            assert isinstance(export["test"], list)


class TestExportToJSON:
    def test_export_dataset_to_list_single_train_split(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export_path = os.path.join(
                cast(str, step._output_folder_path), "export.json"
            )
            export = step.export_to_json(export_path, train_size=1.0)
            assert export == export_path
            assert os.path.isfile(export_path)

    def test_export_dataset_to_list_two_splits(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export_path = os.path.join(
                cast(str, step._output_folder_path), "export.json"
            )
            export = step.export_to_json(
                export_path, validation_size=0.3, test_size=0.7
            )
            assert isinstance(export, dict)
            assert set(export.keys()) == set(["validation", "test"])
            assert export["validation"] == os.path.join(
                cast(str, step._output_folder_path), "export.val.json"
            )
            assert export["test"] == os.path.join(
                cast(str, step._output_folder_path), "export.test.json"
            )
            for v in export.values():
                assert os.path.isfile(v)


class TestExportToCSV:
    def test_export_dataset_to_list_single_train_split(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export_path = os.path.join(
                cast(str, step._output_folder_path), "export.csv"
            )
            export = step.export_to_csv(export_path, train_size=1.0)
            assert export == export_path
            assert os.path.isfile(export_path)

    def test_export_dataset_to_list_two_splits(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export_path = os.path.join(
                cast(str, step._output_folder_path), "export.csv"
            )
            export = step.export_to_csv(export_path, validation_size=0.3, test_size=0.7)
            assert isinstance(export, dict)
            assert set(export.keys()) == set(["validation", "test"])
            assert export["validation"] == os.path.join(
                cast(str, step._output_folder_path), "export.val.csv"
            )
            assert export["test"] == os.path.join(
                cast(str, step._output_folder_path), "export.test.csv"
            )
            for v in export.values():
                assert os.path.isfile(v)


class TestExportToHFDataset:
    def test_export_dataset_to_hf_dataset_single_train_split(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export_path = os.path.join(cast(str, step._output_folder_path), "export")
            export = step.export_to_hf_dataset(export_path, train_size=1.0)
            assert isinstance(export, Dataset)
            assert len(export) == 10
            assert os.path.isdir(export_path)
            assert os.path.isfile(os.path.join(export_path, "dataset_info.json"))

    def test_export_dataset_to_hf_dataset_two_splits(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export_path = os.path.join(cast(str, step._output_folder_path), "export")
            export = step.export_to_hf_dataset(
                export_path, validation_size=0.3, test_size=0.7
            )
            assert isinstance(export, DatasetDict)
            assert set(export.keys()) == set(["validation", "test"])
            assert len(export["validation"]) == 3
            assert len(export["test"]) == 7
            assert os.path.isdir(export_path)
            assert os.path.isdir(os.path.join(export_path, "test"))
            assert os.path.isfile(
                os.path.join(export_path, "test", "dataset_info.json")
            )
            assert os.path.isdir(os.path.join(export_path, "validation"))
            assert os.path.isfile(
                os.path.join(export_path, "validation", "dataset_info.json")
            )


class TestPublishToHFHub:
    def test_code_implementation(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            assert Hasher.hash(getsource(step.publish_to_hf_hub)) == "9640479091d9bc18"

    @pytest.mark.skip(reason="skipping because requires interactive")
    def test_publish_dataset_to_hf_hub_single_train_split(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        def setup(self):
            self.register_data_card(DataCardType.DATASET_NAME, "truthful_qa")
            self.register_data_card(DataCardType.MODEL_NAME, "gpt3")

        with create_datadreamer():
            step = create_test_step(
                name="my-step", inputs=None, output_names=["out1"], setup=setup
            )
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export = step.publish_to_hf_hub("test_dataset_1", train_size=1.0)
            assert export == "https://huggingface.co/datasets/AjayP13/test_dataset_1"

    @pytest.mark.skip(reason="skipping because requires interactive")
    def test_publish_dataset_to_hf_hub_two_splits(
        self, create_datadreamer, create_test_step: Callable[..., Step]
    ):
        with create_datadreamer():
            step = create_test_step(name="my-step", inputs=None, output_names=["out1"])
            step._set_output(Dataset.from_dict({"out1": range(1, 11)}))
            export = step.publish_to_hf_hub(
                "AjayP13/test_dataset_2", validation_size=0.3, test_size=0.7
            )
            assert export == "https://huggingface.co/datasets/AjayP13/test_dataset_2"
