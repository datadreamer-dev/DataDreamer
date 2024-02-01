import json
import os
import tempfile

import pytest
from datasets import Dataset, DatasetDict

from ... import DataDreamer
from ...datasets import OutputDataset, OutputIterableDataset
from ...steps import (
    CSVDataSource,
    DataSource,
    HFDatasetDataSource,
    HFHubDataSource,
    JSONDataSource,
    Step,
    TextDataSource,
)


class TestDataSource:
    def test_from_dict(self, create_datadreamer):
        with create_datadreamer():
            data_source = DataSource("my-dataset", {"out1": range(1, 11)})
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputDataset)
            assert set(data_source.output.column_names) == set(["out1"])
            assert len(data_source.output) == 10

    def test_from_list(self, create_datadreamer):
        with create_datadreamer():
            data_source = DataSource("my-dataset", [{"out1": i} for i in range(1, 11)])
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputDataset)
            assert set(data_source.output.column_names) == set(["out1"])
            assert len(data_source.output) == 10

    def test_from_generator_warning(self, create_datadreamer):
        with create_datadreamer():

            def dataset_generator():
                for i in range(1, 11):
                    yield {"out1": i}

            with pytest.warns(UserWarning):
                DataSource("my-dataset", dataset_generator)

    def test_from_generator(self, create_datadreamer):
        with create_datadreamer():

            def dataset_generator():
                for i in range(1, 11):
                    yield {"out1": i}

            data_source = DataSource("my-dataset", dataset_generator, total_num_rows=10)
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputIterableDataset)
            assert set(data_source.output.column_names) == set(["out1"])
            assert data_source.output.num_rows == 10

    def test_from_dataset(self, create_datadreamer):
        with create_datadreamer():
            data_source = DataSource(
                "my-dataset", Dataset.from_dict({"out1": range(1, 11)})
            )
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputDataset)
            assert set(data_source.output.column_names) == set(["out1"])
            assert len(data_source.output) == 10

    def test_from_dataset_dict_error(self, create_datadreamer):
        with create_datadreamer():
            with pytest.raises(ValueError):
                DataSource(
                    "my-dataset",
                    DatasetDict({"train": Dataset.from_dict({"out1": range(1, 11)})}),
                )

    def test_from_iterable_dataset(self, create_datadreamer):
        with create_datadreamer():
            data_source = DataSource(
                "my-dataset",
                Dataset.from_dict({"out1": range(1, 11)}).to_iterable_dataset(),
                total_num_rows=10,
            )
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputIterableDataset)
            assert set(data_source.output.column_names) == set(["out1"])
            assert data_source.output.num_rows == 10


class TestJSONDataSource:
    def test_from_json_file(self, create_datadreamer):
        with create_datadreamer():
            with tempfile.NamedTemporaryFile(mode="w+") as f:
                json.dump([{"out1": i} for i in range(1, 11)], f)
                f.flush()
                data_source = JSONDataSource("my-dataset", data_files=f.name)
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputDataset)
            assert set(data_source.output.column_names) == set(["out1"])
            assert len(data_source.output) == 10

    def test_from_json_file_splits_error(self, create_datadreamer):
        with create_datadreamer():
            with tempfile.NamedTemporaryFile(mode="w+") as f:
                json.dump([{"out1": i} for i in range(1, 11)], f)
                f.flush()
                with pytest.raises(ValueError):
                    JSONDataSource(
                        "my-dataset",
                        data_files={"train": f.name},  # type: ignore[arg-type]
                    )


class TestCSVDataSource:
    def test_from_csv_file(self, create_datadreamer):
        with create_datadreamer():
            with tempfile.NamedTemporaryFile(mode="w+") as f:
                f.write("out1\n" + "\n".join([str(i) for i in range(1, 11)]))
                f.flush()
                data_source = CSVDataSource("my-dataset", data_files=f.name)
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputDataset)
            assert set(data_source.output.column_names) == set(["out1"])
            assert len(data_source.output) == 10

    def test_from_csv_file_splits_error(self, create_datadreamer):
        with create_datadreamer():
            with tempfile.NamedTemporaryFile(mode="w+") as f:
                f.write("out1\n" + "\n".join([str(i) for i in range(1, 11)]))
                f.flush()
                with pytest.raises(ValueError):
                    CSVDataSource(
                        "my-dataset",
                        data_files={"train": f.name},  # type: ignore[arg-type]
                    )


class TestTextDataSource:
    def test_from_text_file(self, create_datadreamer):
        with create_datadreamer():
            with tempfile.NamedTemporaryFile(mode="w+") as f:
                f.write("\n".join([str(i) for i in range(1, 11)]))
                f.flush()
                data_source = TextDataSource("my-dataset", data_files=f.name)
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputDataset)
            assert set(data_source.output.column_names) == set(["text"])
            assert len(data_source.output) == 10

    def test_from_text_file_splits_error(self, create_datadreamer):
        with create_datadreamer():
            with tempfile.NamedTemporaryFile(mode="w+") as f:
                f.write("\n".join([str(i) for i in range(1, 11)]))
                f.flush()
                with pytest.raises(ValueError):
                    TextDataSource(
                        "my-dataset",
                        data_files={"train": f.name},  # type: ignore[arg-type]
                    )


class TestHFDatasetDataSource:
    def test_from_hf_dataset_folder(self, create_datadreamer):
        with create_datadreamer():
            export_path = os.path.join(DataDreamer.get_output_folder_path(), "export")
            Dataset.from_dict({"out1": range(1, 11)}).save_to_disk(export_path)
            data_source = HFDatasetDataSource("my-dataset", export_path)
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputDataset)
            assert set(data_source.output.column_names) == set(["out1"])
            assert len(data_source.output) == 10

    def test_from_hf_dataset_folder_splits_error(self, create_datadreamer):
        with create_datadreamer():
            export_path = os.path.join(DataDreamer.get_output_folder_path(), "export")
            DatasetDict(
                {"train": Dataset.from_dict({"out1": range(1, 11)})}
            ).save_to_disk(export_path)
            with pytest.raises(FileNotFoundError):
                HFDatasetDataSource("my-dataset", export_path)


class TestHFHubDataSource:
    def test_from_hf_hub(self, create_datadreamer):
        with create_datadreamer():
            data_source = HFHubDataSource(
                "my-dataset", "truthful_qa", "multiple_choice", "validation"
            )
            assert isinstance(data_source, Step)
            assert isinstance(data_source.output, OutputDataset)
            assert set(data_source.output.column_names) == set(
                ["question", "mc1_targets", "mc2_targets"]
            )
            assert len(data_source.output) == 817
            data_card = data_source._data_card
            del data_card["my-dataset"]["Date & Time"]
            assert data_card == {
                "my-dataset": {
                    "Dataset Name": ["truthful_qa"],
                    "Dataset Card": ["https://huggingface.co/datasets/truthful_qa"],
                    "License Information": ["apache-2.0"],
                    "Citation Information": [
                        "@misc{lin2021truthfulqa,\n    title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},\n    author={Stephanie Lin and Jacob Hilton and Owain Evans},\n    year={2021},\n    eprint={2109.07958},\n    archivePrefix={arXiv},\n    primaryClass={cs.CL}\n}"  # noqa: B950
                    ],
                }
            }

    def test_from_hf_hub_splits_error(self, create_datadreamer):
        with create_datadreamer():
            with pytest.raises(ValueError):
                HFHubDataSource("my-dataset", "truthful_qa", "multiple_choice")
