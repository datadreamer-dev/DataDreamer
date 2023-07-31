import json
import tempfile

import pytest

from datasets import Dataset, DatasetDict

from ...datasets import OutputDataset, OutputIterableDataset
from ...steps import DataSource, JSONDataSource, Step


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
