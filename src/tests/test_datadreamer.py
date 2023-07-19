import os

import pytest

from .. import DataDreamer


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


class TestFunctionality:
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
