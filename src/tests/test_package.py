import os
import platform

import pytest
from mypy import api

from .. import __version__
from ..project import RUNNING_IN_PYTEST


class TestPackage:
    def test_version(self):
        assert len(__version__.split(".")) == 3

    def test_running_in_pytest(self):
        assert RUNNING_IN_PYTEST

    @pytest.mark.skipif(
        "GITHUB_ACTIONS" not in os.environ and "PROJECT_CLUSTER" not in os.environ,
        reason="only run on CI",
    )
    def test_python_version(self):
        with open("./scripts/.python-version", "r") as f:
            python_version = f.read().strip()
            assert python_version == platform.python_version()

    def test_mypy(self):
        result = api.run(["src/", "--sqlite-cache", "--explicit-package-bases"])
        if result[0]:
            print("\nType checking report:\n")
            print(result[0])
        if result[1]:
            print("\nError report:\n")
            print(result[1])
        assert result[2] == 0
