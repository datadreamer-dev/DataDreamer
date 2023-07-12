from mypy import api

from .. import __version__
from ..project import RUNNING_IN_PYTEST


class TestPackage:
    def test_version(self):
        assert len(__version__.split(".")) == 3

    def test_running_in_pytest(self):
        assert RUNNING_IN_PYTEST

    def test_mypy(self):
        result = api.run(["src/", "--sqlite-cache", "--explicit-package-bases"])
        if result[0]:
            print("\nType checking report:\n")
            print(result[0])
        if result[1]:
            print("\nError report:\n")
            print(result[1])
        assert result[2] == 0
