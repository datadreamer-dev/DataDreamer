from .. import __version__
from ..project import RUNNING_IN_PYTEST


class TestPackage:
    def test_version(self):
        assert len(__version__.split(".")) == 3

    def test_running_in_pytest(self):
        assert RUNNING_IN_PYTEST
