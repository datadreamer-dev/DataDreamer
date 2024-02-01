import os

import pytest


@pytest.fixture(autouse=True)
def restore_os_environ():
    orig_environ = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(orig_environ)
