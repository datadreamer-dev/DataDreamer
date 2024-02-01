import os

import pytest

from ....utils.fs_utils import clear_dir


@pytest.fixture(autouse=True)
def clear_github_space():
    yield
    if "GITHUB_ACTIONS" in os.environ:
        # Clear the tests data directory to make more disk space available
        try:
            clear_dir("./.tests_data")
            os.system("rm -rf ~/.cache/huggingface/")
        except FileNotFoundError:
            pass
