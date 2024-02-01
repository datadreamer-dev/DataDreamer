import os
import uuid
from typing import Callable

import pytest

from .... import DataDreamer
from ..config import TEST_DIR


@pytest.fixture
def create_datadreamer() -> Callable[..., DataDreamer]:
    def _create_datadreamer(path: None | str = None, **kwargs) -> DataDreamer:
        if path is None:
            path = uuid.uuid4().hex[0:10]
        if path == ":memory:":
            return DataDreamer(path, **kwargs)
        else:
            return DataDreamer(os.path.join(TEST_DIR, path), **kwargs)

    return _create_datadreamer
