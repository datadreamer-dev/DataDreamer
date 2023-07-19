import os
import uuid
from typing import Callable

import pytest

from .... import DataDreamer


@pytest.fixture
def create_datadreamer() -> Callable[..., DataDreamer]:
    def _create_datadreamer(path: None | str = None) -> DataDreamer:
        if path is None:
            path = uuid.uuid4().hex[0:10]
        return DataDreamer(os.path.join("./.tests_data", path))

    return _create_datadreamer
