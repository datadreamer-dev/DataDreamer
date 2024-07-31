# An update in datasets 2.20.0 adding state_dict to IterableDataset seems to have
# broken IterableDataset. This patch is a temporary fix until the issue is resolved.

import contextlib
from unittest.mock import patch

from datasets.iterable_dataset import (
    ArrowExamplesIterable,
    ExamplesIterable,
    TypedExamplesIterable,
)

__original_init_state_dict = TypedExamplesIterable._init_state_dict
__original_examples__iter__ = ExamplesIterable.__iter__
__original_arrowexamples__iter__ = ArrowExamplesIterable.__iter__
_should_reset_state_dict = False


def patched_examples__iter__(self):
    global _should_reset_state_dict
    if _should_reset_state_dict:
        self._init_state_dict()
    return __original_examples__iter__(self)


def patched_arrowexamples__iter__(self):
    global _should_reset_state_dict
    if _should_reset_state_dict:
        self._init_state_dict()
    return __original_arrowexamples__iter__(self)


ExamplesIterable.__iter__ = patched_examples__iter__
ArrowExamplesIterable.__iter__ = patched_arrowexamples__iter__


@contextlib.contextmanager
def apply_datasets_reset_state_hack():
    def patched_init_state_dict(self):
        self._state_dict = None  # Set to None to ensure it is reset
        return __original_init_state_dict(self)

    with patch(
        "datasets.iterable_dataset.TypedExamplesIterable._init_state_dict",
        patched_init_state_dict,
    ):
        yield None


def start_datasets_reset_state_hack():
    global _should_reset_state_dict
    _should_reset_state_dict = True


def stop_datasets_reset_state_hack():
    global _should_reset_state_dict
    _should_reset_state_dict = False
