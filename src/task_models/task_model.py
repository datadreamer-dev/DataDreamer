from abc import abstractmethod
from functools import cached_property
from typing import Callable, Generator, Iterable
from uuid import uuid4

from .._cachable import _Cachable

DEFAULT_BATCH_SIZE = 10


def _check_texts_length(
    self: "TaskModel", max_length_func: Callable[[list[str]], int], texts: list[str]
):
    # Get max text length
    max_text_length = max_length_func(texts)

    # Check max_new_tokens
    if max_text_length > self.model_max_length:
        raise ValueError(
            "The length of your texts exceeds the max length of the model, use"
            " `truncate=True` if you wish to truncate inputs."
        )


class TaskModel(_Cachable):
    def __init__(self, cache_folder_path: None | str = None):
        """Base class for all task models.

        Args:
            cache_folder_path: The path to the cache folder. If ``None``, the default
                cache folder for the DataDreamer session will be used.
        """
        super().__init__(cache_folder_path=cache_folder_path)

    @abstractmethod
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass

    @property
    @abstractmethod
    def model_max_length(self) -> int:
        pass

    @abstractmethod
    def run(
        self,
        texts: Iterable[str],
        truncate: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = True,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_texts: None | int = None,
        return_generator: bool = False,
        **kwargs,
    ) -> Generator[dict[str, float], None, None] | list[dict[str, float]]:
        pass

    @cached_property
    def model_card(self) -> None | str:  # pragma: no cover
        return None

    @cached_property
    def license(self) -> None | str:  # pragma: no cover
        return None

    @cached_property
    def citation(self) -> None | list[str]:  # pragma: no cover
        return None

    @property
    def version(self) -> float:  # pragma: no cover
        return 1.0

    @cached_property
    def display_icon(self) -> str:  # pragma: no cover
        return ""

    @cached_property
    def display_name(self) -> str:  # pragma: no cover
        return super().display_name

    @cached_property
    def _cache_name(self) -> None | str:  # pragma: no cover
        return None

    @property
    def _input_type(self) -> str:  # pragma: no cover
        return "text"

    def __ring_key__(self) -> int:  # pragma: no cover
        return uuid4().int

    def unload_model(self):  # pragma: no cover  # noqa: B027
        """Unloads resources required to run the model from memory."""
        pass


__all__ = ["TaskModel"]
