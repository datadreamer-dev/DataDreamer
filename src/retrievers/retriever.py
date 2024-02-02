import contextlib
import os
from abc import abstractmethod
from functools import cached_property
from typing import Any, Callable, Generator, Iterable, cast
from uuid import uuid4

from datasets.fingerprint import Hasher
from filelock import FileLock

from .. import DataDreamer
from .._cachable import _Cachable
from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..utils.fs_utils import clear_dir, mkdir, rm_dir

DEFAULT_BATCH_SIZE = 10


class Retriever(_Cachable):
    def __init__(
        self,
        texts: None | OutputDatasetColumn | OutputIterableDatasetColumn,
        cache_folder_path: None | str = None,
    ):
        """Base class for all retrievers.

        Args:
            texts: The texts to index for retrieval.
            cache_folder_path: The path to the cache folder. If ``None``, the default
                cache folder for the DataDreamer session will be used.
        """
        super().__init__(cache_folder_path=cache_folder_path)
        self.texts = texts
        self.texts_fingerprint = Hasher.hash(
            (self.texts.step.fingerprint, self.texts.column_names)
            if self.texts is not None
            else None
        )

    def _initialize_retriever_index_folder(self):
        if DataDreamer.initialized() and not DataDreamer.is_running_in_memory():
            clear_dir(cast(str, self._tmp_retriever_index_folder))
            rm_dir(cast(str, self._retriever_index_folder))

    @property
    def _tmp_retriever_index_folder(self) -> None | str:
        if DataDreamer.initialized() and not DataDreamer.is_running_in_memory():
            cls_name = self.__class__.__name__
            path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "retrievers",
                f"{cls_name}_{self._cache_name}_{self.version}",
                self.texts_fingerprint + ".tmp",
            )
            mkdir(path)
            return path
        return None

    def _retriever_index_folder_lock(self) -> Any:
        if DataDreamer.initialized() and not DataDreamer.is_running_in_memory():
            cls_name = self.__class__.__name__
            path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "retrievers",
                f"{cls_name}_{self._cache_name}_{self.version}",
                self.texts_fingerprint + ".flock",
            )
            mkdir(os.path.dirname(path))
            return FileLock(path)
        return contextlib.nullcontext()

    @property
    def _retriever_index_folder(self) -> None | str:
        if DataDreamer.initialized() and not DataDreamer.is_running_in_memory():
            cls_name = self.__class__.__name__
            path = os.path.join(
                DataDreamer.get_output_folder_path(),
                ".cache",
                "retrievers",
                f"{cls_name}_{self._cache_name}_{self.version}",
                self.texts_fingerprint,
            )
            return path
        return None

    def _finalize_retriever_index_folder(self):
        if DataDreamer.initialized() and not DataDreamer.is_running_in_memory():
            rm_dir(cast(str, self._retriever_index_folder))
            os.rename(
                cast(str, self._tmp_retriever_index_folder),
                cast(str, self._retriever_index_folder),
            )
            rm_dir(cast(str, self._tmp_retriever_index_folder))

    @property
    @abstractmethod
    def index(self):
        pass

    def _run_over_batches(  # noqa: C901
        self,
        run_batch: Callable[..., list[Any]],
        get_max_input_length_function: None | Callable[[], dict[str, Any]],
        max_model_length: None | int | Callable,
        inputs: Iterable[Any],
        batch_size: int = 1,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = True,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_inputs: None | int = None,
        **kwargs,
    ) -> Generator[Any, None, None]:
        yield from self._run_over_batches_locked(
            run_batch=run_batch,
            get_max_input_length_function=get_max_input_length_function,
            max_model_length=max_model_length,
            inputs=inputs,
            batch_size=batch_size,
            batch_scheduler_buffer_size=batch_scheduler_buffer_size,
            adaptive_batch_size=adaptive_batch_size,
            progress_interval=progress_interval,
            force=force,
            cache_only=cache_only,
            verbose=verbose,
            log_level=log_level,
            total_num_inputs=total_num_inputs,
            **kwargs,
        )

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
    def display_icon(self) -> str:
        return " 🔎 "

    @cached_property
    def display_name(self) -> str:
        return super().display_name

    @cached_property
    def _cache_name(self) -> None | str:  # pragma: no cover
        return None

    @property
    def _input_type(self) -> str:
        return "query"

    def __ring_key__(self) -> int:  # pragma: no cover
        return uuid4().int

    def unload_model(self):  # pragma: no cover  # noqa: B027
        """Unloads resources required to run the retriever from memory."""
        pass


__all__ = ["Retriever"]
