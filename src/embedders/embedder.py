from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Generator, Iterable

from ..task_models.task_model import DEFAULT_BATCH_SIZE, TaskModel


class Embedder(TaskModel):
    def __init__(self, model_name: str, cache_folder_path: None | str = None):
        """Base class for all embedders.

        Args:
            model_name: The name of the model to use.
            cache_folder_path: The path to the cache folder. If ``None``, the default
                cache folder for the DataDreamer session will be used.
        """
        super().__init__(cache_folder_path=cache_folder_path)
        self.model_name = model_name

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

    @property
    @abstractmethod
    def dims(self) -> int:
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
        # Apply an instruction over the inputs if there is one
        if kwargs.get("instruction", None) is not None:
            instruction = kwargs["instruction"]

            def apply_instruction(instruction: str, text: str) -> str:
                return instruction + text

            inputs = map(partial(apply_instruction, instruction), inputs)

        return super()._run_over_batches(
            run_batch=run_batch,
            get_max_input_length_function=get_max_input_length_function,
            max_model_length=self.model_max_length,
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

    @abstractmethod
    def run(  # type:ignore[override]
        self,
        texts: Iterable[str],
        truncate: bool = False,
        instruction: None | str = None,
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
    ) -> Generator[Any, None, None] | list[Any]:
        pass

    def unload_model(self):  # pragma: no cover  # noqa: B027
        """Unloads resources required to run the model from memory."""
        pass


__all__ = ["Embedder"]
