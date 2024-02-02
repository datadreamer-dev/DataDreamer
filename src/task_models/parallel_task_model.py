from typing import Generator, Iterable, cast

from .._cachable import _ParallelCachable
from .task_model import DEFAULT_BATCH_SIZE, TaskModel


class ParallelTaskModel(_ParallelCachable, TaskModel):
    def __init__(self, *task_models: TaskModel):
        """
        Creates a task model that will run multiple task models in parallel. See
        :doc:`running models in parallel
        <./pages/advanced_usage/parallelization/running_models_on_multiple_gpus>`
        for more details.

        Args:
            *task_models: The task models to run in parallel.
        """
        super().__init__(*task_models, cls=TaskModel)
        self.task_models = cast(list[TaskModel], self.cachables)

    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        return self.task_models[0].count_tokens(value=value)

    @property
    def model_max_length(self) -> int:  # pragma: no cover
        return self.task_models[0].model_max_length

    def run(  # type:ignore[override]
        self, texts: Iterable[str], *args, **kwargs
    ) -> Generator[str | list[str], None, None] | list[str | list[str]]:
        kwargs["batch_size"] = kwargs.pop("batch_size", DEFAULT_BATCH_SIZE)
        results_generator = self._run_in_parallel(texts, *args, **kwargs)
        if not kwargs.get("return_generator", False):
            return list(results_generator)
        else:
            return results_generator

    def unload_model(self):
        for llm in self.task_models:
            llm.unload_model()


__all__ = ["ParallelTaskModel"]
