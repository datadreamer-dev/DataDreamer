from typing import cast

from ..task_models.parallel_task_model import ParallelTaskModel
from .embedder import Embedder


class ParallelEmbedder(ParallelTaskModel, Embedder):
    def __init__(self, *embedders: Embedder):
        """
        Creates an embedder that will run multiple embedders in parallel. See
        :doc:`running models in parallel
        <./pages/advanced_usage/parallelization/running_models_on_multiple_gpus>`
        for more details.

        Args:
            *embedders: The embedders to run in parallel.
        """
        super().__init__(*embedders)
        self.embedders = cast(list[Embedder], self.cachables)

    @property
    def model_max_length(self) -> int:
        return self.embedders[0].model_max_length

    @property
    def dims(self) -> int:
        return self.embedders[0].dims


__all__ = ["ParallelEmbedder"]
