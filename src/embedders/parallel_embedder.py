from typing import cast

from ..task_models.parallel_task_model import ParallelTaskModel
from .embedder import Embedder


class ParallelEmbedder(ParallelTaskModel, Embedder):
    def __init__(self, *embedders: Embedder):
        super().__init__(*embedders)
        self.embedders = cast(list[Embedder], self.cachables)

    @property
    def model_max_length(self) -> int:
        return self.embedders[0].model_max_length

    @property
    def dims(self) -> int:
        return self.embedders[0].dims


__all__ = ["ParallelEmbedder"]
