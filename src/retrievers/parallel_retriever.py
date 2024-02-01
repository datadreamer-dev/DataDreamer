from typing import Generator, Iterable, cast

from .._cachable import _ParallelCachable
from .retriever import DEFAULT_BATCH_SIZE, Retriever


class ParallelRetriever(_ParallelCachable, Retriever):
    def __init__(self, *retrievers: Retriever):
        super().__init__(*retrievers, cls=Retriever)
        self.retrievers = cast(list[Retriever], self.cachables)

    @property
    def index(self):  # pragma: no cover
        return self.retrievers[0].index

    def run(
        self, queries: Iterable[str], *args, **kwargs
    ) -> Generator[str | list[str], None, None] | list[str | list[str]]:
        kwargs["batch_size"] = kwargs.pop("batch_size", DEFAULT_BATCH_SIZE)
        results_generator = self._run_in_parallel(queries, *args, **kwargs)
        if not kwargs.get("return_generator", False):
            return list(results_generator)
        else:
            return results_generator

    def unload_model(self):
        for llm in self.retrievers:
            llm.unload_model()


__all__ = ["ParallelRetriever"]
