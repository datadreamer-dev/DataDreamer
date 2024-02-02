import gc
import os
from functools import cached_property
from itertools import chain, islice
from typing import Any, Callable, Generator, Iterable, cast

import numpy as np
import torch
from datasets.fingerprint import Hasher
from sqlitedict import SqliteDict

from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..embedders.embedder import Embedder
from ..utils.device_utils import device_to_device_id
from ..utils.fs_utils import safe_fn
from ..utils.import_utils import ignore_faiss_warnings
from .retriever import DEFAULT_BATCH_SIZE, Retriever

with ignore_faiss_warnings():
    import faiss


class EmbeddingRetriever(Retriever):
    def __init__(
        self,
        texts: OutputDatasetColumn | OutputIterableDatasetColumn,
        embedder: Embedder,
        truncate: bool = False,
        index_batch_size: int = DEFAULT_BATCH_SIZE,
        index_instruction: None | str = None,
        query_instruction: None | str = None,
        cache_folder_path: None | str = None,
        device: None | int | str | torch.device | list[int | str | torch.device] = None,
        **kwargs,
    ):
        """Loads an embedding retriever.

        Args:
            texts: The texts to index for retrieval.
            embedder: The embedder to use for embedding the texts.
            truncate: Whether to truncate the texts.
            index_batch_size: The batch size to use for indexing.
            index_instruction: An instruction to prepend to the texts when indexing.
            query_instruction: An instruction to prepend to the texts when querying.
            cache_folder_path: The path to the cache folder. If ``None``, the default
                cache folder for the DataDreamer session will be used.
            device: The type to use for the model weights.
            **kwargs: Additional keyword arguments to pass to the embedder.
        """
        super().__init__(texts=texts, cache_folder_path=cache_folder_path)
        self.embedder = embedder
        self.truncate = truncate
        self.index_batch_size = index_batch_size
        self.index_instruction = index_instruction
        self.query_instruction = query_instruction
        self.device = device
        if self.device and not isinstance(self.device, list):  # pragma: no cover
            self.device = [self.device]
        elif (
            self.device and isinstance(self.device, list) and len(self.device) == 0
        ):  # pragma: no cover
            self.device = None

        if self.device:  # pragma: no cover
            self.device = list(
                device_id
                for device_id in map(
                    device_to_device_id,
                    cast(list[int | str | torch.device], self.device),
                )
                if device_id is not None
            )
            if len(self.device) == 0:
                self.device = None
        self.kwargs = kwargs

    @cached_property
    def index(self):
        index_logger = self.get_logger(key="index")
        with self._retriever_index_folder_lock():
            # Create the index if it doesn't exist
            if not self._retriever_index_folder or not os.path.exists(
                self._retriever_index_folder
            ):
                self._initialize_retriever_index_folder()
                index_logger.info("Building index.")
                index = faiss.IndexFlatIP(self.embedder.dims)
                if self.device is not None:  # pragma: no cover
                    index = faiss.index_cpu_to_gpus_list(index=index, gpus=self.device)
                index_lookup = SqliteDict(
                    ":memory:"
                    if not self._tmp_retriever_index_folder
                    else os.path.join(self._tmp_retriever_index_folder, "faiss.db"),
                    tablename="lookup",
                    journal_mode="WAL",
                    autocommit=False,
                )

                # Add texts to index
                assert self.texts is not None
                texts_iter = iter(enumerate(self.texts))
                while True:
                    batch = list(zip(*list(islice(texts_iter, self.index_batch_size))))
                    if len(batch) == 0:
                        break
                    ids_batch, texts_batch = batch
                    texts_embedded = np.vstack(
                        cast(
                            list,
                            self.embedder.run(
                                texts=texts_batch,
                                truncate=self.truncate,
                                instruction=self.index_instruction,
                                batch_size=self.index_batch_size,
                                **self.kwargs,
                            ),
                        )
                    )
                    index.add(texts_embedded)
                    for id_, text in zip(ids_batch, texts_batch):
                        index_lookup[id_] = text
                    index_lookup.commit()

                # Write the index to disk
                if self._tmp_retriever_index_folder:
                    index_logger.info("Writing index to disk.")
                    faiss.write_index(
                        index,
                        os.path.join(self._tmp_retriever_index_folder, "faiss.index"),
                    )
                    del index
                    index_lookup.conn.execute("PRAGMA journal_mode=OFF")
                    index_lookup.commit()
                    index_lookup.close()
                    del index_lookup
                    gc.collect()
                    self._finalize_retriever_index_folder()
                    index_logger.info("Finished writing index to disk.")

            # Load the index from disk
            if self._retriever_index_folder and os.path.exists(
                self._retriever_index_folder
            ):
                index = faiss.read_index(
                    os.path.join(self._retriever_index_folder, "faiss.index"),
                    faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
                )
                if self.device is not None:  # pragma: no cover
                    index = faiss.index_cpu_to_gpus_list(index=index, gpus=self.device)
                index_lookup = SqliteDict(
                    ":memory:"
                    if not self._retriever_index_folder
                    else os.path.join(self._retriever_index_folder, "faiss.db"),
                    tablename="lookup",
                    journal_mode="WAL",
                    autocommit=False,
                )

            return index, index_lookup

    def _batch_lookup(self, indices: list[int]) -> dict[int, dict[str, Any]]:
        _, index_lookup = self.index
        results = {}
        indices_iter = iter(set(indices))
        while True:
            indices_batch = list(islice(indices_iter, 900))
            if len(indices_batch) == 0:
                break
            lookup_query = (
                "SELECT key, value FROM lookup"
                f' WHERE key IN ({",".join(["?"] * len(indices_batch))})'
            )
            results.update(
                {
                    int(result[0]): index_lookup.decode(result[1])
                    for result in index_lookup.conn.select(
                        lookup_query, [str(idx) for idx in indices_batch]
                    )
                }
            )
        return results

    def _run_batch(
        self,
        max_length_func: Callable[[list[str]], int],
        inputs: list[str],
        k: int = 5,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs,
    ) -> list[dict[str, Any]]:
        queries = inputs
        final_kwargs = self.kwargs.copy()
        final_kwargs.update(kwargs)
        queries_embedded = np.vstack(
            cast(
                list,
                self.embedder.run(
                    texts=queries,
                    truncate=self.truncate,
                    instruction=self.query_instruction,
                    batch_size=len(queries),
                    **final_kwargs,
                ),
            )
        )
        index, _ = self.index
        scores, results = index.search(queries_embedded, k=k)
        texts_lookup = self._batch_lookup(indices=list(chain.from_iterable(results)))
        return [
            {
                "indices": list(filter(lambda idx: idx >= 0, query_result)),
                "texts": [texts_lookup[idx] for idx in query_result if idx != -1],
                "scores": [
                    score
                    for score, idx in zip(query_result_scores, query_result)
                    if idx != -1
                ],
            }
            for query_result_scores, query_result in zip(scores, results)
        ]

    def run(
        self,
        queries: Iterable[Any],
        k: int = 5,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = False,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_queries: None | int = None,
        return_generator: bool = False,
        **kwargs,
    ) -> Generator[dict[str, Any], None, None] | list[dict[str, Any]]:
        adaptive_batch_size = False

        def get_max_length_function() -> dict[str, Any]:
            def max_length_func(queries: list[Any]) -> int:  # pragma: no cover
                return 0

            return {"max_length_func": max_length_func}

        results_generator = self._run_over_batches(
            run_batch=self._run_batch,
            get_max_input_length_function=get_max_length_function,
            max_model_length=1,
            inputs=queries,
            k=k,
            batch_size=batch_size,
            batch_scheduler_buffer_size=batch_scheduler_buffer_size,
            adaptive_batch_size=adaptive_batch_size,
            progress_interval=progress_interval,
            force=force,
            cache_only=cache_only,
            verbose=verbose,
            log_level=log_level,
            total_num_inputs=total_num_queries,
            **kwargs,
        )
        if not return_generator:
            return list(results_generator)
        else:
            return results_generator

    @cached_property
    def license(self) -> None | str:
        return "https://github.com/facebookresearch/faiss/blob/main/LICENSE"

    @cached_property
    def citation(self) -> None | list[str]:
        citations = []
        citations.append(
            """
@article{johnson2019billion,
  title={Billion-scale similarity search with {GPUs}},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
            """.strip()
        )
        return citations

    @property
    def version(self) -> float:
        return 1.0

    @cached_property
    def display_name(self) -> str:
        assert self.texts is not None
        return (
            super().display_name
            + f" ({self.embedder.model_name}) ({self.texts.column_names[0]})"
        )

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.embedder.model_name, allow_slashes=False)]
        to_hash: list[Any] = []
        if len(to_hash) > 0:  # pragma: no cover
            names.append(Hasher.hash(to_hash))
        return "_".join(names)

    def unload_model(self):
        # Delete cached index
        if "index" in self.__dict__:
            del self.__dict__["index"]

        # Garbage collect
        gc.collect()

    def __getstate__(self):  # pragma: no cover
        state = super().__getstate__()

        # Remove cached index before serializing
        state.pop("index", None)

        return state


__all__ = ["EmbeddingRetriever"]
