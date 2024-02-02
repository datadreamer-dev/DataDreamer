from functools import cached_property
from typing import Any, Callable, Generator, Iterable

import numpy as np
import torch
from tiktoken import Encoding

from ..llms.together import Together, _get_final_tokenizer_model_name
from ..task_models.task_model import DEFAULT_BATCH_SIZE, _check_texts_length
from ..utils import ring_utils as ring
from ..utils.hf_model_utils import get_model_embedding_size
from ..utils.import_utils import ignore_transformers_warnings
from .embedder import Embedder

with ignore_transformers_warnings():
    from transformers import PretrainedConfig


class TogetherEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        api_key: None | str = None,
        max_context_length: None | int = None,
        tokenizer_model_name: str | None = None,
        tokenizer_revision: None | str = None,
        tokenizer_trust_remote_code: bool = False,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        warn_tokenizer_model_name: bool | None = True,
        warn_max_context_length: bool | None = True,
        **kwargs,
    ):
        """Loads a `Together AI <https://www.together.ai/>`_ embedder.

        Args:
            model_name: The name of the model to use.
            api_key: The API key to use for the API.
            max_context_length: The maximum context length to use for the model. If
                ``None``, the maximum context length will be inferred.
            tokenizer_model_name: The name of the tokenizer model to use. If ``None``,
                the tokenizer model will be inferred.
            tokenizer_revision: The revision of the tokenizer model to use.
            tokenizer_trust_remote_code: Whether to trust remote code for the
                tokenizer.
            retry_on_fail: Whether to retry API calls if they fail.
            cache_folder_path: The path to the cache folder. If ``None``, the default
                cache folder for the DataDreamer session will be used.
            warn_tokenizer_model_name: Whether to warn if the tokenizer model name is
                inferred and not explicitly specified.
            warn_max_context_length: Whether to warn if the maximum context length is
                inferred and not explicitly specified.
            **kwargs: Additional keyword arguments to pass to the Together client.
        """
        super().__init__(model_name=model_name, cache_folder_path=cache_folder_path)
        self.api_key = api_key
        self.max_context_length = max_context_length
        final_tokenizer_model_name = _get_final_tokenizer_model_name(
            model_name=model_name, tokenizer_model_name=tokenizer_model_name
        )
        self._tokenizer_model_name = final_tokenizer_model_name
        self.tokenizer_revision = tokenizer_revision
        self.tokenizer_trust_remote_code = tokenizer_trust_remote_code
        self.warn_tokenizer_model_name = warn_tokenizer_model_name
        self.warn_max_context_length = warn_max_context_length
        self._warned_max_context_length = False
        self.kwargs = kwargs

        # Setup API calling helpers
        self.retry_on_fail = retry_on_fail

    @cached_property
    def retry_wrapper(self):
        return Together.retry_wrapper.func(self)  # type: ignore[attr-defined]

    @cached_property
    def tokenizer_model_name(self) -> str:
        return Together.tokenizer_model_name.func(self)  # type: ignore[attr-defined]

    @cached_property
    def config(self) -> None | PretrainedConfig:
        return Together.config.func(self)  # type: ignore[attr-defined]

    @cached_property
    def client(self) -> Any:
        return Together.client.func(self)  # type: ignore[attr-defined]

    @cached_property
    def tokenizer(self) -> Encoding:
        return Together.tokenizer.func(self)  # type: ignore[attr-defined]

    @ring.lru(maxsize=5000)
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        return Together.count_tokens._callable.wrapped_callable(self, value)

    @cached_property
    def model_max_length(self) -> int:
        return Together.get_max_context_length._callable.wrapped_callable(
            self, max_new_tokens=0
        )

    @cached_property
    def dims(self) -> int:
        return get_model_embedding_size(
            model_name=self.tokenizer_model_name, config=self.config
        )

    @torch.no_grad()
    def _run_batch(
        self,
        max_length_func: Callable[[list[str]], int],
        inputs: list[str],
        truncate: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs,
    ) -> list[np.ndarray]:
        texts = inputs
        if not truncate:
            _check_texts_length(self=self, max_length_func=max_length_func, texts=texts)

        embeddings = np.asarray(
            [
                e.embedding
                for e in self.retry_wrapper(
                    func=self.client.Embeddings.create,
                    model=self.model_name,
                    input=texts,
                    **kwargs,
                ).data
            ]
        )
        return (
            (embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True))
            if kwargs.pop("normalize_embeddings", True)
            else list(embeddings)
        )

    def run(  # type:ignore[override]
        self,
        texts: Iterable[str],
        truncate: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = False,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_texts: None | int = None,
        return_generator: bool = False,
        **kwargs,
    ) -> Generator[np.ndarray, None, None] | list[np.ndarray]:
        def get_max_length_function() -> dict[str, Any]:
            def max_length_func(texts: list[str]) -> int:
                return max([self.count_tokens(t) for t in texts])

            return {"max_length_func": max_length_func}

        results_generator = self._run_over_batches(
            run_batch=self._run_batch,
            get_max_input_length_function=get_max_length_function,
            max_model_length=self.model_max_length,
            inputs=texts,
            truncate=truncate,
            batch_size=batch_size,
            batch_scheduler_buffer_size=batch_scheduler_buffer_size,
            adaptive_batch_size=adaptive_batch_size,
            progress_interval=progress_interval,
            force=force,
            cache_only=cache_only,
            verbose=verbose,
            log_level=log_level,
            total_num_inputs=total_num_texts,
            **kwargs,
        )
        if not return_generator:
            return list(results_generator)
        else:
            return results_generator

    @cached_property
    def model_card(self) -> None | str:
        return Together.model_card.func(self)  # type: ignore[attr-defined]

    @cached_property
    def license(self) -> None | str:
        return Together.license.func(self)  # type: ignore[attr-defined]

    @cached_property
    def citation(self) -> None | list[str]:
        return Together.citation.func(self)  # type: ignore[attr-defined]

    @property
    def version(self) -> float:
        return 1.0

    @cached_property
    def display_name(self) -> str:
        return super().display_name + f" ({self.model_name})"

    @cached_property
    def _cache_name(self) -> None | str:
        return Together._cache_name.func(self)  # type: ignore[attr-defined]

    def unload_model(self):
        return Together.unload_model(self)  # type: ignore[arg-type]

    def __getstate__(self):  # pragma: no cover
        return Together.__getstate__(self)  # type: ignore[arg-type]


__all__ = ["TogetherEmbedder"]
