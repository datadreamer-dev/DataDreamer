import gc
from functools import cached_property
from typing import Any, Callable, Generator, Iterable

import numpy as np
import openai
import torch
import torch._dynamo
from datasets.fingerprint import Hasher
from tiktoken import Encoding

from ..llms.openai import OpenAI
from ..task_models.task_model import DEFAULT_BATCH_SIZE, _check_texts_length
from ..utils import ring_utils as ring
from ..utils.arg_utils import DEFAULT, Default, default_to
from ..utils.fs_utils import safe_fn
from .embedder import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        dimensions: int | Default = DEFAULT,
        organization: None | str = None,
        api_key: None | str = None,
        base_url: None | str = None,
        api_version: None | str = None,
        retry_on_fail: bool = False,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        """Loads an `OpenAI <https://openai.com/>`_ embedder.

        Args:
            model_name: The name of the model to use.
            dimensions: The number of dimensions to use for the embeddings. If ``None``,
                the default number of dimensions for the model will be used.
            organization: The organization to use for the API.
            api_key: The API key to use for the API.
            base_url: The base URL to use for the API.
            api_version: The version of the API to use.
            retry_on_fail: Whether to retry API calls if they fail.
            cache_folder_path: The path to the cache folder. If ``None``, the default
                cache folder for the DataDreamer session will be used.
            **kwargs: Additional keyword arguments to pass to the OpenAI client.
        """
        super().__init__(model_name=model_name, cache_folder_path=cache_folder_path)
        self._dimensions = dimensions
        self.organization = organization
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.kwargs = kwargs

        # Setup API calling helpers
        self.retry_on_fail = retry_on_fail

    @cached_property
    def retry_wrapper(self):
        return OpenAI.retry_wrapper.func(self)  # type: ignore[attr-defined]

    @cached_property
    def client(self) -> openai.OpenAI | openai.AzureOpenAI:
        return OpenAI.client.func(self)  # type: ignore[attr-defined]

    @cached_property
    def tokenizer(self) -> Encoding:
        return OpenAI.tokenizer.func(self)  # type: ignore[attr-defined]

    @ring.lru(maxsize=5000)
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        return OpenAI.count_tokens._callable.wrapped_callable(self, value)

    @cached_property
    def model_max_length(self) -> int:
        return 8191

    @cached_property
    def dims(self) -> int:  # pragma: no cover
        default_dims = 1536
        if "-large" in self.model_name:
            default_dims = 3072
        return default_to(self._dimensions, default_dims)

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

        optional_kwargs = {}
        if not isinstance(self._dimensions, Default):
            optional_kwargs["dimensions"] = self._dimensions
        return [
            np.asarray(e.embedding)
            for e in self.retry_wrapper(
                func=self.client.embeddings.create,
                model=self.model_name,
                input=texts,
                **optional_kwargs,
                **kwargs,
            ).data
        ]

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
        return "https://openai.com/blog/new-embedding-models-and-api-updates/"

    @cached_property
    def license(self) -> None | str:
        return OpenAI.license.func(self)  # type: ignore[attr-defined]

    @cached_property
    def citation(self) -> None | list[str]:
        citations = []
        citations.append(
            """
@article{Neelakantan2022TextAC,
  title={Text and Code Embeddings by Contrastive Pre-Training},
  author={Arvind Neelakantan and Tao Xu and Raul Puri and Alec Radford and Jesse"""
            """ Michael Han and Jerry Tworek and Qiming Yuan and Nikolas A. Tezak and"""
            """ Jong Wook Kim and Chris Hallacy and Johannes Heidecke and Pranav Shyam"""
            """ and Boris Power and Tyna Eloundou Nekoul and Girish Sastry and"""
            """ Gretchen Krueger and David P. Schnurr and Felipe Petroski Such and"""
            """ Kenny Sai-Kin Hsu and Madeleine Thompson and Tabarak Khan and Toki"""
            """ Sherbakov and Joanne Jang and Peter Welinder and Lilian Weng},
  journal={ArXiv},
  year={2022},
  volume={abs/2201.10005},
  url={https://api.semanticscholar.org/CorpusID:246275593}
}
            """.strip()
        )
        return citations

    @property
    def version(self) -> float:
        return 1.0

    @cached_property
    def display_name(self) -> str:
        return (
            super().display_name + f" ({self.model_name}) ({self._dimensions})"
            if not isinstance(self._dimensions, Default)
            else super().display_name + f" ({self.model_name})"
        )

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        if not isinstance(self._dimensions, Default):
            names.append(str(self._dimensions))
        to_hash: list[Any] = []
        if len(to_hash) > 0:  # pragma: no cover
            names.append(Hasher.hash(to_hash))
        return "_".join(names)

    def unload_model(self):
        # Delete cached model and tokenizer
        if "client" in self.__dict__:
            del self.__dict__["client"]
        if "tokenizer" in self.__dict__:
            del self.__dict__["tokenizer"]

        # Garbage collect
        gc.collect()

    def __getstate__(self):  # pragma: no cover
        state = super().__getstate__()

        # Remove cached client or tokenizer before serializing
        state.pop("retry_wrapper", None)
        state.pop("client", None)
        state.pop("tokenizer", None)

        return state


__all__ = ["OpenAIEmbedder"]
