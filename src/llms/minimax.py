import os
from functools import cached_property
from typing import Any, Callable

import openai

from ..utils import ring_utils as ring
from .llm import (
    DEFAULT_BATCH_SIZE,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)
from .openai import OpenAI

# MiniMax model context lengths and max output lengths
_MINIMAX_CONTEXT_LENGTHS: dict[str, int] = {
    "MiniMax-M2.7": 1000000,
    "MiniMax-M2.7-highspeed": 1000000,
    "MiniMax-M2.5": 204800,
    "MiniMax-M2.5-highspeed": 204800,
}

_MINIMAX_MAX_OUTPUT_LENGTHS: dict[str, int] = {
    "MiniMax-M2.7": 16384,
    "MiniMax-M2.7-highspeed": 16384,
    "MiniMax-M2.5": 8192,
    "MiniMax-M2.5-highspeed": 8192,
}

_MINIMAX_BASE_URL = "https://api.minimax.io/v1"


class MiniMax(OpenAI):
    """A MiniMax LLM provider that uses MiniMax's OpenAI-compatible API.

    MiniMax provides large language models accessible via an OpenAI-compatible
    API endpoint. Supported models include ``MiniMax-M2.7``,
    ``MiniMax-M2.7-highspeed``, ``MiniMax-M2.5``, and
    ``MiniMax-M2.5-highspeed``.

    Args:
        model_name: The name of the MiniMax model to use.
        system_prompt: An optional system prompt to use.
        api_key: The MiniMax API key. If ``None``, the ``MINIMAX_API_KEY``
            environment variable will be used.
        retry_on_fail: Whether to retry on failure.
        cache_folder_path: The path to the cache folder.
        **kwargs: Additional keyword arguments passed to the OpenAI client.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: None | str = None,
        api_key: None | str = None,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt or "You are a helpful assistant.",
            api_key=api_key or os.environ.get("MINIMAX_API_KEY"),
            base_url=_MINIMAX_BASE_URL,
            retry_on_fail=retry_on_fail,
            cache_folder_path=cache_folder_path,
            **kwargs,
        )

    @cached_property
    def client(self) -> openai.OpenAI:
        other_kwargs: dict[str, Any] = {}
        if self.api_key:
            other_kwargs["api_key"] = self.api_key
        return openai.OpenAI(
            base_url=_MINIMAX_BASE_URL,
            **other_kwargs,
            **self.kwargs,
        )

    @ring.lru(maxsize=128)
    def get_max_context_length(self, max_new_tokens: int) -> int:
        """Gets the maximum context length for the model.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.

        Returns:
            The maximum context length.
        """
        # Use known context lengths for MiniMax models
        max_context_length = _MINIMAX_CONTEXT_LENGTHS.get(self.model_name, 204800)
        # Account for chat format tokens (system prompt + message framing)
        format_tokens = 4 * 3 + self.count_tokens(self.system_prompt or "")
        return max_context_length - max_new_tokens - format_tokens

    def _get_max_output_length(self) -> None | int:
        return _MINIMAX_MAX_OUTPUT_LENGTHS.get(self.model_name, 8192)

    def _run_batch(
        self,
        max_length_func: Callable[[list[str]], int],
        inputs: list[str],
        max_new_tokens: None | int = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        n: int = 1,
        stop: None | str | list[str] = None,
        repetition_penalty: None | float = None,
        logit_bias: None | dict[int, float] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: None | int = None,
        **kwargs,
    ) -> list[str] | list[list[str]]:
        # MiniMax requires temperature in (0.0, 1.0]
        if temperature == 0.0:
            temperature = 0.01
        elif temperature > 1.0:
            temperature = 1.0

        return super()._run_batch(
            max_length_func=max_length_func,
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            repetition_penalty=repetition_penalty,
            logit_bias=logit_bias,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )

    @cached_property
    def model_card(self) -> None | str:
        return "https://platform.minimaxi.com/document/Models"

    @cached_property
    def license(self) -> None | str:
        return "https://platform.minimaxi.com/document/Terms%20of%20service"

    @cached_property
    def citation(self) -> None | list[str]:
        return None


__all__ = ["MiniMax"]
