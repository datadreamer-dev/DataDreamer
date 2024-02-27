import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, partial
from typing import Any, Callable

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_any,
    wait_exponential,
)

from .._cachable._cachable import _StrWithSeed
from ..utils.import_utils import import_module
from ._llm_api import LLMAPI
from .llm import (
    DEFAULT_BATCH_SIZE,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)


class MistralAI(LLMAPI):
    def __init__(
        self,
        model_name: str,
        api_key: None | str = None,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            chat_prompt_template=None,
            system_prompt=None,
            max_context_length=(
                # Today, all Mistral models on the API support 32k
                # - 10 for template tokens
                32768 - 10
            ),
            tokenizer_model_name="mistralai/Mistral-7B-Instruct-v0.1",
            tokenizer_revision=None,
            tokenizer_trust_remote_code=False,
            retry_on_fail=retry_on_fail,
            cache_folder_path=cache_folder_path,
            warn_max_context_length=False,
            **kwargs,
        )
        self.api_key = api_key

    @cached_property
    def retry_wrapper(self):
        MistralException, MistralAPIException, MistralConnectionException = (
            import_module("mistralai.exceptions").MistralException,
            import_module("mistralai.exceptions").MistralAPIException,
            import_module("mistralai.exceptions").MistralConnectionException,
        )

        # Create a retry wrapper function
        tenacity_logger = self.get_logger(key="retry", verbose=True, log_level=None)

        @retry(
            retry=retry_if_exception_type(MistralException),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(MistralAPIException),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(MistralConnectionException),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        def _retry_wrapper(func, **kwargs):
            return func(**kwargs)

        _retry_wrapper.__wrapped__.__module__ = None  # type: ignore[attr-defined]
        _retry_wrapper.__wrapped__.__qualname__ = f"{self.__class__.__name__}.run"  # type: ignore[attr-defined]
        return _retry_wrapper

    @cached_property
    def client(self) -> Any:  # pragma: no cover
        MistralClient = import_module("mistralai.client").MistralClient
        mistral = MistralClient(
            api_key=self.api_key or os.environ.get("MISTRAL_API_KEY", None)
        )
        return mistral

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
        prompts = inputs
        assert (
            stop is None or stop == []
        ), f"`stop` is not supported for {type(self).__name__}"
        assert (
            repetition_penalty is None
        ), f"`repetition_penalty` is not supported for {type(self).__name__}"
        assert (
            logit_bias is None
        ), f"`logit_bias` is not supported for {type(self).__name__}"
        assert n == 1, f"Only `n` = 1 is supported for {type(self).__name__}"

        ChatMessage = import_module("mistralai.models.chat_completion").ChatMessage

        # Check max_new_tokens
        max_new_tokens = _check_max_new_tokens_possible(
            self=self,
            max_length_func=max_length_func,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )

        # Set temperature and top_p
        temperature, top_p = _check_temperature_and_top_p(
            temperature=temperature,
            top_p=top_p,
            supports_zero_temperature=False,
            supports_zero_top_p=False,
        )

        # Run the model
        def get_generated_texts(self, kwargs, prompt) -> list[str]:
            messages = [ChatMessage(role="user", content=prompt)]
            response = self.retry_wrapper(
                func=self.client.chat,
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                random_seed=(
                    seed + _StrWithSeed.total_per_input_seeds([prompt])
                    if seed is not None
                    else None
                ),
                safe_mode=kwargs.pop("safe_mode", False),
                **kwargs,
            )
            return [choice.message.content.strip() for choice in response.choices]

        if batch_size not in self.executor_pools:
            self.executor_pools[batch_size] = ThreadPoolExecutor(max_workers=batch_size)
        generated_texts_batch = list(
            self.executor_pools[batch_size].map(
                partial(get_generated_texts, self, kwargs), prompts
            )
        )

        if n == 1:
            return [batch[0] for batch in generated_texts_batch]
        else:  # pragma: no cover
            return generated_texts_batch

    @cached_property
    def model_card(self) -> None | str:
        return "https://docs.mistral.ai/models"

    @cached_property
    def license(self) -> None | str:
        return "https://mistral.ai/terms-of-use/"

    @cached_property
    def citation(self) -> None | list[str]:  # pragma: no cover
        return [
            """@article{jiang2023mistral,
  title={Mistral 7B},
  author={Jiang, Albert Q and Sablayrolles, Alexandre and Mensch, Arthur and"""
            """ Bamford, Chris and Chaplot, Devendra Singh and Casas, Diego de las and"""
            """ Bressand, Florian and Lengyel, Gianna and Lample, Guillaume and Saulnier,"""
            """ Lucile and others},
  journal={arXiv preprint arXiv:2310.06825},
  year={2023}
}"""
        ]


__all__ = ["MistralAI"]
