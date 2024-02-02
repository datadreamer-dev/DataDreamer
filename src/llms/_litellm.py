import logging
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

from ..utils import ring_utils as ring
from ..utils.import_utils import ignore_litellm_warnings
from ._llm_api import LLMAPI
from .llm import (
    DEFAULT_BATCH_SIZE,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)


class LiteLLM(LLMAPI):
    def __init__(
        self,
        model_name: str,
        api_key: None | str = None,
        aws_access_key_id: None | str = None,
        aws_secret_access_key: None | str = None,
        aws_region_name: None | str = None,
        vertex_project: None | str = None,
        vertex_location: None | str = None,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            retry_on_fail=retry_on_fail,
            cache_folder_path=cache_folder_path,
            warn_tokenizer_model_name=False,
            warn_max_context_length=False,
            **kwargs,
        )
        self._model_name_prefix = ""
        self.api_key = api_key
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region_name = aws_region_name
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location

    @cached_property
    def retry_wrapper(self):
        with ignore_litellm_warnings():
            from litellm.exceptions import (
                APIConnectionError,
                APIError,
                RateLimitError,
                ServiceUnavailableError,
            )

        # Create a retry wrapper function
        tenacity_logger = self.get_logger(key="retry", verbose=True, log_level=None)

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential(multiplier=1, min=10, max=60),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(ServiceUnavailableError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(APIError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(APIConnectionError),
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
    def client(self) -> Any:
        with ignore_litellm_warnings():
            from litellm import completion

        return completion

    @ring.lru(maxsize=128)
    def get_max_context_length(self, max_new_tokens: int) -> int:
        """Gets the maximum context length for the model. When ``max_new_tokens`` is
        greater than 0, the maximum number of tokens that can be used for the prompt
        context is returned.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.

        Returns:
            The maximum context length.
        """  # pragma: no cover
        with ignore_litellm_warnings():
            from litellm import get_max_tokens

        return get_max_tokens(model=self._model_name_prefix + self.model_name)

    @ring.lru(maxsize=5000)
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        with ignore_litellm_warnings():
            from litellm import token_counter

        with ignore_litellm_warnings():
            return token_counter(
                self._model_name_prefix + self.model_name,
                messages=[{"user": "role", "content": value}],
            )

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
        assert seed is None, f"`seed` is not supported for {type(self).__name__}"
        assert (
            logit_bias is None
        ), f"`logit_bias` is not supported for {type(self).__name__}"

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

        # Setup optional kwargs
        optional_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            max_tokens=max_new_tokens,
            presence_penalty=repetition_penalty,
            api_key=self.api_key,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_region_name=self.aws_region_name,
            vertex_project=self.vertex_project,
            vertex_location=self.vertex_location,
        )
        optional_kwargs = {
            kw: optional_kwargs[kw]
            for kw in optional_kwargs
            if optional_kwargs[kw] is not None
        }

        # Run the model
        def get_generated_texts(self, kwargs, prompt) -> list[str]:
            response = self.retry_wrapper(
                func=self.client,
                model=self._model_name_prefix + self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **optional_kwargs,
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
        else:
            return generated_texts_batch


__all__ = ["LiteLLM"]
