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

from ..utils.arg_utils import AUTO, Default
from ..utils.hf_hub_utils import (
    _has_file,
    get_citation_info,
    get_license_info,
    get_model_card_url,
)
from ..utils.import_utils import import_module
from ._llm_api import LLMAPI
from ._tokenizers import TOGETHER_TOKENIZERS
from .llm import (
    DEFAULT_BATCH_SIZE,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)


def _get_final_tokenizer_model_name(
    model_name: str, tokenizer_model_name: str | None
) -> str | None:
    if _has_file(repo_id=model_name, filename="config.json", repo_type="model"):
        default_tokenizer_model_name = model_name
    elif model_name in TOGETHER_TOKENIZERS:
        default_tokenizer_model_name = TOGETHER_TOKENIZERS[model_name]
    else:
        default_tokenizer_model_name = None
    final_tokenizer_model_name = tokenizer_model_name or default_tokenizer_model_name
    return final_tokenizer_model_name


class Together(LLMAPI):
    def __init__(
        self,
        model_name: str,
        chat_prompt_template: None | str | Default = AUTO,
        system_prompt: None | str | Default = AUTO,
        api_key: None | str = None,
        max_context_length: None | int = None,
        tokenizer_model_name: str | None = None,
        tokenizer_revision: None | str = None,
        tokenizer_trust_remote_code: bool = False,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        final_tokenizer_model_name = _get_final_tokenizer_model_name(
            model_name=model_name, tokenizer_model_name=tokenizer_model_name
        )
        super().__init__(
            model_name=model_name,
            chat_prompt_template=chat_prompt_template,
            system_prompt=system_prompt,
            max_context_length=max_context_length,
            tokenizer_model_name=final_tokenizer_model_name,
            tokenizer_revision=tokenizer_revision,
            tokenizer_trust_remote_code=tokenizer_trust_remote_code,
            retry_on_fail=retry_on_fail,
            cache_folder_path=cache_folder_path,
            warn_max_context_length=(
                final_tokenizer_model_name is None
                or (
                    final_tokenizer_model_name == tokenizer_model_name
                    and final_tokenizer_model_name != model_name
                )
            ),
            **kwargs,
        )
        self.api_key = api_key

    @cached_property
    def retry_wrapper(self):
        together = import_module("together")
        requests = import_module("requests")

        # Create a retry wrapper function
        tenacity_logger = self.get_logger(key="retry", verbose=True, log_level=None)

        @retry(
            retry=retry_if_exception_type(together.RateLimitError),
            wait=wait_exponential(multiplier=1, min=10, max=60),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(together.ResponseError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(requests.exceptions.HTTPError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(together.TogetherException),
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
        together = import_module("together")
        if self.api_key:
            together.api_key = self.api_key  # type: ignore[attr-defined]
        return together

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
        assert n == 1, f"Only `n` = 1 is supported for {type(self).__name__}"

        # Check max_new_tokens
        max_new_tokens = _check_max_new_tokens_possible(
            self=self,
            max_length_func=max_length_func,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )

        # Set temperature and top_p
        temperature, top_p = _check_temperature_and_top_p(
            temperature=temperature, top_p=top_p, supports_zero_temperature=False
        )

        # Run the model
        def get_generated_texts(self, kwargs, prompt) -> list[str]:
            response = self.retry_wrapper(
                func=self.client.Complete.create,
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                stop=stop or [],
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            return [choice["text"].strip() for choice in response["output"]["choices"]]

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
        return (
            get_model_card_url(self.model_name)
            or "https://docs.together.ai/docs/inference-models"
        )

    @cached_property
    def license(self) -> None | str:
        return (
            get_license_info(self.model_name, repo_type="model")
            or "https://together.ai/terms-of-service"
        )

    @cached_property
    def citation(self) -> None | list[str]:
        return get_citation_info(self.model_name, repo_type="model")


__all__ = ["Together"]
