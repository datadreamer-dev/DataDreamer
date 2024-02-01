import gc
import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, partial
from typing import Any, Callable, Generator, Iterable

from datasets.fingerprint import Hasher
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_any,
    wait_exponential,
)

from .._cachable._cachable import _StrWithSeed
from ..utils.arg_utils import AUTO, Default
from ..utils.fs_utils import safe_fn
from ..utils.import_utils import ignore_pydantic_warnings
from .hf_transformers import CachedTokenizer, HFTransformers
from .llm import (
    DEFAULT_BATCH_SIZE,
    LLM,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)

with ignore_pydantic_warnings():
    from huggingface_hub import InferenceClient


class HFAPIEndpoint(HFTransformers):
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        chat_prompt_template: None | str | Default = AUTO,
        system_prompt: None | str | Default = AUTO,
        token: None | str = None,
        revision: None | str = None,
        trust_remote_code: bool = False,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            chat_prompt_template=chat_prompt_template,
            system_prompt=system_prompt,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_folder_path=cache_folder_path,
            **kwargs,
        )
        self.endpoint = endpoint
        self.token = token

        # Setup API calling helpers
        self.retry_on_fail = retry_on_fail
        self.executor_pools: dict[int, ThreadPoolExecutor] = {}

    @cached_property
    def retry_wrapper(self):
        # Create a retry wrapper function
        tenacity_logger = self.get_logger(key="retry", verbose=True, log_level=None)

        @retry(
            retry=retry_if_exception_type(Exception),
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
    def client(self) -> InferenceClient:
        return InferenceClient(model=self.endpoint, token=self.token, **self.kwargs)

    def _is_batch_size_exception(self, e: BaseException) -> bool:  # pragma: no cover
        return False

    def _run_batch(
        self,
        cached_tokenizer: CachedTokenizer,
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
            temperature=temperature,
            top_p=top_p,
            supports_zero_temperature=False,
            supports_zero_top_p=False,
            supports_one_top_p=False,
        )

        # Run the model
        def get_generated_texts(self, kwargs, prompt) -> list[str]:
            generated_text = self.retry_wrapper(
                func=self.client.text_generation,
                model=self.endpoint,
                prompt=prompt,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                return_full_text=False,
                seed=(
                    seed + _StrWithSeed.total_per_input_seeds([prompt])
                    if seed is not None
                    else None
                ),
                stop_sequences=stop,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            return [generated_text]

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

    def run(
        self,
        prompts: Iterable[str],
        max_new_tokens: None | int = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        n: int = 1,
        stop: None | str | list[str] = None,
        repetition_penalty: None | float = None,
        logit_bias: None | dict[int, float] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = False,
        seed: None | int = None,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_prompts: None | int = None,
        return_generator: bool = False,
        **kwargs,
    ) -> Generator[str | list[str], None, None] | list[str | list[str]]:
        return super().run(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stop=stop,
            repetition_penalty=repetition_penalty,
            logit_bias=logit_bias,
            batch_size=batch_size,
            batch_scheduler_buffer_size=batch_scheduler_buffer_size,
            adaptive_batch_size=adaptive_batch_size,
            seed=seed,
            progress_interval=progress_interval,
            force=force,
            cache_only=cache_only,
            verbose=verbose,
            log_level=log_level,
            total_num_prompts=total_num_prompts,
            return_generator=return_generator,
            **kwargs,
        )

    @cached_property
    def display_name(self) -> str:
        name = (
            re.sub(r"/(.*)", r"\1", urllib.parse.urlparse(self.endpoint).path).strip()
            or self.endpoint
        )
        return LLM.display_name.func(self) + f" ({name})"  # type: ignore[attr-defined]

    @cached_property
    def _cache_name(self) -> None | str:
        names = [
            safe_fn(self.endpoint, allow_slashes=False),
            safe_fn(self.model_name, allow_slashes=False),
        ]
        if self.revision:
            names.append(self.revision)
        to_hash: list[Any] = []
        if len(to_hash) > 0:  # pragma: no cover
            names.append(Hasher.hash(to_hash))
        return "_".join(names)

    def unload_model(self):
        # Delete cached client and tokenizer
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
        state["executor_pools"].clear()

        return state


__all__ = ["HFAPIEndpoint"]
