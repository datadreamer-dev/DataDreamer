import gc
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, lru_cache, partial
from typing import Any, Callable, Generator, Iterable, cast

import openai
import tiktoken
from datasets.fingerprint import Hasher
from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_any,
    wait_exponential,
)
from tiktoken import Encoding

from ..utils import ring_utils as ring
from ..utils.fs_utils import safe_fn
from .llm import (
    DEFAULT_BATCH_SIZE,
    LLM,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)


@lru_cache(maxsize=None)
def _normalize_model_name(model_name: str) -> str:
    if ":" in model_name:  # pragma: no cover
        # Handles extracting the model name from a fine-tune
        # model name like "ft:babbage-002:org:datadreamer:xxxxxxx"
        model_name = model_name.split(":")[1]
    return model_name


@lru_cache(maxsize=None)
def _is_gpt_3(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(
        gpt3_name in model_name for gpt3_name in ["davinci", "ada", "curie", "gpt-3-"]
    )


@lru_cache(maxsize=None)
def _is_gpt_3_5(model_name: str):
    model_name = _normalize_model_name(model_name)
    return any(gpt35_name in model_name for gpt35_name in ["gpt-3.5-", "gpt-35-"])


@lru_cache(maxsize=None)
def _is_gpt_3_5_legacy(model_name: str):
    model_name = _normalize_model_name(model_name)
    return _is_gpt_3_5(model_name) and (
        "-0613" in model_name
        or (_is_instruction_tuned(model_name) and not _is_chat_model(model_name))
    )


@lru_cache(maxsize=None)
def _is_gpt_4(model_name: str):
    model_name = _normalize_model_name(model_name)
    return model_name == "gpt-4" or any(
        gpt4_name in model_name for gpt4_name in ["gpt-4-"]
    )


@lru_cache(maxsize=None)
def _is_preview_model(model_name: str):
    model_name = _normalize_model_name(model_name)
    return "-preview" in model_name


@lru_cache(maxsize=None)
def _is_chat_model(model_name: str):
    model_name = _normalize_model_name(model_name)
    return (
        _is_gpt_3_5(model_name) or _is_gpt_4(model_name)
    ) and not model_name.endswith("-instruct")


@lru_cache(maxsize=None)
def _is_instruction_tuned(model_name: str):
    model_name = _normalize_model_name(model_name)
    return (
        _is_chat_model(model_name)
        or model_name.startswith("text-")
        or model_name.endswith("-instruct")
    )


class OpenAIException(Exception):
    pass


class OpenAI(LLM):
    def __init__(
        self,
        model_name: str,
        system_prompt: None | str = None,
        organization: None | str = None,
        api_key: None | str = None,
        base_url: None | str = None,
        api_version: None | str = None,
        retry_on_fail: bool = True,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(cache_folder_path=cache_folder_path)
        self.model_name = model_name
        self.organization = organization
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.kwargs = kwargs
        self.system_prompt = system_prompt
        if self.system_prompt is None and _is_chat_model(self.model_name):
            self.system_prompt = "You are a helpful assistant."

        # Setup API calling helpers
        self.retry_on_fail = retry_on_fail
        self.executor_pools: dict[int, ThreadPoolExecutor] = {}

    @cached_property
    def retry_wrapper(self):
        # Create a retry wrapper function
        tenacity_logger = self.get_logger(key="retry", verbose=True, log_level=None)

        @retry(
            retry=retry_if_exception_type(openai.RateLimitError),
            wait=wait_exponential(multiplier=1, min=10, max=60),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(openai.InternalServerError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(openai.APIError),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(OpenAIException),
            wait=wait_exponential(multiplier=1, min=3, max=300),
            before_sleep=before_sleep_log(tenacity_logger, logging.INFO),
            after=after_log(tenacity_logger, logging.INFO),
            stop=stop_any(lambda _: not self.retry_on_fail),  # type: ignore[arg-type]
            reraise=True,
        )
        @retry(
            retry=retry_if_exception_type(openai.APIConnectionError),
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
    def client(self) -> openai.OpenAI | openai.AzureOpenAI:
        other_kwargs: dict[str, Any] = {
            "max_retries": sys.maxsize if self.retry_on_fail else 0
        }
        is_azure = self.base_url and "azure.com" in self.base_url
        if self.organization:  # pragma: no cover
            other_kwargs["organization"] = self.organization
        if self.api_key:  # pragma: no cover
            other_kwargs["api_key" if not is_azure else "azure_ad_token"] = self.api_key
        if self.base_url:  # pragma: no cover
            other_kwargs["base_url"] = self.base_url
        if self.api_version and is_azure:  # pragma: no cover
            other_kwargs["api_version"] = self.api_version
        return (
            openai.OpenAI(**other_kwargs, **self.kwargs)
            if not is_azure
            else openai.AzureOpenAI(**other_kwargs, **self.kwargs)
        )

    @cached_property
    def tokenizer(self) -> Encoding:
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:  # pragma: no cover
            return tiktoken.get_encoding("cl100k_base")

    @ring.lru(maxsize=128)
    def get_max_context_length(self, max_new_tokens: int) -> int:  # pragma: no cover
        """Gets the maximum context length for the model. When ``max_new_tokens`` is
        greater than 0, the maximum number of tokens that can be used for the prompt
        context is returned.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.

        Returns:
            The maximum context length.
        """  # pragma: no cover
        model_name = _normalize_model_name(self.model_name)
        format_tokens = 0
        if _is_chat_model(model_name):
            # Each message is up to 4 tokens and there are 3 messages
            # (system prompt, user prompt, assistant response)
            # and then we have to account for the system prompt
            format_tokens = 4 * 3 + self.count_tokens(cast(str, self.system_prompt))
        if "32k" in model_name:
            max_context_length = 32768
        elif "16k" in model_name:
            max_context_length = 16384
        elif _is_preview_model(model_name):
            max_context_length = 128000
        elif _is_gpt_3_5(self.model_name):
            if _is_gpt_3_5_legacy(self.model_name):
                max_context_length = 4096
            else:
                max_context_length = 16385
        elif model_name.startswith("text-davinci"):
            max_context_length = 4097
        elif model_name.startswith("code-davinci"):
            max_context_length = 8001
        elif any(
            model_name.startswith(prefix)
            for prefix in ["text-curie", "text-babbage", "text-ada"]
        ) or model_name in ["ada", "babbage", "curie", "davinci"]:
            max_context_length = 2049
        else:
            max_context_length = 8192
        return max_context_length - max_new_tokens - format_tokens

    def _get_max_output_length(self) -> None | int:  # pragma: no cover
        if (_is_gpt_4(self.model_name) and _is_preview_model(self.model_name)) or (
            _is_gpt_3_5(self.model_name) and not (_is_gpt_3_5_legacy(self.model_name))
        ):
            return 4096
        return None

    @ring.lru(maxsize=5000)
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        return len(self.tokenizer.encode(value))

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

        # Check max_new_tokens
        max_new_tokens = _check_max_new_tokens_possible(
            self=self,
            max_length_func=max_length_func,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )

        # Set temperature and top_p
        temperature, top_p = _check_temperature_and_top_p(
            temperature=temperature, top_p=top_p
        )

        # Run the model
        optional_kwargs = dict(
            stop=stop,
            presence_penalty=repetition_penalty,
            logit_bias=logit_bias,
            seed=seed,
        )
        optional_kwargs = {
            kw: optional_kwargs[kw]
            for kw in optional_kwargs
            if optional_kwargs[kw] is not None
        }
        if _is_chat_model(self.model_name):

            def get_generated_texts(self, kwargs, optional_kwargs, prompt) -> list[str]:
                kwargs = kwargs.copy()
                messages = [
                    {"role": "system", "content": f"{kwargs['system_prompt']}"},
                    {"role": "user", "content": prompt},
                ]
                del kwargs["system_prompt"]
                response = self.retry_wrapper(
                    func=self.client.chat.completions.create,
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    n=n,
                    **optional_kwargs,
                    **kwargs,
                )
                return [choice.message.content.strip() for choice in response.choices]

            if batch_size not in self.executor_pools:
                self.executor_pools[batch_size] = ThreadPoolExecutor(
                    max_workers=batch_size
                )
            generated_texts_batch = list(
                self.executor_pools[batch_size].map(
                    partial(get_generated_texts, self, kwargs, optional_kwargs), prompts
                )
            )
        else:
            response = self.retry_wrapper(
                func=self.client.completions.create,
                model=self.model_name,
                prompt=prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=n,
                **optional_kwargs,
            )
            texts = cast(
                list[str], [choice.text.strip() for choice in response.choices]
            )
            generated_texts_batch = [
                list(batch)
                for batch in zip(*(iter(texts),) * (len(texts) // len(prompts)))
            ]
        if n == 1:
            return [batch[0] for batch in generated_texts_batch]
        else:
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
        if _is_chat_model(self.model_name):
            kwargs["system_prompt"] = self.system_prompt

        def get_max_length_function() -> dict[str, Any]:
            def max_length_func(prompts: list[str]) -> int:
                return max([self.count_tokens(p) for p in prompts])

            return {"max_length_func": max_length_func}

        results_generator = self._run_over_batches(
            run_batch=self._run_batch,
            get_max_input_length_function=get_max_length_function,
            max_model_length=self.get_max_context_length,
            inputs=prompts,
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
            total_num_inputs=total_num_prompts,
            **kwargs,
        )
        if not return_generator:
            return list(results_generator)
        else:
            return results_generator

    @cached_property
    def model_card(self) -> None | str:
        if _is_gpt_3(self.model_name) or _is_gpt_3_5(self.model_name):
            return (
                "https://github.com/openai/gpt-3/blob/"
                "d7a9bb505df6f630f9bab3b30c889e52f22eb9ea/model-card.md"
            )
        if _is_gpt_4(self.model_name):
            return "https://cdn.openai.com/papers/gpt-4-system-card.pdf"
        return None  # pragma: no cover

    @cached_property
    def license(self) -> None | str:
        return "https://openai.com/policies"

    @cached_property
    def citation(self) -> None | list[str]:
        citations = []
        if _is_gpt_3(self.model_name) or _is_gpt_3_5(self.model_name):
            citations.append(
                """
@article{brown2020language,
  title={Language models are few-shot learners},
  author={Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and"""
                """ Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam,"""
                """ Pranav and Sastry, Girish and Askell, Amanda and others},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={1877--1901},
  year={2020}
}
                """.strip()
            )
        if _is_gpt_4(self.model_name):
            citations.append(
                """
@article{OpenAI2023GPT4TR,
  title={GPT-4 Technical Report},
  author={OpenAI},
  journal={ArXiv},
  year={2023},
  volume={abs/2303.08774},
  url={https://api.semanticscholar.org/CorpusID:257532815}
}
                """.strip()
            )
        if _is_instruction_tuned(self.model_name):
            citations.append(
                """
@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeffrey and Jiang, Xu and Almeida, Diogo and"""
                """ Wainwright, Carroll and Mishkin, Pamela and Zhang, Chong and"""
                """ Agarwal, Sandhini and Slama, Katarina and Ray, Alex and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={27730--27744},
  year={2022}
}
                """.strip()
            )
        return citations

    @cached_property
    def display_name(self) -> str:
        return super().display_name + f" ({self.model_name})"

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        to_hash: list[Any] = []
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


__all__ = ["OpenAI"]
