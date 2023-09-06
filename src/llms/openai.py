import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import openai
import tiktoken
from tenacity import (after_log, before_sleep_log, retry,
                      retry_if_exception_type, wait_exponential)

from datasets.fingerprint import Hasher

from ..utils.fs_utils import safe_fn
from .llm import LLM

logger = logging.getLogger("datadreamer.llms.openai")


def _is_chat_model(model_name: str):
    return "gpt-3.5-" in model_name or "gpt-35-" in model_name or "gpt-4-" in model_name


@retry(
    retry=retry_if_exception_type(openai.error.RateLimitError),
    wait=wait_exponential(multiplier=1, min=10, max=60),
    before_sleep=before_sleep_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
)
@retry(
    retry=retry_if_exception_type(openai.error.ServiceUnavailableError),
    wait=wait_exponential(multiplier=1, min=3, max=10),
    before_sleep=before_sleep_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
)
@retry(
    retry=retry_if_exception_type(openai.error.APIError),
    wait=wait_exponential(multiplier=1, min=3, max=10),
    before_sleep=before_sleep_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
)
@retry(
    retry=retry_if_exception_type(openai.error.APIConnectionError),
    wait=wait_exponential(multiplier=1, min=3, max=10),
    before_sleep=before_sleep_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
)
def openai_retry_wrapper(func, **kwargs):
    return func(**kwargs)


class OpenAI(LLM):
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "You are a helpful assistant.",
        organization: None | str = None,
        api_key: None | str = None,
    ):
        super().__init__()
        self.model_name = model_name
        if organization:  # pragma: no cover
            openai.organization = organization
        if api_key:  # pragma: no cover
            openai.api_key = api_key
        self.system_prompt = system_prompt
        self.executor_pools = {}

    def get_max_context_length(self, max_new_tokens: int) -> int:  # pragma: no cover
        format_tokens = 0
        if _is_chat_model(self.model_name):
            # Each message is up to 4 tokens and there are 3 messages
            # (system prompt, user prompt, assistant response)
            format_tokens = 4 * 3
        if "32k" in self.model_name:
            max_context_length = 32768
        elif "16k" in self.model_name:
            max_context_length = 16384
        elif "gpt-3.5-turbo" in self.model_name or "gpt-35-turbo" in self.model_name:
            max_context_length = 4096
        elif self.model_name.startswith("text-davinci"):
            max_context_length = 4097
        elif self.model_name.startswith("code-davinci"):
            max_context_length = 8001
        elif any(
            self.model_name.startswith(prefix)
            for prefix in ["text-curie", "text-babbage", "text-ada"]
        ) or self.model_name in ["ada", "babbage", "curie", "davinci"]:
            max_context_length = 2049
        else:
            max_context_length = 8192
        return max_context_length - max_new_tokens - format_tokens

    def count_tokens(self, value: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model_name)
        return len(encoding.encode(value))

    def _run_batch(
        self,
        args_cache_key: str,
        prompts: list[str],
        max_new_tokens: int = 999999999999999999,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stop: None | str | list[str] = None,
        repetition_penalty: None | float = None,
        logit_bias: None | dict[int, float] = None,
        batch_size: int = 10,
    ) -> list[str] | list[list[str]]:
        optional_kwargs = dict(
            stop=stop,
            presence_penalty=repetition_penalty,
            logit_bias=logit_bias,
        )
        optional_kwargs = {
            kw: optional_kwargs[kw]
            for kw in optional_kwargs
            if optional_kwargs[kw] is not None
        }
        if _is_chat_model(self.model_name):

            def get_generated_texts(prompt):
                messages = [
                    {"role": "system", "content": f"{self.system_prompt}"},
                    {"role": "user", "content": prompt},
                ]
                response = openai_retry_wrapper(
                    func=openai.ChatCompletion.create,
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    n=n,
                    **optional_kwargs,
                )
                return [
                    choice.message["content"].strip() for choice in response["choices"]
                ]

            if batch_size not in self.executor_pools:
                self.executor_pools[batch_size] = ThreadPoolExecutor(
                    max_workers=batch_size
                )
            generated_texts_batch = self.executor_pools[batch_size].map(
                get_generated_texts, prompts
            )
        else:
            response = openai_retry_wrapper(
                func=openai.Completion.create,
                engine=self.model_name,
                prompt=prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                n=n,
                **optional_kwargs,
            )
            texts = [choice["text"].strip() for choice in response.choices]
            generated_texts_batch = [
                list(batch)
                for batch in zip(*(iter(texts),) * (len(texts) // len(prompts)))
            ]
        if n == 1:
            results = [batch[0] for batch in generated_texts_batch]
        else:
            results = generated_texts_batch
        if self.get_cache():
            cache, lock = self.get_cache()
            with lock:
                for prompt, result in zip(prompts, results):
                    cache_key = self._compute_cache_key(args_cache_key, prompt)
                    cache[cache_key] = result
                cache.commit()
        return results

    def run(
        self,
        prompts: list[str],
        max_new_tokens: int = 999999999999999999,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stop: None | str | list[str] = None,
        repetition_penalty: None | float = None,
        logit_bias: None | dict[int, float] = None,
        batch_size: int = 10,
    ) -> list[str] | list[list[str]]:
        args_cache_key = Hasher.hash(
            dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                repetition_penalty=repetition_penalty,
                logit_bias=logit_bias,
            )
        )
        prompts_iter = iter(prompts)
        generated_texts = []
        while True:
            prompts_batch = list(islice(prompts_iter, batch_size))
            if len(prompts_batch) == 0:
                break
            generated_texts.extend(
                self._run_batch(
                    args_cache_key=args_cache_key,
                    prompts=prompts_batch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                    repetition_penalty=repetition_penalty,
                    logit_bias=logit_bias,
                    batch_size=batch_size,
                )
            )
        return generated_texts

    @property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        to_hash = [self.system_prompt]
        names.append(Hasher.hash(to_hash))
        return "_".join(names)


__all__ = ["OpenAI"]
