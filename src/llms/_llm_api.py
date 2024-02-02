import gc
import logging
import warnings
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
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

from ..utils import ring_utils as ring
from ..utils.arg_utils import AUTO, Default
from ..utils.fs_utils import safe_fn
from ..utils.hf_hub_utils import _has_file
from ..utils.hf_model_utils import (
    get_config,
    get_model_max_context_length,
    get_model_prompt_template,
    get_tokenizer,
    is_encoder_decoder,
)
from ..utils.import_utils import ignore_transformers_warnings
from ._tokenizers import _model_name_to_tokenizer_model_name
from .llm import DEFAULT_BATCH_SIZE, LLM

with ignore_transformers_warnings():
    from transformers import PretrainedConfig, PreTrainedTokenizer


class LLMAPI(LLM):
    def __init__(
        self,
        model_name: str,
        chat_prompt_template: None | str | Default = AUTO,
        system_prompt: None | str | Default = AUTO,
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
        super().__init__(cache_folder_path=cache_folder_path)
        self.model_name = model_name
        self.chat_prompt_template, self.system_prompt = get_model_prompt_template(
            model_name=self.model_name,
            revision=None,
            chat_prompt_template=chat_prompt_template,
            system_prompt=system_prompt,
        )
        self.max_context_length = max_context_length
        self._tokenizer_model_name = tokenizer_model_name
        self.tokenizer_revision = tokenizer_revision
        self.tokenizer_trust_remote_code = tokenizer_trust_remote_code
        self.warn_tokenizer_model_name = warn_tokenizer_model_name
        self.warn_max_context_length = warn_max_context_length
        self._warned_max_context_length = False
        self.kwargs = kwargs

        # Setup API calling helpers
        self.retry_on_fail = retry_on_fail
        self.executor_pools: dict[int, ThreadPoolExecutor] = {}

    @cached_property
    def retry_wrapper(self):  # pragma: no cover
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
    def tokenizer_model_name(self) -> str:
        default_tokenizer_model_name = _model_name_to_tokenizer_model_name(
            self.model_name
        )
        if self._tokenizer_model_name is None and self.warn_tokenizer_model_name:
            warnings.warn(
                f"Could not detect the tokenizer for {self.display_name}"
                f" and will default to using the `{default_tokenizer_model_name}`"
                " tokenizer. Please explicitly set it with"
                f" {self.__class__.__name__}(..., tokenizer_model_name=)"
                " to remove this warning.",
                stacklevel=2,
            )
        return self._tokenizer_model_name or default_tokenizer_model_name

    @cached_property
    def config(self) -> None | PretrainedConfig:
        if _has_file(
            repo_id=self.tokenizer_model_name,
            filename="config.json",
            repo_type="model",
            revision=self.tokenizer_revision,
        ):
            return get_config(
                model_name=self.tokenizer_model_name,
                revision=self.tokenizer_revision,
                trust_remote_code=self.tokenizer_trust_remote_code,
            )
        else:
            return None

    @cached_property
    def _is_encoder_decoder(self) -> bool:  # pragma: no cover
        if self.config is not None:
            return is_encoder_decoder(self.config)
        else:
            return False

    @abstractmethod
    @cached_property
    def client(self) -> Any:
        pass

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        return get_tokenizer(
            model_name=self.tokenizer_model_name,
            revision=self.tokenizer_revision,
            trust_remote_code=self.tokenizer_trust_remote_code,
            **self.kwargs,
        )

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
        model_name_lower = self.model_name.lower()
        if self.max_context_length is not None:
            return self.max_context_length
        if self.config is not None:
            estimate = get_model_max_context_length(
                model_name=self.model_name, config=self.config
            )
        else:  # pragma: no cover
            if "-32k" in model_name_lower:
                estimate = 32768
            elif "-16k" in model_name_lower:
                estimate = 16384
            elif "-8k" in model_name_lower:
                estimate = 8192
            elif "-4k" in model_name_lower:
                estimate = 4096
            else:
                estimate = 2048
        if self.warn_max_context_length and not self._warned_max_context_length:
            self._warned_max_context_length = True
            warnings.warn(
                f"Could not detect the model context length for {self.display_name}"
                f" and will default to {estimate} tokens. Please explicitly set it with"
                f" {self.__class__.__name__}(..., max_context_length=)"
                " to remove this warning.",
                stacklevel=2,
            )
        return estimate - max_new_tokens

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

    def _is_batch_size_exception(self, e: BaseException) -> bool:  # pragma: no cover
        return False

    @abstractmethod
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
        pass

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
    def display_name(self) -> str:
        return super().display_name + f" ({self.model_name})"

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        names.append(
            str(self.model_max_length)
            if hasattr(self, "model_max_length")
            else str(self.get_max_context_length(max_new_tokens=0))
        )
        to_hash: list[Any] = []
        if self.tokenizer_model_name:
            to_hash.append(self.tokenizer_model_name)
            if self.tokenizer_revision:
                to_hash.append(self.tokenizer_revision)
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
        if "executor_pools" in state:
            state["executor_pools"].clear()

        return state


__all__ = ["LLMAPI"]
