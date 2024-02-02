import warnings
from functools import cached_property, partial
from typing import Any, Generator, Iterable

from datasets.fingerprint import Hasher

from ..logging import logger as datadreamer_logger
from ..utils import ring_utils as ring
from ..utils.arg_utils import AUTO, Default
from ..utils.background_utils import RunIfTimeout
from ..utils.fs_utils import safe_fn
from ..utils.hf_model_utils import get_model_prompt_template
from ..utils.import_utils import ignore_transformers_warnings
from .hf_transformers import HFTransformers
from .llm import DEFAULT_BATCH_SIZE, LLM

with ignore_transformers_warnings():
    from ctransformers import AutoModelForCausalLM, AutoTokenizer
    from ctransformers.transformers import CTransformersTokenizer
    from transformers import PreTrainedModel, PreTrainedTokenizer


def _add_tokens_patched(*args, **kwargs):
    # Makes ctransformers support transformers==4.34.0.dev0
    return 0


CTransformersTokenizer._add_tokens = _add_tokens_patched


class CTransformers(HFTransformers):
    def __init__(
        self,
        model_name: str,
        model_type: None | str = None,
        model_file: None | str = None,
        max_context_length: None | int = None,
        chat_prompt_template: None | str | Default = AUTO,
        system_prompt: None | str | Default = AUTO,
        revision: None | str = None,
        threads: None | int = None,
        gpu_layers: int = 0,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        LLM.__init__(self, cache_folder_path=cache_folder_path)
        self.model_name = model_name
        self.model_type = model_type
        self.model_file = model_file
        self.max_context_length = max_context_length
        if self.max_context_length is None:
            warnings.warn(
                "CTransformers may not provide an accurate model context length."
                " Explicitly set it with CTransformers(..., max_context_length=)"
                " to remove this warning.",
                stacklevel=2,
            )
        self.chat_prompt_template, self.system_prompt = get_model_prompt_template(
            model_name=self.model_name,
            revision=revision,
            chat_prompt_template=chat_prompt_template,
            system_prompt=system_prompt,
        )
        self.revision = revision
        self.threads = threads
        self.gpu_layers = gpu_layers
        self.kwargs = kwargs

    @cached_property
    def _is_encoder_decoder(self) -> bool:
        return False

    @cached_property
    def model(self) -> PreTrainedModel:
        # Load model
        log_if_timeout = RunIfTimeout(
            partial(
                lambda self: self.get_logger(
                    key="model", log_level=datadreamer_logger.level
                ).info("Loading..."),
                self,
            ),
            timeout=10.0,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.revision,
            model_type=self.model_type,
            model_file=self.model_file,
            **self.kwargs,
            gpu_layers=self.gpu_layers,
            hf=True,
        )

        # Set threads
        if self.threads is not None:
            model._llm._config.threads = self.threads

        # Finished loading
        log_if_timeout.stop(
            partial(
                lambda self: self.get_logger(
                    key="model", log_level=datadreamer_logger.level
                ).info("Finished loading."),
                self,
            )
        )

        return model

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(self.model)

    @ring.lru(maxsize=128)
    def get_max_context_length(self, max_new_tokens: int) -> int:
        """Gets the maximum context length for the model. When ``max_new_tokens`` is
        greater than 0, the maximum number of tokens that can be used for the prompt
        context is returned.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.

        Returns:
            The maximum context length.
        """
        if self.max_context_length is None:
            return self.model._llm.context_length - max_new_tokens
        else:
            return self.max_context_length - max_new_tokens

    def _is_batch_size_exception(self, e: BaseException) -> bool:  # pragma: no cover
        return False

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
            total_num_inputs=total_num_prompts,
            return_generator=return_generator,
            **kwargs,
        )

    @property
    def version(self) -> float:
        return 1.0

    @cached_property
    def display_name(self) -> str:
        if self.model_file is not None:
            return (
                LLM.display_name.func(self) + f" ({self.model_name}/{self.model_file})"  # type: ignore[attr-defined]
            )
        else:
            return LLM.display_name.func(self) + f" ({self.model_name})"  # type: ignore[attr-defined]

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        if (
            self.max_context_length is not None
            and self.max_context_length != self.model._llm.context_length
        ):
            names.append(str(self.max_context_length))
        if self.revision:
            names.append(self.revision)
        to_hash: list[Any] = []
        if self.model_type is not None:
            to_hash.append(self.model_type)
        if self.model_file is not None:
            to_hash.append(self.model_file)
        kwargs_filtered = {
            key: self.kwargs[key] for key in ["local_files_only"] if key in self.kwargs
        }
        if len(kwargs_filtered) > 0:
            to_hash.append(kwargs_filtered)
        if len(to_hash) > 0:  # pragma: no cover
            names.append(Hasher.hash(to_hash))
        return "_".join(names)


__all__ = ["CTransformers"]
