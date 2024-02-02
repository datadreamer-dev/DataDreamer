import gc
import re
from functools import cached_property, partial
from types import MethodType
from typing import Any, Callable, Generator, Iterable, cast

import torch
import torch._dynamo
from datasets.fingerprint import Hasher
from transformers import logging as transformers_logging

from .._cachable._cachable import _StrWithSeed
from ..logging import logger as datadreamer_logger
from ..utils import ring_utils as ring
from ..utils.arg_utils import AUTO, Default
from ..utils.background_utils import RunIfTimeout
from ..utils.fs_utils import safe_fn
from ..utils.hf_hub_utils import get_citation_info, get_license_info, get_model_card_url
from ..utils.hf_model_utils import (
    HF_TRANSFORMERS_CITATION,
    PEFT_CITATION,
    convert_dtype,
    get_config,
    get_model_max_context_length,
    get_model_prompt_template,
    get_tokenizer,
    is_encoder_decoder,
)
from ..utils.import_utils import ignore_transformers_warnings
from .llm import (
    DEFAULT_BATCH_SIZE,
    LLM,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)

with ignore_transformers_warnings():
    from optimum.bettertransformer import BetterTransformer
    from optimum.bettertransformer.models import BetterTransformerManager
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        LogitsProcessorList,
        PreTrainedModel,
        PreTrainedTokenizer,
        RepetitionPenaltyLogitsProcessor,
        SequenceBiasLogitsProcessor,
        StoppingCriteria,
        StoppingCriteriaList,
        pipeline,
    )
    from transformers.utils.quantization_config import QuantizationConfigMixin


class CachedTokenizer:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.cache: dict[str, Any] = {}

    @classmethod
    def tokenizer_encode_cached(cls, cache, orig_method, self, *args, **kwargs):
        if "max_length" not in kwargs:
            kwargs["max_length"] = None
        if "truncation" not in kwargs:
            kwargs["truncation"] = None
        fingerprint = Hasher.hash([orig_method.__name__, args, kwargs])
        if fingerprint not in cache:
            cache[fingerprint] = orig_method(*args, **kwargs)
        return cache[fingerprint]

    def __call__(self, *args, **kwargs):
        return CachedTokenizer.tokenizer_encode_cached(
            self.cache, self.tokenizer.__call__, self.tokenizer, *args, **kwargs
        )

    def __getattr__(self, name):
        if (
            hasattr(self.tokenizer, name)
            and type(getattr(self.tokenizer, name)) == MethodType
            and "encode" in name
        ):
            return MethodType(
                partial(
                    CachedTokenizer.tokenizer_encode_cached,
                    self.cache,
                    getattr(self.tokenizer, name),
                ),
                self.tokenizer,
            )
        else:
            return getattr(self.tokenizer, name)


def _is_ctransformers(self: "HFTransformers") -> bool:
    from .ctransformers import CTransformers

    return isinstance(self, CTransformers)


def _is_petals(self: "HFTransformers") -> bool:
    from .petals import Petals

    return isinstance(self, Petals)


class SequenceStoppingCriteria(StoppingCriteria):
    def __init__(self, stop: str | list[str], prompts: list[str], tokenizer: Any):
        self.target_sequences: list[str]
        if isinstance(stop, str):
            self.target_sequences = [stop]
        else:
            self.target_sequences = stop
        self.prompt_replace_re = re.compile(
            "^("
            + "|".join(
                re.escape(p)
                for p in sorted(prompts, key=lambda p: len(p), reverse=True)
            )
            + ")",
            flags=re.MULTILINE,
        )
        self.target_sequences_match_re = re.compile(
            ".*(" + "|".join(re.escape(t) for t in self.target_sequences) + ").*",
            flags=re.MULTILINE,
        )
        self.target_sequences_replace_re = re.compile(
            "("
            + "|".join(
                re.escape(t)
                for t in sorted(
                    self.target_sequences, key=lambda t: len(t), reverse=True
                )
            )
            + ").*$",
            flags=re.MULTILINE,
        )
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        should_stop = []
        for row in input_ids:
            generated_text = self.tokenizer.decode(
                row, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            generated_text = re.sub(self.prompt_replace_re, "", generated_text)
            should_stop.append(
                re.search(self.target_sequences_match_re, generated_text) is not None
            )
        return all(should_stop)


class HFTransformers(LLM):
    def __init__(
        self,
        model_name: str,
        chat_prompt_template: None | str | Default = AUTO,
        system_prompt: None | str | Default = AUTO,
        revision: None | str = None,
        trust_remote_code: bool = False,
        device: None | int | str | torch.device = None,
        device_map: None | dict | str = None,
        dtype: None | str | torch.dtype = None,
        quantization_config: None | QuantizationConfigMixin | dict = None,
        adapter_name: None | str = None,
        adapter_kwargs: None | dict = None,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        super().__init__(cache_folder_path=cache_folder_path)
        self.model_name = model_name
        self.chat_prompt_template, self.system_prompt = get_model_prompt_template(
            model_name=self.model_name,
            revision=revision,
            chat_prompt_template=chat_prompt_template,
            system_prompt=system_prompt,
        )
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.device_map = device_map
        self.dtype = convert_dtype(dtype)
        self.quantization_config = quantization_config
        self.kwargs = kwargs
        self.adapter_name = adapter_name
        self.adapter_kwargs = adapter_kwargs
        if self.adapter_kwargs is not None and self.adapter_name is None:
            raise ValueError(
                "Cannot use adapter_kwargs if no adapter_name is specified."
            )
        if self.adapter_name is not None and self.adapter_kwargs is None:
            self.adapter_kwargs = {}

        # Load config
        self.config = get_config(
            model_name=self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )

    @cached_property
    def _is_encoder_decoder(self) -> bool:
        return is_encoder_decoder(self.config)

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
        if self._is_encoder_decoder:
            auto_cls = AutoModelForSeq2SeqLM
        else:
            auto_cls = AutoModelForCausalLM
        model = auto_cls.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            quantization_config=self.quantization_config,
            torch_dtype=self.dtype,
            **self.kwargs,
        )

        # Send model to accelerator device
        if self.device_map is None:
            model = model.to(self.device)

        # Load adapter
        if self.adapter_name:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, self.adapter_name, **cast(dict, self.adapter_kwargs)
            )

        # Switch model to eval mode
        model.eval()

        # Apply BetterTransformer
        if BetterTransformerManager.cannot_support(
            model.config.model_type
        ) or not BetterTransformerManager.supports(model.config.model_type):
            model = model  # pragma: no cover
        else:
            model = BetterTransformer.transform(model)

        # Torch compile
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)

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
        return get_tokenizer(
            model_name=self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
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
        """
        max_context_length = get_model_max_context_length(
            model_name=self.model_name, config=self.config
        )
        if self._is_encoder_decoder:
            return max_context_length
        else:
            return max_context_length - max_new_tokens

    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        """_summary_

        Args:
            value (_type_): _description_

        Returns:
            _type_: _description_
        """
        return len(self.tokenizer.encode(value))

    @torch.no_grad()
    def _run_batch(  # noqa: C901
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

        # Get the model
        model = self.model

        # Set seed
        if seed is not None:
            torch.manual_seed(seed + _StrWithSeed.total_per_input_seeds(inputs))
            if torch.cuda.is_available():  # pragma: no cover
                torch.cuda.manual_seed_all(
                    seed + _StrWithSeed.total_per_input_seeds(inputs)
                )

        # Determine whether pipeline is supported
        use_pipeline = True
        if self._is_encoder_decoder or _is_petals(self):
            use_pipeline = False

        # Encode prompts
        if not use_pipeline:
            model_inputs = cached_tokenizer.batch_encode_plus(
                prompts, return_tensors="pt", padding=True, add_special_tokens=True
            ).to(model.device)

        # Get max prompt length
        if not use_pipeline:
            max_prompt_length = int(model_inputs["input_ids"].shape[1])
        else:
            max_prompt_length = max_length_func(prompts)

        # Check max_new_tokens
        max_new_tokens = _check_max_new_tokens_possible(
            self=self,
            max_length_func=lambda x: max_prompt_length,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )

        # Set temperature and top_p
        temperature, top_p = _check_temperature_and_top_p(
            temperature=temperature, top_p=top_p, supports_zero_temperature=False
        )

        # Generate and decode
        logits_processor_list = []
        if logit_bias is not None:
            logits_processor_list.append(
                SequenceBiasLogitsProcessor(
                    sequence_bias={
                        (token_id,): float(bias)
                        for token_id, bias in logit_bias.items()
                    }
                )
            )
        if repetition_penalty is not None:
            logits_processor_list.append(
                RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)
            )
        stopping_criteria_list = []
        if stop is not None and len(stop) > 0:
            sequence_stopping_criteria = SequenceStoppingCriteria(
                stop=stop, prompts=prompts, tokenizer=cached_tokenizer
            )
            stopping_criteria_list.append(sequence_stopping_criteria)
        logits_processor = LogitsProcessorList(logits_processor_list)
        stopping_criteria = StoppingCriteriaList(stopping_criteria_list)
        generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=cached_tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            num_return_sequences=n,
            **kwargs,
        )
        if not use_pipeline:
            if _is_petals(self) and "attention_mask" in model_inputs:
                del model_inputs["attention_mask"]
            outputs = model.generate(**model_inputs, **generation_kwargs)
            texts = cached_tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            generated_texts_batch = [
                list(batch)
                for batch in zip(*(iter(texts),) * (len(texts) // len(prompts)))
            ]
        else:
            transformers_logging_verbosity = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity(transformers_logging.CRITICAL)
            original_padding_side = cached_tokenizer.tokenizer.padding_side
            pipeline_optional_kwargs = (
                {} if self.device_map is not None else {"device": model.device}
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=cached_tokenizer,
                **pipeline_optional_kwargs,
            )
            transformers_logging.set_verbosity(transformers_logging_verbosity)
            try:
                if not self._is_encoder_decoder:
                    cached_tokenizer.tokenizer.padding_side = "left"
                generated_texts_batch = [
                    [result["generated_text"] for result in batch]
                    for batch in pipe(
                        prompts,
                        batch_size=len(prompts),
                        **generation_kwargs,
                        add_special_tokens=True,
                        return_full_text=False,
                    )
                ]
            finally:
                cached_tokenizer.tokenizer.padding_side = original_padding_side

        # Post-process and return
        for prompt, batch in zip(prompts, generated_texts_batch):
            for idx in range(len(batch)):
                if (
                    not use_pipeline and not self._is_encoder_decoder
                ):  # pragma: no cover
                    # TODO: The coverage of this line is sometimes not tested
                    # because it depends on the TestPetals tests, which
                    # are not reliably run. This is because
                    # use_pipeline = False if the model is a Petals model
                    batch[idx] = batch[idx][len(prompt) :]
                if stop is not None and len(stop) > 0:
                    batch[idx] = re.sub(
                        sequence_stopping_criteria.target_sequences_replace_re,
                        "",
                        batch[idx],
                    ).strip()
                else:
                    batch[idx] = batch[idx].strip()
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
        adaptive_batch_size: bool = True,
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
        if _is_ctransformers(self):
            batch_size = 1
            adaptive_batch_size = False
            self.model._llm._config.batch_size = 1
            if hasattr(self.model._llm._config, "context_length"):
                self.model._llm._config.context_length = self.get_max_context_length(
                    max_new_tokens=0
                )
            if seed is not None:
                self.model._llm._config.seed = seed
            kwargs = {
                kwarg: value for kwarg, value in kwargs.items() if kwarg in ["top_k"]
            }
            for kwarg, value in list(kwargs.items()):  # pragma: no cover
                if hasattr(self.model._llm._config, kwarg) and kwarg not in ["top_k"]:
                    setattr(self.model._llm._config, kwarg, value)
                    del kwargs[kwarg]

        def get_max_length_function(tokenizer: Any) -> dict[str, Any]:
            cached_tokenizer = CachedTokenizer(tokenizer)

            def max_length_func(
                cached_tokenizer: CachedTokenizer, prompts: list[str]
            ) -> int:
                return max(
                    [
                        len(
                            cached_tokenizer(
                                p,
                                **{
                                    "padding": False,
                                    "add_special_tokens": True,
                                    "return_tensors": "pt",
                                },
                            )["input_ids"][0]
                        )
                        for p in prompts
                    ]
                )

            return {
                "max_length_func": partial(max_length_func, cached_tokenizer),
                "cached_tokenizer": cached_tokenizer,
            }

        results_generator = self._run_over_batches(
            run_batch=self._run_batch,
            get_max_input_length_function=partial(
                get_max_length_function, self.tokenizer
            ),
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
        return get_model_card_url(self.model_name)

    @cached_property
    def license(self) -> None | str:
        return get_license_info(
            self.model_name, repo_type="model", revision=self.revision
        )

    @cached_property
    def citation(self) -> None | list[str]:
        model_citations = get_citation_info(
            self.model_name, repo_type="model", revision=self.revision
        )
        citations = []
        citations.append(HF_TRANSFORMERS_CITATION)
        if hasattr(self, "adapter_name") and self.adapter_name:
            citations.append(PEFT_CITATION)
            adapter_citations = get_citation_info(
                self.adapter_name, repo_type="model", revision=self.revision
            )
        else:
            adapter_citations = None
        if isinstance(model_citations, list):
            citations.extend(model_citations)
        if isinstance(adapter_citations, list):
            citations.extend(adapter_citations)
        return citations

    @property
    def version(self) -> float:
        return 1.0

    @cached_property
    def display_name(self) -> str:
        if self.adapter_name:
            return super().display_name + f" ({self.model_name} + {self.adapter_name})"
        else:
            return super().display_name + f" ({self.model_name})"

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        if self.adapter_name:
            names.append(safe_fn(self.adapter_name, allow_slashes=False))
        if self.revision:
            names.append(self.revision)
        names.append(
            str(self.dtype)
            if self.dtype is not None
            else (
                str(self.config.torch_dtype)
                if hasattr(self.config, "torch_dtype")
                and self.config.torch_dtype is not None
                else str(torch.get_default_dtype())
            )
        )
        to_hash: list[Any] = []
        if self.quantization_config:  # pragma: no cover
            to_hash.append(self.quantization_config)
        if len(to_hash) > 0:  # pragma: no cover
            names.append(Hasher.hash(to_hash))
        return "_".join(names)

    def unload_model(self):
        # Delete cached model and tokenizer
        if "model" in self.__dict__:
            del self.__dict__["model"]
        if "tokenizer" in self.__dict__:
            del self.__dict__["tokenizer"]

        # Garbage collect
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()

    def __getstate__(self):  # pragma: no cover
        state = super().__getstate__()

        # Remove cached model or tokenizer before serializing
        state.pop("model", None)
        state.pop("tokenizer", None)

        return state


__all__ = ["HFTransformers"]
