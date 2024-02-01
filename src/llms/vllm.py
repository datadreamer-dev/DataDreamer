import gc
import logging
import os
from functools import cached_property, partial
from typing import Any, Callable, Generator, Iterable

import dill
import torch
from datasets.fingerprint import Hasher

from .. import DataDreamer
from ..logging import logger as datadreamer_logger
from ..utils.arg_utils import AUTO, Default
from ..utils.background_utils import RunIfTimeout, proxy_resource_in_background
from ..utils.device_utils import get_device_env_variables, is_cpu_device
from ..utils.fs_utils import safe_fn
from ..utils.hf_model_utils import get_tokenizer
from ..utils.import_utils import ignore_transformers_warnings, import_module
from .hf_transformers import CachedTokenizer, HFTransformers
from .llm import (
    DEFAULT_BATCH_SIZE,
    _check_max_new_tokens_possible,
    _check_temperature_and_top_p,
)

with ignore_transformers_warnings():
    from transformers import PreTrainedTokenizer


class VLLM(HFTransformers):  # pragma: no cover
    def __init__(
        self,
        model_name: str,
        chat_prompt_template: None | str | Default = AUTO,
        system_prompt: None | str | Default = AUTO,
        revision: None | str = None,
        trust_remote_code: bool = False,
        device: None | int | str | torch.device | list[int | str | torch.device] = None,
        dtype: None | str | torch.dtype = None,
        quantization: None | str = None,
        swap_space: int = 1,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        assert device is not None, (
            "vLLM requires the `device` parameter to be a GPU device or a list of GPU"
            " devices."
        )
        device = [device] if not isinstance(device, list) else device  # type
        assert not any([is_cpu_device(d) for d in device]), (
            "vLLM requires the `device` parameter to be a GPU device or a list of GPU"
            " devices."
        )
        super().__init__(
            model_name=model_name,
            chat_prompt_template=chat_prompt_template,
            system_prompt=system_prompt,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device=device,  # type:ignore[arg-type]
            dtype=dtype,
            cache_folder_path=cache_folder_path,
            **kwargs,
        )
        self.quantization = quantization
        if self.quantization is None and "-awq" in model_name.lower():
            self.quantization = "awq"
        self.swap_space = swap_space

    @cached_property
    def model(self) -> Any:
        env = os.environ.copy()
        assert isinstance(self.device, list)
        env.update(get_device_env_variables(self.device))
        kwargs = self.kwargs.copy()
        tensor_parallel_size = kwargs.pop("tensor_parallel_size", len(self.device))

        class LLMResource:
            def __init__(self_resource):
                # Disable VLLM loggers
                if DataDreamer.initialized() and not DataDreamer.ctx.hf_log:
                    vllm_logging = import_module("vllm.logger")
                    _old_init_logger = vllm_logging.init_logger

                    def _monkey_patch_init_logger(*args, **kwargs):
                        logger = _old_init_logger(*args, **kwargs)
                        logger.level = logging.ERROR
                        return logger

                    vllm_logging.init_logger = _monkey_patch_init_logger
                    logging.getLogger("vllm.engine.llm_engine").level = logging.ERROR

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
                LLM = import_module("vllm").LLM
                self_resource.model = LLM(
                    model=self.model_name,
                    trust_remote_code=self.trust_remote_code,
                    dtype=str(self.dtype).replace("torch.", "")
                    if self.dtype is not None
                    else "auto",
                    quantization=self.quantization,
                    revision=self.revision,
                    swap_space=self.swap_space,
                    tensor_parallel_size=tensor_parallel_size,
                    **kwargs,
                )

                # Finished loading
                log_if_timeout.stop(
                    partial(
                        lambda self: self.get_logger(
                            key="model", log_level=datadreamer_logger.level
                        ).info("Finished loading."),
                        self,
                    )
                )

            def get_generated_texts_batch(self_resource, args, kwargs):
                args = dill.loads(args)
                kwargs = dill.loads(kwargs)
                outputs = self_resource.model.generate(*args, **kwargs)
                generated_texts_batch = [
                    [o.text for o in batch.outputs] for batch in outputs
                ]
                return generated_texts_batch

        return proxy_resource_in_background(LLMResource, env=env)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        return get_tokenizer(
            model_name=self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )

    def _is_batch_size_exception(self, e: BaseException) -> bool:
        # TODO (fix later if vLLM updates):
        # This is not possible with VLLM yet (detect when CUDA OOM)
        return False

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
        assert (
            logit_bias is None
        ), f"`logit_bias` is not supported for {type(self).__name__}"
        assert seed is None, f"`seed` is not supported for {type(self).__name__}"

        SamplingParams = import_module("vllm").SamplingParams

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

        # Generate
        sampling_params = SamplingParams(
            n=n,
            presence_penalty=(
                repetition_penalty if repetition_penalty is not None else 0.0
            ),
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            max_tokens=max_new_tokens,
            **kwargs,
        )
        generated_texts_batch = self.model.proxy.get_generated_texts_batch(
            args=dill.dumps((prompts, sampling_params)),
            kwargs=dill.dumps({"use_tqdm": False}),
        )

        # Post-process and return
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
    def citation(self) -> None | list[str]:
        citations = super().citation or []
        citations.append(
            """
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin"""
            """ Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
            """.strip()
        )
        return citations

    @property
    def version(self) -> float:
        return 1.0

    @cached_property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        if self.revision:
            names.append(self.revision)
        names.append(str(self.dtype))
        to_hash: list[Any] = []
        if self.quantization is not None:  # pragma: no cover
            to_hash.append(self.quantization)
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

    def __getstate__(self):  # pragma: no cover
        state = super().__getstate__()

        # Remove cached model or tokenizer before serializing
        state.pop("model", None)
        state.pop("tokenizer", None)

        return state


__all__ = ["VLLM"]
