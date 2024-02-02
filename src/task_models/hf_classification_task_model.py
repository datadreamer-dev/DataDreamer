import gc
from functools import cached_property, partial
from typing import Any, Callable, Generator, Iterable, cast

import torch
import torch._dynamo
from datasets.fingerprint import Hasher
from transformers import logging as transformers_logging

from ..logging import logger as datadreamer_logger
from ..utils.background_utils import RunIfTimeout
from ..utils.fs_utils import safe_fn
from ..utils.hf_hub_utils import get_citation_info, get_license_info, get_model_card_url
from ..utils.hf_model_utils import (
    HF_TRANSFORMERS_CITATION,
    PEFT_CITATION,
    convert_dtype,
    get_config,
    get_model_max_context_length,
    get_tokenizer,
    is_encoder_decoder,
)
from ..utils.import_utils import ignore_transformers_warnings
from .task_model import DEFAULT_BATCH_SIZE, TaskModel, _check_texts_length

with ignore_transformers_warnings():
    from transformers import (
        AutoModelForSequenceClassification,
        PreTrainedModel,
        PreTrainedTokenizer,
        pipeline,
    )


class HFClassificationTaskModel(TaskModel):
    def __init__(
        self,
        model_name: str,
        revision: None | str = None,
        trust_remote_code: bool = False,
        device: None | int | str | torch.device = None,
        device_map: None | dict | str = None,
        dtype: None | str | torch.dtype = None,
        adapter_name: None | str = None,
        adapter_kwargs: None | dict = None,
        cache_folder_path: None | str = None,
        **kwargs,
    ):
        """Loads a `HFClassificationTaskModel <https://huggingface.co/docs/transformers/tasks/sequence_classification>`_
        task model.

        Args:
            model_name: The name of the model to use.
            revision: The version (commit hash) of the model to use.
            trust_remote_code: Whether to trust remote code.
            device: The device to use for the model.
            device_map: The device map to use for the model.
            dtype: The type to use for the model weights.
            adapter_name: The name of the adapter to use.
            adapter_kwargs: Additional keyword arguments to pass the PeftModel
                constructor.
            cache_folder_path: The path to the cache folder. If ``None``, the default
                cache folder for the DataDreamer session will be used.
            **kwargs: Additional keyword arguments to pass to the Hugging Face model
                constructor.
        """
        super().__init__(cache_folder_path=cache_folder_path)
        self.model_name = model_name
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.device_map = device_map
        self.dtype = convert_dtype(dtype)
        self.kwargs = kwargs
        self.adapter_name = adapter_name
        self.adapter_kwargs = adapter_kwargs
        if (
            self.adapter_kwargs is not None and self.adapter_name is None
        ):  # pragma: no cover
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
    def _is_encoder_decoder(self) -> bool:  # pragma: no cover
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
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            torch_dtype=self.dtype,
            **self.kwargs,
        )
        model.config.eos_token_id = self.tokenizer.eos_token_id
        model.config.pad_token_id = self.tokenizer.pad_token_id

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

        # Torch compile
        # torch._dynamo.config.suppress_errors = True
        # model = torch.compile(model)

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

    @property
    def model_max_length(self) -> int:  # pragma: no cover
        return get_model_max_context_length(
            model_name=self.model_name, config=self.config
        )

    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        return len(self.tokenizer.encode(value))

    @torch.no_grad()
    def _run_batch(  # noqa: C901
        self,
        max_length_func: Callable[[list[str]], int],
        inputs: list[str],
        truncate: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs,
    ) -> list[dict[str, float]]:
        texts = inputs

        # Get the model
        model = self.model

        # Get inputs length length
        if not truncate:
            _check_texts_length(self=self, max_length_func=max_length_func, texts=texts)

        # Run model
        transformers_logging_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity(transformers_logging.CRITICAL)
        pipeline_optional_kwargs = (
            {} if self.device_map is not None else {"device": model.device}
        )
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=self.tokenizer,
            **pipeline_optional_kwargs,
        )
        transformers_logging.set_verbosity(transformers_logging_verbosity)
        label2id = model.config.label2id
        results = pipe(
            texts,
            batch_size=len(texts),
            add_special_tokens=True,
            top_k=len(label2id),
            **kwargs,
        )

        # Return a labeled dictionary
        return [
            {
                next(filter(lambda r: r["label"] == label, result))["label"]: next(
                    filter(lambda r: r["label"] == label, result)
                )["score"]
                for label in label2id.keys()
            }
            for result in results
        ]

    def run(
        self,
        texts: Iterable[str],
        truncate: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_scheduler_buffer_size: None | int = None,
        adaptive_batch_size: bool = True,
        progress_interval: None | int = 60,
        force: bool = False,
        cache_only: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        total_num_texts: None | int = None,
        return_generator: bool = False,
        **kwargs,
    ) -> Generator[dict[str, float], None, None] | list[dict[str, float]]:
        def get_max_length_function() -> dict[str, Any]:
            def max_length_func(texts: list[str]) -> int:
                return max([self.count_tokens(t) for t in texts])

            return {"max_length_func": max_length_func}

        results_generator = self._run_over_batches(
            run_batch=self._run_batch,
            get_max_input_length_function=get_max_length_function,
            max_model_length=self.model_max_length,
            inputs=texts,
            truncate=truncate,
            batch_size=batch_size,
            batch_scheduler_buffer_size=batch_scheduler_buffer_size,
            adaptive_batch_size=adaptive_batch_size,
            progress_interval=progress_interval,
            force=force,
            cache_only=cache_only,
            verbose=verbose,
            log_level=log_level,
            total_num_inputs=total_num_texts,
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
        if isinstance(model_citations, list):  # pragma: no cover
            citations.extend(model_citations)
        if isinstance(adapter_citations, list):  # pragma: no cover
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
        if self.revision:  # pragma: no cover
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


__all__ = ["HFClassificationTaskModel"]
