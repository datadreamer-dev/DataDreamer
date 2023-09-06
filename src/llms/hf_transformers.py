import warnings
from functools import cached_property

import torch

from datasets.fingerprint import Hasher

from ..utils.fs_utils import safe_fn
from .llm import LLM

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="distutils Version classes are deprecated.*",
        module="torch.utils.tensorboard",
    )
    from optimum.bettertransformer import BetterTransformer
    from optimum.bettertransformer.models import BetterTransformerManager
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
    )
    from transformers.utils.quantization_config import QuantizationConfigMixin


class HFTransformers(LLM):
    def __init__(
        self,
        model_name: str,
        revision: None | str = None,
        trust_remote_code: bool = False,
        device: None | int | str | torch.device = None,
        device_map: None | dict | str = None,
        dtype: None | str | torch.dtype = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        quantization_config: None | QuantizationConfigMixin | dict = None,
        **kwargs: dict,
    ):
        super().__init__()
        self.model_name = model_name
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.device = device
        self.device_map = device_map
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        if self.load_in_4bit:  # pragma: no cover
            bfloat16_supported = False
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                bfloat16_supported = True
            self.quantization_config = quantization_config or BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if bfloat16_supported else torch.float16
                ),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            self.quantization_config = quantization_config or None
        self.kwargs = kwargs
        self.config = AutoConfig.from_pretrained(
            model_name, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            **self.kwargs,
        )

    @cached_property
    def model(self) -> PreTrainedModel:
        model = AutoModel.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            device_map=self.device_map,
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            quantization_config=self.quantization_config,
            torch_dtype=self.dtype,
            **self.kwargs,
        )
        model = model.to(self.device)
        if BetterTransformerManager.cannot_support(
            model.config.model_type
        ) or not BetterTransformerManager.supports(model.config.model_type):
            return model  # pragma: no cover
        else:
            return BetterTransformer.transform(model)

    def get_max_context_length(self, max_new_tokens: int) -> int:
        is_encoder_decoder = (
            hasattr(self.config, "is_encoder_decoder")
            and self.config.is_encoder_decoder
        )
        if hasattr(self.config, "n_positions"):
            max_context_length = self.config.n_positions
        elif hasattr(self.config, "max_position_embeddings"):
            max_context_length = self.config.max_position_embeddings
        else:
            raise RuntimeError(
                f"Could not get the max content length of the model: '{self.model_name}'."
            )  # pragma: no cover
        if is_encoder_decoder:
            return max_context_length
        else:
            return max_context_length - max_new_tokens

    def count_tokens(self, value: str) -> int:
        return len(self.tokenizer.encode(value))

    @property
    def _cache_name(self) -> None | str:
        names = [safe_fn(self.model_name, allow_slashes=False)]
        to_hash = []
        if self.revision:
            names.append(self.revision)
        names.append(str(self.model.dtype))
        if self.load_in_4bit:  # pragma: no cover
            names.append("4bit")
        if self.load_in_8bit:  # pragma: no cover
            names.append("8bit")
        if self.quantization_config:  # pragma: no cover
            to_hash.append(self.quantization_config)
        if len(to_hash) > 0:  # pragma: no cover
            names.append(Hasher.hash(to_hash))
        return "_".join(names)


__all__ = ["HFTransformers"]
