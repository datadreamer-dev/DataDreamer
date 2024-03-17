from copy import copy
from functools import cache
from typing import Any

import torch

from .arg_utils import Default, default_to
from .import_utils import ignore_transformers_warnings

with ignore_transformers_warnings():
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )

HF_TRANSFORMERS_CITATION = """
@inproceedings{Wolf_Transformers_State-of-the-Art_Natural_2020,
  author = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien"""
""" and Delangue, Clement and Moi, Anthony and Cistac, Perric and"""
""" Ma, Clara and Jernite, Yacine and Plu, Julien and Xu, Canwen"""
""" and Le Scao, Teven and Gugger, Sylvain and Drame, Mariama"""
""" and Lhoest, Quentin and Rush, Alexander M.},
  month = oct,
  pages = {38--45},
  publisher = {Association for Computational Linguistics},
  title = {{Transformers: State-of-the-Art Natural Language Processing}},
  url = {https://www.aclweb.org/anthology/2020.emnlp-demos.6},
  year = {2020}
}
""".strip()


PEFT_CITATION = """
@Misc{peft,
  title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes"""
""" Belkada and Sayak Paul},
  howpublished = {\\url{https://github.com/huggingface/peft}},
  year =         {2022}
}
""".strip()


def get_model_prompt_template(
    model_name: str,
    revision: None | str,
    chat_prompt_template: None | str | Default,
    system_prompt: None | str | Default,
) -> tuple[None | str, None | str]:
    from ..llms._chat_prompt_templates import (
        _model_name_to_chat_prompt_template,
        _model_name_to_system_prompt,
    )

    _chat_prompt_template = default_to(
        chat_prompt_template,
        _model_name_to_chat_prompt_template(model_name=model_name, revision=revision),
    )
    _system_prompt = default_to(
        system_prompt,
        _model_name_to_system_prompt(
            chat_prompt_template=_chat_prompt_template,
            model_name=model_name,
            revision=revision,
        ),
    )
    del chat_prompt_template, system_prompt
    if _system_prompt is not None and (
        _chat_prompt_template is None
        or "{{system_prompt}}" not in _chat_prompt_template
    ):
        raise ValueError(
            "Cannot use system prompt if no `chat_prompt_template` is specified."
        )
    if (
        _chat_prompt_template
        and "{{system_prompt}}" in _chat_prompt_template
        and _system_prompt is None
    ):
        raise ValueError(
            "`system_prompt` cannot be null if using a `chat_prompt_template` with a"
            " system prompt."
        )
    return _chat_prompt_template, _system_prompt


def convert_dtype(dtype: None | str | torch.dtype) -> None | torch.dtype:
    if isinstance(dtype, str):
        str_to_torch_dtypes: dict[str, torch.dtype] = {
            str(getattr(torch, attr)).replace("torch.", "").lower(): getattr(
                torch, attr
            )
            for attr in dir(torch)
            if isinstance(getattr(torch, attr), torch.dtype)
        }
        return str_to_torch_dtypes[dtype.replace("torch.", "").lower()]
    else:
        return dtype


@cache
def get_config(
    model_name: str, revision: None | str, trust_remote_code: bool
) -> PretrainedConfig:
    return AutoConfig.from_pretrained(
        model_name, revision=revision, trust_remote_code=trust_remote_code
    )


def get_tokenizer(
    model_name: str, revision: None | str, trust_remote_code: bool, **kwargs
) -> PreTrainedTokenizer:
    tokenizer_config = get_config(
        model_name=model_name, revision=revision, trust_remote_code=trust_remote_code
    )
    model_max_length = get_model_max_context_length(
        model_name=model_name, config=tokenizer_config
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        model_max_length=model_max_length,
        **kwargs,
    )

    # Setup tokenizer
    if not tokenizer._pad_token or tokenizer.pad_token_id < 0:
        tokenizer.pad_token = tokenizer.eos_token

    # Silence warnings
    tokenizer.deprecation_warnings = {"Asking-to-pad-a-fast-tokenizer": True}

    return tokenizer


def is_encoder_decoder(config: PretrainedConfig) -> bool:
    return hasattr(config, "is_encoder_decoder") and config.is_encoder_decoder


def get_model_max_context_length(model_name: str, config: PretrainedConfig) -> int:
    if hasattr(config, "max_sequence_length"):
        max_context_length = config.max_sequence_length
    elif hasattr(config, "n_positions"):
        max_context_length = config.n_positions
    elif hasattr(config, "max_position_embeddings"):
        max_context_length = config.max_position_embeddings
    elif hasattr(config, "seq_length"):  # pragma: no cover
        # https://huggingface.co/THUDM/chatglm3-6b/blob/main/config.json
        max_context_length = config.seq_length
    elif hasattr(config, "embedding_size"):  # pragma: no cover
        # https://huggingface.co/rrivera1849/LUAR-MUD/blob/main/config.json
        max_context_length = config.embedding_size
    else:
        if "bloom" in model_name:  # pragma: no cover
            max_context_length = 2048
        elif config.model_type in ["t5", "mt5"]:
            max_context_length = 512
        else:
            raise RuntimeError(
                f"Could not get the max content length of the model: '{model_name}'."
            )  # pragma: no cover
    return max_context_length


def get_model_embedding_size(
    model_name: str, config: PretrainedConfig
) -> int:  # pragma: no cover
    if hasattr(config, "hidden_size"):
        embedding_size = config.hidden_size
    elif hasattr(config, "n_embed"):
        embedding_size = config.n_embed
    elif hasattr(config, "d_model"):
        embedding_size = config.d_model
    else:
        raise RuntimeError(
            f"Could not get the embedding size of the model: '{model_name}'."
        )  # pragma: no cover
    return embedding_size


def is_peft_model(model: Any) -> bool:
    return model.__class__.__name__.startswith("PeftModel")


def get_base_model_from_peft_model(model: Any) -> PreTrainedModel:
    base_model = model.base_model
    if hasattr(base_model, "model"):
        base_model = base_model.model
    return base_model


def get_orig_model(model):  # pragma: no cover
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    else:
        return model


def validate_peft_config(model, peft_config: None | Any):
    peft_config = copy(peft_config)
    if (
        peft_config is not None
        and hasattr(peft_config, "target_modules")
        and peft_config.target_modules is None
    ):
        target_modules = []
        fan_in_fan_out = getattr(peft_config, "fan_in_fan_out", False)
        for name, module in model.named_modules():
            module_type = module.__class__.__name__.lower()
            if module_type.startswith("linear") or module_type.startswith("conv1d"):
                target_module = name.split(".")[-1]
                if target_module not in ["lm_head"]:
                    if module_type.startswith("conv1d"):
                        fan_in_fan_out = True
                    target_modules.append(target_module)
        peft_config.target_modules = list(set(target_modules))
        peft_config.fan_in_fan_out = fan_in_fan_out
    return peft_config
