# type: ignore
# ruff: noqa

import warnings

import torch

from ...utils.import_utils import ignore_transformers_warnings


def get_peft_model_cls():  # pragma: no cover
    # Two warnings we can't silence are thrown by peft at import-time so
    # we import this library only when needed
    with ignore_transformers_warnings():
        from peft import PeftConfig, PeftModel, PeftModelForFeatureExtraction, PeftType
        from peft.utils import _get_batch_size

    class CustomPeftModelForFeatureExtraction(PeftModelForFeatureExtraction):
        """
        Peft model for extracting features/embeddings from transformer models

        Args:
            model ([`~transformers.PreTrainedModel`]): Base transformer model.
            peft_config ([`PeftConfig`]): Peft config.

        **Attributes**:
            - **config** ([`~transformers.PretrainedConfig`]) -- The configuration object of the base model.

        Example:

            ```py
            >>> from transformers import AutoModel
            >>> from peft import PeftModelForFeatureExtraction, get_peft_config

            >>> config = {
            ...     "peft_type": "LORA",
            ...     "task_type": "FEATURE_EXTRACTION",
            ...     "inference_mode": False,
            ...     "r": 16,
            ...     "target_modules": ["query", "value"],
            ...     "lora_alpha": 32,
            ...     "lora_dropout": 0.05,
            ...     "fan_in_fan_out": False,
            ...     "bias": "none",
            ... }
            >>> peft_config = get_peft_config(config)
            >>> model = AutoModel.from_pretrained("bert-base-cased")
            >>> peft_model = PeftModelForFeatureExtraction(model, peft_config)
            >>> peft_model.print_trainable_parameters()
            ```
        """

        def __init__(
            self,
            model: torch.nn.Module,
            peft_config: PeftConfig,
            adapter_name: str = "default",
        ):
            super().__init__(model, peft_config, adapter_name)

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            peft_config = self.active_peft_config
            if not peft_config.is_prompt_learning:
                return self.get_base_model()(input_ids)

            batch_size = _get_batch_size(input_ids, inputs_embeds)
            if attention_mask is not None:
                # concat prompt attention mask
                prefix_attention_mask = torch.ones(
                    batch_size, peft_config.num_virtual_tokens
                ).to(attention_mask.device)
                attention_mask = torch.cat(
                    (prefix_attention_mask, attention_mask), dim=1
                )

            if kwargs.get("position_ids", None) is not None:
                warnings.warn(
                    "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                )
                kwargs["position_ids"] = None
            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None
            kwargs.update(
                {
                    "attention_mask": attention_mask,
                    "output_attentions": output_attentions,
                    "output_hidden_states": output_hidden_states,
                    "return_dict": return_dict,
                }
            )

            if peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size)
                return self.get_base_model()(
                    input_ids=input_ids, past_key_values=past_key_values, **kwargs
                )
            else:
                if inputs_embeds is None:
                    inputs_embeds = self.word_embeddings(input_ids)
                prompts = self.get_prompt(batch_size=batch_size)
                prompts = prompts.to(inputs_embeds.dtype)
                inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
                return self.get_base_model()(inputs_embeds=inputs_embeds, **kwargs)

    return CustomPeftModelForFeatureExtraction
