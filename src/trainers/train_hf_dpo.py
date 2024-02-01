import os
from functools import cached_property
from typing import Any

import torch

from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..steps import Step
from ..steps.step_operations import _INTERNAL_STEP_OPERATION_KEY
from ..utils.arg_utils import AUTO, Default
from ..utils.distributed_utils import is_distributed, not_main_process
from ..utils.import_utils import ignore_transformers_warnings, ignore_trl_warnings
from ._train_hf_base import (
    CustomDataCollatorWithPadding,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    _prepare_inputs_and_outputs,
    _start_hf_trainer,
    _wrap_trainer_cls,
    get_logging_callback,
)
from .train_hf_finetune import TrainHFFineTune

with ignore_transformers_warnings():
    from transformers import EarlyStoppingCallback, PreTrainedModel
    from transformers.trainer_callback import PrinterCallback
    from transformers.training_args import OptimizerNames, SchedulerType
    from transformers.utils.quantization_config import QuantizationConfigMixin


class _PreComputeRefLogProbs(Step):
    def setup(self):
        self.register_arg("pre_compute_func", help="The pre-compute function.")
        self.register_arg(
            "dataset_fingerprint", help="The dataset (for fingerprinting purposes)."
        )

    def run(self):
        return self.args["pre_compute_func"]()


setattr(_PreComputeRefLogProbs, _INTERNAL_STEP_OPERATION_KEY, True)


class TrainHFDPO(TrainHFFineTune):
    def __init__(
        self,
        name: str,
        model_name: str,
        chat_prompt_template: None | str | Default = AUTO,
        system_prompt: None | str | Default = AUTO,
        revision: None | str = None,
        trust_remote_code: bool = False,
        device: None | int | str | torch.device | list[int | str | torch.device] = None,
        dtype: None | str | torch.dtype = None,
        quantization_config: None | QuantizationConfigMixin | dict = None,
        peft_config: None | Any = None,
        distributed_config: dict[str, Any] | Default = AUTO,
        fsdp: bool | str | list[str] | Default = AUTO,
        fsdp_config: None | dict[str, Any] | Default = AUTO,
        force: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            model_name=model_name,
            chat_prompt_template=chat_prompt_template,
            system_prompt=system_prompt,
            revision=revision,
            trust_remote_code=trust_remote_code,
            device=device,
            dtype=dtype,
            quantization_config=quantization_config,
            peft_config=peft_config,
            distributed_config=distributed_config,
            fsdp=fsdp,
            fsdp_config=fsdp_config,
            force=force,
            verbose=verbose,
            log_level=log_level,
            **kwargs,
        )

    def _train(  # type:ignore[override] # noqa: C901
        self,
        train_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_chosen: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_rejected: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_chosen: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_rejected: OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        optim: OptimizerNames | str = "adamw_torch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler_type: SchedulerType | str = "linear",
        warmup_steps: int = 0,
        neftune_noise_alpha: None | float = None,
        dpo_beta: float = 0.1,
        loss_type: str = "sigmoid",
        disable_dropout: bool = True,
        precompute_ref_log_probs: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        data_collator = kwargs.pop("data_collator", None)

        with ignore_trl_warnings():
            from ._vendored.dpo_trainer import DPOTrainer  # type: ignore[attr-defined]

        # Prepare datasets
        assert (
            self._is_encoder_decoder or truncate
        ), "`truncate=False` is not supported for this model."
        train_dataset, validation_dataset, _, _ = _prepare_inputs_and_outputs(
            self,
            train_columns={
                ("train_prompts", "Train Prompts"): train_prompts,
                ("train_chosen", "Train Chosen Generations"): train_chosen,
                ("train_rejected", "Train Rejected Generations"): train_rejected,
            },
            validation_columns={
                ("validation_prompts", "Validation Prompts"): validation_prompts,
                (
                    "validation_chosen",
                    "Validation Chosen Generations",
                ): validation_chosen,
                (
                    "validation_rejected",
                    "Validation Rejected Generations",
                ): validation_rejected,
            },
            truncate=truncate,
            dpo=True,
        )

        # We have already tokenized the dataset, so don't let DPOTrainer try to tokenize.
        train_dataset.map = (  # type:ignore[method-assign,union-attr]
            lambda *args, **kwargs: train_dataset
        )
        validation_dataset.map = (  # type:ignore[method-assign,union-attr]
            lambda *args, **kwargs: validation_dataset
        )

        # Prepare compute metrics
        compute_metrics = kwargs.pop("compute_metrics", None)

        # Prepare callbacks
        callbacks = [get_logging_callback(self)]
        if (
            "early_stopping_patience" not in kwargs
            or kwargs["early_stopping_patience"] is not None
        ):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=kwargs.pop("early_stopping_patience", 5),
                    early_stopping_threshold=kwargs.pop(
                        "early_stopping_threshold", 0.0
                    ),
                )
            )
        kwargs.pop("early_stopping_patience", None)
        kwargs.pop("early_stopping_threshold", None)
        callbacks += kwargs.pop("callbacks", [])

        # Trainer overrides
        trainer_cls = kwargs.pop("trainer_cls", None)
        trainer_override_kwargs = {
            kwarg: kwargs.pop(kwarg)
            for kwarg in ["optimizers", "optimizer", "lr_scheduler", "compute_loss"]
            if kwarg in kwargs
        }

        # Prepare preprocess_logits_for_metrics
        preprocess_logits_for_metrics = kwargs.pop(
            "preprocess_logits_for_metrics", None
        )

        # Prepare model and reference model
        self.seed = seed
        model = self._create_model()
        if self.peft_config or precompute_ref_log_probs:
            # DPOTrainer will automatically use the PEFT model with the adapters disabled
            # as the reference model.
            # OR...
            # If we are pre-computing the ref log probs, they will be computed at the
            # beginning of training before the model weights are updataed, so we don't
            # need to keep a separate reference model at all.
            ref_model = None
        else:
            ref_model = self._create_model(is_ref_model=True)

        # Prepare training arguments
        if self._is_encoder_decoder:
            training_args_cls = Seq2SeqTrainingArguments
        else:
            training_args_cls = TrainingArguments
        training_args = training_args_cls(
            remove_unused_columns=False,
            _device=self.device,
            _model=model,
            fsdp=self.fsdp,
            fsdp_config=self.fsdp_config,
            report_to=kwargs.pop("report_to", None),
            run_name=f"DataDreamer - {self.name}",
            disable_tqdm=True,
            output_dir=os.path.join(self._output_folder_path, "_checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            optim=optim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            logging_strategy=kwargs.pop("logging_strategy", None) or "steps",
            logging_steps=kwargs.pop("logging_steps", 1),
            evaluation_strategy=kwargs.pop("evaluation_strategy", None) or "epoch",
            save_strategy=kwargs.pop("save_strategy", None) or "epoch",
            save_total_limit=kwargs.pop("save_total_limit", 1),
            save_safetensors=True,
            metric_for_best_model=kwargs.pop("metric_for_best_model", None)
            or "eval_rewards/margins",
            greater_is_better=kwargs.pop("greater_is_better", True),
            load_best_model_at_end=kwargs.pop("load_best_model_at_end", True),
            seed=seed,
            neftune_noise_alpha=neftune_noise_alpha,
            **kwargs,
        )

        # Setup trainer
        other_fields_to_keep = []
        if precompute_ref_log_probs:
            other_fields_to_keep = [
                "reference_chosen_logps",
                "reference_rejected_logps",
            ]
        if self._is_encoder_decoder:
            # Prepare data collator
            data_collator = data_collator or CustomDataCollatorWithPadding(
                tokenizer=self.tokenizer,
                fields_to_pad=[
                    {
                        "name": "prompt_input_ids",
                        "output_name": "prompt_input_ids",
                        "output_attention_mask_name": "prompt_attention_mask",
                    },
                    {
                        "name": "chosen_labels",
                        "output_name": "chosen_labels",
                        "pad_token_id": -100,
                    },
                    {
                        "name": "rejected_labels",
                        "output_name": "rejected_labels",
                        "pad_token_id": -100,
                    },
                ],
                fields_to_keep=[
                    "prompt",
                    "chosen",
                    "rejected",
                    "chosen_response_only",
                    "rejected_response_only",
                ]
                + other_fields_to_keep,
            )
        else:
            # Prepare data collator
            left_tokenizer = self.__class__.tokenizer.func(self)  # type: ignore[attr-defined]
            left_tokenizer.padding_side = "left"
            data_collator = data_collator or CustomDataCollatorWithPadding(
                tokenizer=self.tokenizer,
                fields_to_pad=[
                    {
                        "name": "prompt_input_ids",
                        "output_name": "prompt_input_ids",
                        "output_attention_mask_name": "prompt_attention_mask",
                        "tokenizer": left_tokenizer,
                    },
                    {
                        "name": "chosen_input_ids",
                        "output_name": "chosen_input_ids",
                        "output_attention_mask_name": "chosen_attention_mask",
                    },
                    {
                        "name": "chosen_labels",
                        "output_name": "chosen_labels",
                        "pad_token_id": -100,
                        "keep_first_pad_token": True,
                    },
                    {
                        "name": "rejected_input_ids",
                        "output_name": "rejected_input_ids",
                        "output_attention_mask_name": "rejected_attention_mask",
                    },
                    {
                        "name": "rejected_labels",
                        "output_name": "rejected_labels",
                        "pad_token_id": -100,
                        "keep_first_pad_token": True,
                    },
                ],
                fields_to_keep=[
                    "prompt",
                    "chosen",
                    "rejected",
                    "chosen_response_only",
                    "rejected_response_only",
                ]
                + other_fields_to_keep,
            )
        trainer = _wrap_trainer_cls(
            trainer_cls=trainer_cls or DPOTrainer, **trainer_override_kwargs
        )(
            label_pad_token_id=-100,
            padding_value=0,
            is_encoder_decoder=self._is_encoder_decoder,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            model=model,
            ref_model=ref_model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=training_args,
            beta=dpo_beta,
            loss_type=loss_type,
            disable_dropout=disable_dropout,
            precompute_ref_log_probs=precompute_ref_log_probs,
            generate_during_eval=False,
        )
        assert trainer.use_dpo_data_collator is False
        trainer.use_dpo_data_collator = True
        trainer.remove_callback(PrinterCallback)

        # Setup models for DDP/FSDP (this is needed for DPO working on distributed)
        # TODO (fix later if TRL updates):
        # See: https://github.com/huggingface/trl/issues/1147
        # TODO (fix later if TRL updates):
        # See: https://github.com/huggingface/trl/pull/1160
        if is_distributed():  # pragma: no cover
            prepared_model = trainer._wrap_model(
                trainer.model, training=True, dataloader=None
            )
            if hasattr(trainer.lr_scheduler, "step"):
                prepared_model, trainer.optimizer = trainer.accelerator.prepare(
                    prepared_model, trainer.optimizer
                )
            else:
                (
                    prepared_model,
                    trainer.optimizer,
                    trainer.lr_scheduler,
                ) = trainer.accelerator.prepare(
                    prepared_model, trainer.optimizer, trainer.lr_scheduler
                )
            trainer.model_wrapped = prepared_model
            if trainer.is_fsdp_enabled:
                trainer.model = prepared_model
            if trainer.ref_model is not None:
                trainer.ref_model = trainer.accelerator.prepare_model(trainer.ref_model)
            trainer.accelerator.prepare_model = lambda model, *args, **kwargs: model

        # Pre-compute ref_log_probs
        if precompute_ref_log_probs:

            def pre_compute_train():
                trainer.get_train_dataloader()
                return trainer.train_dataset

            pre_compute_train_step_done = os.path.join(
                self._output_folder_path,
                "pre-compute-reference-log-probs-on-train-dataset",
                "step.json",
            )
            if not_main_process() and not os.path.isfile(
                pre_compute_train_step_done
            ):  # pragma: no cover
                pre_compute_train()
            trainer.train_datset = _PreComputeRefLogProbs(
                "Pre-compute Reference Log Probs on Train Dataset",
                args={
                    "pre_compute_func": pre_compute_train,
                    "dataset_fingerprint": [
                        c.fingerprint
                        for c in [train_prompts, train_chosen, train_rejected]
                    ],
                },
            ).output.dataset
            trainer._precomputed_train_ref_log_probs = True
            assert os.path.isfile(pre_compute_train_step_done)

            def pre_compute_eval():
                trainer.get_eval_dataloader()
                return trainer.eval_dataset

            pre_compute_validation_step_done = os.path.join(
                self._output_folder_path,
                "pre-compute-reference-log-probs-on-validation-dataset",
                "step.json",
            )
            if not_main_process() and not os.path.isfile(
                pre_compute_validation_step_done
            ):  # pragma: no cover
                pre_compute_eval()
            trainer.eval_datset = _PreComputeRefLogProbs(
                "Pre-compute Reference Log Probs on Validation Dataset",
                args={
                    "pre_compute_func": pre_compute_eval,
                    "dataset_fingerprint": [
                        c.fingerprint
                        for c in [
                            validation_prompts,
                            validation_chosen,
                            validation_rejected,
                        ]
                    ],
                },
            ).output.dataset
            trainer._precomputed_eval_ref_log_probs = True
            assert os.path.isfile(pre_compute_validation_step_done)

        # Start the trainer
        _start_hf_trainer(self, trainer)

        # Save the model to disk
        self._save_model(
            training_args=training_args,
            model=trainer.model,
            tokenizer=trainer.tokenizer,
            accelerator=trainer.accelerator,
            fsdp=trainer.is_fsdp_enabled,
        )

        # Clean up resources after training
        self.unload_model()

    def train(  # type:ignore[override]
        self,
        train_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_chosen: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_rejected: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_chosen: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_rejected: OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        optim: OptimizerNames | str = "adamw_torch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler_type: SchedulerType | str = "linear",
        warmup_steps: int = 0,
        neftune_noise_alpha: None | float = None,
        dpo_beta: float = 0.1,
        loss_type: str = "kto_pair",
        disable_dropout: bool = True,
        seed: int = 42,
        **kwargs,
    ) -> "TrainHFDPO":
        self._setup_folder_and_resume(
            train_prompts=train_prompts,
            train_chosen=train_chosen,
            train_rejected=train_rejected,
            validation_prompts=validation_prompts,
            validation_chosen=validation_chosen,
            validation_rejected=validation_rejected,
            truncate=truncate,
            epochs=epochs,
            batch_size=batch_size,
            optim=optim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            neftune_noise_alpha=neftune_noise_alpha,
            dpo_beta=dpo_beta,
            loss_type=loss_type,
            disable_dropout=disable_dropout,
            seed=seed,
            **kwargs,
        )
        return self

    def export_to_disk(self, path: str, adapter_only: bool = False) -> PreTrainedModel:
        return super().export_to_disk(path=path, adapter_only=adapter_only)

    def publish_to_hf_hub(
        self,
        repo_id: str,
        branch: None | str = None,
        private: bool = False,
        token: None | str = None,
        adapter_only: bool = False,
        is_synthetic: bool = True,
        **kwargs,
    ) -> str:  # pragma: no cover
        return super().publish_to_hf_hub(
            repo_id=repo_id,
            branch=branch,
            private=private,
            token=token,
            adapter_only=adapter_only,
            **kwargs,
        )

    @cached_property
    def citation(self) -> None | list[str]:
        citations = super().citation or []
        citations.append(
            """
@misc{vonwerra2022trl,
  author = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward"""
            """ Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang},
  title = {TRL: Transformer Reinforcement Learning},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/huggingface/trl}}
}
        """.strip()
        )
        citations.append(
            """
@article{rafailov2023direct,
  title={Direct preference optimization: Your language model is secretly a reward model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano"""
            """ and Manning, Christopher D and Finn, Chelsea},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023}
}
        """.strip()
        )
        return citations


__all__ = ["TrainHFDPO"]
