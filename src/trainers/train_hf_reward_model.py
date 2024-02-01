import json
import os
from functools import cached_property, partial
from typing import Any, Callable

import evaluate
import numpy as np
import torch
from torch.nn import functional as F

from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..utils.arg_utils import AUTO, Default
from ..utils.device_utils import _TrainingArgumentDeviceOverrideMixin
from ..utils.distributed_utils import not_distributed_or_main_process
from ..utils.hf_model_utils import get_base_model_from_peft_model
from ..utils.import_utils import ignore_transformers_warnings, ignore_trl_warnings
from ._train_hf_base import (
    CustomDataCollatorWithPadding,
    TrainingArguments,
    _prepare_inputs_and_outputs,
    _start_hf_trainer,
    _TrainHFBase,
    _wrap_trainer_cls,
    get_logging_callback,
)
from .train_hf_classifier import TrainHFClassifier
from .trainer import JointMetric, _monkey_patch_TrainerState__post_init__

with ignore_transformers_warnings():
    from transformers import EarlyStoppingCallback, PreTrainedModel
    from transformers.trainer_callback import PrinterCallback
    from transformers.training_args import OptimizerNames, SchedulerType
    from transformers.utils.quantization_config import QuantizationConfigMixin


class TrainHFRewardModel(TrainHFClassifier):
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
        _TrainHFBase.__init__(
            self,
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
        self._train_method: Callable
        if self.peft_config:  # pragma: no cover
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import TaskType

            self.peft_config.task_type = TaskType.SEQ_CLS

    def _train_with_pairs(
        self,
        train_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_chosen: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_rejected: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_chosen: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_rejected: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_chosen_scores: None
        | OutputDatasetColumn
        | OutputIterableDatasetColumn = None,
        train_rejected_scores: None
        | OutputDatasetColumn
        | OutputIterableDatasetColumn = None,
        validation_chosen_scores: None
        | OutputDatasetColumn
        | OutputIterableDatasetColumn = None,
        validation_rejected_scores: None
        | OutputDatasetColumn
        | OutputIterableDatasetColumn = None,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        optim: OptimizerNames | str = "adamw_torch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler_type: SchedulerType | str = "linear",
        warmup_steps: int = 0,
        neftune_noise_alpha: None | float = None,
        seed: int = 42,
        **kwargs,
    ):
        data_collator = kwargs.pop("data_collator", None)

        with ignore_trl_warnings():
            from trl import RewardConfig as _RewardConfig, RewardTrainer
        _monkey_patch_TrainerState__post_init__()

        # Validate arguments
        assert (train_chosen_scores is None) == (train_rejected_scores is None), (
            "You must either specify both `train_chosen_scores` and "
            "`train_rejected_scores` or set both to `None` if you only have"
            " generations."
        )
        assert (validation_chosen_scores is None) == (
            validation_rejected_scores is None
        ), (
            "You must either specify both `validation_chosen_scores` and "
            "`validation_rejected_scores` or set both to `None` if you only have"
            " generations."
        )
        assert (train_chosen_scores is None) == (validation_chosen_scores is None), (
            "You must either specify both `train_chosen_scores` and "
            "`validation_chosen_scores` or set both to `None` if you only have"
            " generations."
        )

        # Prepare datasets
        assert (
            self._is_encoder_decoder or truncate
        ), "`truncate=False` is not supported for this model."
        train_columns = {
            ("train_prompts", "Train Prompts"): train_prompts,
            ("train_chosen", "Train Chosen Generations"): train_chosen,
            ("train_rejected", "Train Rejected Generations"): train_rejected,
        }
        if train_chosen_scores is not None and train_rejected_scores is not None:
            train_columns.update(
                {
                    (
                        "train_chosen_scores",
                        "Train Chosen Generation Scores",
                    ): train_chosen_scores,
                    (
                        "train_rejected_scores",
                        "Train Rejected Generation Scores",
                    ): train_rejected_scores,
                }
            )
        validation_columns = {
            ("validation_prompts", "Validation Prompts"): validation_prompts,
            ("validation_chosen", "Validation Chosen Generations"): validation_chosen,
            (
                "validation_rejected",
                "Validation Rejected Generations",
            ): validation_rejected,
        }
        if (
            validation_chosen_scores is not None
            and validation_rejected_scores is not None
        ):
            validation_columns.update(
                {
                    (
                        "validation_chosen_scores",
                        "Validation Chosen Generation Scores",
                    ): validation_chosen_scores,
                    (
                        "validation_rejected_scores",
                        "Validation Rejected Generation Scores",
                    ): validation_rejected_scores,
                }
            )
        train_dataset, validation_dataset, _, _ = _prepare_inputs_and_outputs(
            self,
            train_columns=train_columns,
            validation_columns=validation_columns,
            truncate=truncate,
            reward_pairs=True,
        )
        label2id = {"reward": 0}
        id2label = {v: k for k, v in label2id.items()}

        # Prepare data collator
        data_collator = data_collator or CustomDataCollatorWithPadding(
            tokenizer=self.tokenizer,
            fields_to_pad=[
                {
                    "name": "input_ids_chosen",
                    "output_name": "input_ids_chosen",
                    "output_attention_mask_name": "attention_mask_chosen",
                },
                {
                    "name": "input_ids_rejected",
                    "output_name": "input_ids_rejected",
                    "output_attention_mask_name": "attention_mask_rejected",
                },
            ],
            fields_to_keep=["margin"] if train_chosen_scores is not None else None,
            extra_column_names_to_add={"return_loss": True},
        )

        # Prepare compute metrics
        def compute_accuracy_metrics(accuracy, eval_pred):
            predictions, labels = eval_pred
            loss = F.cross_entropy(
                input=torch.tensor(predictions),
                target=torch.tensor(labels).to(torch.int64),
            ).item()
            hard_predictions = np.argmax(predictions, axis=1)
            accuracy_metrics = accuracy.compute(
                predictions=hard_predictions, references=labels
            )
            return {
                **accuracy_metrics,
                "joint_metric": JointMetric(
                    is_joint_metric=True,
                    primary=accuracy_metrics["accuracy"],
                    primary_name="f1",
                    secondary=(-1 * loss),
                    secondary_name="loss",
                    secondary_inversed=True,
                ),
            }

        compute_metrics = kwargs.pop("compute_metrics", None) or partial(
            compute_accuracy_metrics, evaluate.load("accuracy")
        )

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

        # Prepare model
        self.seed = seed
        model = self._create_model(label2id=label2id, id2label=id2label)
        base_model = model
        if self.peft_config:
            base_model = get_base_model_from_peft_model(model)
        base_model.config.function_to_apply = "none"

        # Prepare training arguments
        class RewardConfig(_TrainingArgumentDeviceOverrideMixin, _RewardConfig):
            pass

        # This makes the dynamically-created RewardConfig pickle-able
        globals()["RewardConfig"] = RewardConfig
        globals()["RewardConfig"].__qualname__ = "RewardConfig"

        training_args = RewardConfig(
            remove_unused_columns=False,
            max_length=self.tokenizer.model_max_length,
            gradient_checkpointing=kwargs.pop("gradient_checkpointing", False),
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
            or "eval_joint_metric",
            greater_is_better=kwargs.pop("greater_is_better", True),
            load_best_model_at_end=kwargs.pop("load_best_model_at_end", True),
            seed=seed,
            neftune_noise_alpha=neftune_noise_alpha,
            **kwargs,
        )

        # Setup trainer
        trainer = _wrap_trainer_cls(
            trainer_cls=trainer_cls or RewardTrainer, **trainer_override_kwargs
        )(
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            model=model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=training_args,
        )
        assert trainer.use_reward_data_collator is False
        trainer.use_reward_data_collator = True
        trainer.remove_callback(PrinterCallback)

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
        if not_distributed_or_main_process():
            with open(
                os.path.join(self._output_folder_path, "_model", "label2id.json"), "w+"
            ) as f:
                json.dump(label2id, f, indent=4)
            with open(
                os.path.join(self._output_folder_path, "_model", "id2label.json"), "w+"
            ) as f:
                json.dump(id2label, f, indent=4)
            with open(
                os.path.join(
                    self._output_folder_path, "_model", "is_multi_target.json"
                ),
                "w+",
            ) as f:
                json.dump(False, f, indent=4)

        # Clean up resources after training
        self.unload_model()

    def _train_with_scores(
        self,
        train_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_generations: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_scores: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_generations: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_scores: OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        optim: OptimizerNames | str = "adamw_torch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler_type: SchedulerType | str = "linear",
        warmup_steps: int = 0,
        neftune_noise_alpha: None | float = None,
        seed: int = 42,
        **kwargs,
    ):
        data_collator = kwargs.pop("data_collator", None)

        with ignore_transformers_warnings():
            from transformers import Trainer

        # Prepare datasets
        assert (
            self._is_encoder_decoder or truncate
        ), "`truncate=False` is not supported for this model."
        train_dataset, validation_dataset, _, _ = _prepare_inputs_and_outputs(
            self,
            train_columns={
                ("train_input", "Train Prompts"): train_prompts,
                ("train_output", "Train Generations"): train_generations,
                ("label", "Train Scores"): train_scores,
            },
            validation_columns={
                ("validation_input", "Validation Prompts"): validation_prompts,
                ("validation_output", "Validation Generations"): validation_generations,
                ("label", "Validation Scores"): validation_scores,
            },
            truncate=truncate,
            reward_scores=True,
        )
        label2id = {"reward": 0}
        id2label = {v: k for k, v in label2id.items()}

        # Prepare data collator
        data_collator = data_collator or CustomDataCollatorWithPadding(
            tokenizer=self.tokenizer,
            fields_to_pad=[
                {
                    "name": "input_ids",
                    "output_name": "input_ids",
                    "output_attention_mask_name": "attention_mask",
                }
            ],
            fields_to_keep=["labels"],
        )

        # Prepare compute metrics
        def compute_mse_metrics(eval_pred):
            predictions, labels = eval_pred
            if isinstance(predictions, tuple):  # pragma: no cover
                predictions = predictions[0]
            predictions = [pred[0] for pred in predictions]
            mse_metrics = {
                "mse": F.mse_loss(torch.tensor(predictions), torch.tensor(labels))
            }
            return {**mse_metrics}

        compute_metrics = kwargs.pop("compute_metrics", None) or compute_mse_metrics

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

        # Prepare model
        self.seed = seed
        model = self._create_model(label2id=label2id, id2label=id2label)
        base_model = model
        if self.peft_config:
            base_model = get_base_model_from_peft_model(model)
        base_model.config.problem_type = "regression"
        base_model.config.function_to_apply = "none"

        # Prepare training arguments
        training_args = TrainingArguments(
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
            or "eval_mse",
            greater_is_better=kwargs.pop("greater_is_better", False),
            load_best_model_at_end=kwargs.pop("load_best_model_at_end", True),
            seed=seed,
            neftune_noise_alpha=neftune_noise_alpha,
            **kwargs,
        )

        # Setup trainer
        trainer = _wrap_trainer_cls(
            trainer_cls=trainer_cls or Trainer, **trainer_override_kwargs
        )(
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            model=model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=training_args,
        )
        trainer.remove_callback(PrinterCallback)

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
        with open(
            os.path.join(self._output_folder_path, "_model", "label2id.json"), "w+"
        ) as f:
            json.dump(label2id, f, indent=4)
        with open(
            os.path.join(self._output_folder_path, "_model", "id2label.json"), "w+"
        ) as f:
            json.dump(id2label, f, indent=4)
        with open(
            os.path.join(self._output_folder_path, "_model", "is_multi_target.json"),
            "w+",
        ) as f:
            json.dump(False, f, indent=4)

        # Clean up resources after training
        self.unload_model()

    def _train(self, *args, **kwargs):
        return self._train_method(*args, **kwargs)

    def train(self, *args, **kwargs) -> "TrainHFRewardModel":
        raise RuntimeError(
            "Do not use `.train()` for `TrainHFRewardModel`. Instead, use"
            " `.train_with_pairs()`, `.train_with_pairs_and_scores()`,"
            " `.train_with_scores()`."
        )

    def train_with_pairs(
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
        seed: int = 42,
        **kwargs,
    ) -> "TrainHFRewardModel":
        self._train_method = self._train_with_pairs
        self._setup_folder_and_resume(
            train_prompts=train_prompts,
            train_chosen=train_chosen,
            train_chosen_scores=None,
            train_rejected=train_rejected,
            train_rejected_scores=None,
            validation_prompts=validation_prompts,
            validation_chosen=validation_chosen,
            validation_chosen_scores=None,
            validation_rejected=validation_rejected,
            validation_rejected_scores=None,
            truncate=truncate,
            epochs=epochs,
            batch_size=batch_size,
            optim=optim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            neftune_noise_alpha=neftune_noise_alpha,
            seed=seed,
            **kwargs,
        )
        return self

    def train_with_pairs_and_scores(
        self,
        train_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_chosen: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_chosen_scores: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_rejected: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_rejected_scores: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_chosen: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_chosen_scores: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_rejected: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_rejected_scores: OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        optim: OptimizerNames | str = "adamw_torch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler_type: SchedulerType | str = "linear",
        warmup_steps: int = 0,
        neftune_noise_alpha: None | float = None,
        seed: int = 42,
        **kwargs,
    ) -> "TrainHFRewardModel":
        self._train_method = self._train_with_pairs
        self._setup_folder_and_resume(
            train_prompts=train_prompts,
            train_chosen=train_chosen,
            train_chosen_scores=train_chosen_scores,
            train_rejected=train_rejected,
            train_rejected_scores=train_rejected_scores,
            validation_prompts=validation_prompts,
            validation_chosen=validation_chosen,
            validation_chosen_scores=validation_chosen_scores,
            validation_rejected=validation_rejected,
            validation_rejected_scores=validation_rejected_scores,
            truncate=truncate,
            epochs=epochs,
            batch_size=batch_size,
            optim=optim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            neftune_noise_alpha=neftune_noise_alpha,
            seed=seed,
            **kwargs,
        )
        return self

    def train_with_scores(
        self,
        train_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_generations: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_scores: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_generations: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_scores: OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        optim: OptimizerNames | str = "adamw_torch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler_type: SchedulerType | str = "linear",
        warmup_steps: int = 0,
        neftune_noise_alpha: None | float = None,
        seed: int = 42,
        **kwargs,
    ) -> "TrainHFRewardModel":
        self._train_method = self._train_with_scores
        self._setup_folder_and_resume(
            train_prompts=train_prompts,
            train_generations=train_generations,
            train_scores=train_scores,
            validation_prompts=validation_prompts,
            validation_generations=validation_generations,
            validation_scores=validation_scores,
            truncate=truncate,
            epochs=epochs,
            batch_size=batch_size,
            optim=optim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            neftune_noise_alpha=neftune_noise_alpha,
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
@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeffrey and Jiang, Xu and Almeida, Diogo and"""
            """ Wainwright, Carroll and Mishkin, Pamela and Zhang, Chong and Agarwal,"""
            """ Sandhini and Slama, Katarina and Ray, Alex and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={27730--27744},
  year={2022}
}
        """.strip()
        )
        return citations


__all__ = ["TrainHFRewardModel"]
