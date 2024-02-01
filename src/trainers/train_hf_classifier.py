import json
import os
from functools import partial
from typing import Any, Callable, Type

import evaluate
import numpy as np
import torch
from datasets import Sequence, Value  # type:ignore[attr-defined]
from torch.nn import functional as F

from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..utils.arg_utils import AUTO, Default
from ..utils.distributed_utils import not_distributed_or_main_process
from ..utils.import_utils import ignore_transformers_warnings
from ._train_hf_base import (
    TrainingArguments,
    _prepare_inputs_and_outputs,
    _start_hf_trainer,
    _TrainHFBase,
    _wrap_trainer_cls,
    get_logging_callback,
)
from .trainer import JointMetric, _monkey_patch_TrainerState__post_init__

with ignore_transformers_warnings():
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        PreTrainedModel,
    )
    from transformers.trainer_callback import PrinterCallback
    from transformers.training_args import OptimizerNames, SchedulerType
    from transformers.utils.quantization_config import QuantizationConfigMixin


class TrainHFClassifier(_TrainHFBase):
    def __init__(
        self,
        name: str,
        model_name: str,
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
            chat_prompt_template=None,
            system_prompt=None,
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
        self.chat_prompt_template = None
        self.system_prompt = None
        if self.peft_config:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import TaskType

            self.peft_config.task_type = TaskType.SEQ_CLS

    @property
    def auto_cls(self) -> Type:
        return AutoModelForSequenceClassification

    def _train(  # type:ignore[override]
        self,
        train_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_output: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_output: OutputDatasetColumn | OutputIterableDatasetColumn,
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
        with ignore_transformers_warnings():
            from transformers import Trainer
        _monkey_patch_TrainerState__post_init__()

        # Prepare datasets
        (
            train_dataset,
            validation_dataset,
            label2id,
            is_multi_target,
        ) = _prepare_inputs_and_outputs(
            self,
            train_columns={
                ("input_ids", "Train Input"): train_input,
                ("label", "Train Output"): train_output,
            },
            validation_columns={
                ("input_ids", "Validation Input"): validation_input,
                ("label", "Validation Output"): validation_output,
            },
            truncate=truncate,
        )
        id2label = {v: k for k, v in label2id.items()}
        assert (
            len(id2label) > 1
        ), "There must be at least 2 output labels in your dataset."
        if is_multi_target:
            train_dataset = train_dataset.cast_column(
                "label",
                Sequence(
                    feature=Value(dtype="float64"),
                    length=train_dataset.features["label"].length,  # type:ignore[union-attr]
                    id=None,
                ),
            )
            validation_dataset = validation_dataset.cast_column(
                "label",
                Sequence(
                    feature=Value(dtype="float64"),
                    length=validation_dataset.features["label"].length,  # type:ignore[union-attr]
                    id=None,
                ),
            )

        # Prepare data collator
        data_collator = kwargs.pop("data_collator", None) or DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )

        # Prepare compute metrics
        def compute_accuracy_metrics(accuracy, f1, eval_pred):
            predictions, labels = eval_pred
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            loss_fct: Callable
            if is_multi_target:
                loss_fct = F.binary_cross_entropy_with_logits
            else:
                loss_fct = F.cross_entropy
            loss = loss_fct(
                input=torch.tensor(predictions), target=torch.tensor(labels)
            ).item()
            if is_multi_target:
                hard_predictions = np.where(predictions > 0.5, 1, 0)
            else:
                hard_predictions = np.argmax(predictions, axis=1)
            accuracy_metrics = accuracy.compute(
                predictions=hard_predictions, references=labels
            )
            f1_metrics = f1.compute(
                predictions=hard_predictions, references=labels, average="micro"
            )
            return {
                **accuracy_metrics,
                **f1_metrics,
                "joint_metric": JointMetric(
                    is_joint_metric=True,
                    primary=f1_metrics["f1"],
                    primary_name="f1",
                    secondary=(-1 * loss),
                    secondary_name="loss",
                    secondary_inversed=True,
                ),
            }

        compute_metrics = kwargs.pop("compute_metrics", None) or partial(
            compute_accuracy_metrics,
            evaluate.load(
                "accuracy", config_name="multilabel" if is_multi_target else None
            ),
            evaluate.load("f1", config_name="multilabel" if is_multi_target else None),
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
        model = self._create_model(
            label2id=label2id, id2label=id2label, is_multi_target=is_multi_target
        )

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
            save_total_limit=kwargs.pop("save_total_limit", 3),
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
                json.dump(is_multi_target, f, indent=4)

        # Clean up resources after training
        self.unload_model()

    def train(  # type:ignore[override]
        self,
        train_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_output: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_output: OutputDatasetColumn | OutputIterableDatasetColumn,
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
    ) -> "TrainHFClassifier":
        self._setup_folder_and_resume(
            train_input=train_input,
            train_output=train_output,
            validation_input=validation_input,
            validation_output=validation_output,
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

    def _load(self, with_optimizations: bool = True):
        with open(
            os.path.join(self._output_folder_path, "_model", "is_multi_target.json"),
            "r",
        ) as f:
            is_multi_target = json.load(f)
            self.is_multi_target = is_multi_target
        with open(
            os.path.join(self._output_folder_path, "_model", "label2id.json"), "r"
        ) as f:
            label2id = json.load(f)
            self.label2id = label2id
        with open(
            os.path.join(self._output_folder_path, "_model", "id2label.json"), "r"
        ) as f:
            id2label = json.load(f)
            id2label = {int(k): v for k, v in id2label.items()}
            self.id2label = id2label
        model = self._load_model(
            label2id=label2id,
            id2label=id2label,
            is_multi_target=is_multi_target,
            with_optimizations=with_optimizations,
        )
        return model

    def export_to_disk(self, path: str, adapter_only: bool = False) -> PreTrainedModel:
        return super().export_to_disk(path=path, adapter_only=adapter_only)

    def _publish_info(
        self, repo_id: str, branch: None | str = None, adapter_only: bool = False
    ) -> dict[str, Any]:  # pragma: no cover
        publish_info = super()._publish_info(
            repo_id=repo_id, branch=branch, adapter_only=adapter_only
        )
        publish_info["pipeline_tag"] = "text-classification"
        auto_cls_name = self.auto_cls.__name__
        if self.peft_config and adapter_only:
            function_to_apply = (
                "base_model.config.function_to_apply = 'none'\n"
                if len(self.label2id) == 1
                else ""
            )
            if self.is_multi_target:
                problem_type = f", problem_type={repr('multi_label_classification')}"
            else:
                problem_type = ""
            body = (
                "## Example Usage\n\n```python3\n"
                f"import torch\n"
                f"from transformers import {auto_cls_name}, AutoTokenizer, pipeline\n"
                f"from peft import PeftModel\n"
                f"\n"
                f"torch.manual_seed({self.seed}) # Set seed\n"
                f"if torch.cuda.is_available():\n"
                f"    torch.cuda.manual_seed_all({self.seed}) # Set seed\n"
                f"\n"
                f"tokenizer = AutoTokenizer.from_pretrained({repr(repo_id)},"
                f" revision={repr(branch)}) # Load tokenizer\n"
                f"label2id = {repr(self.label2id)}\n"
                f"id2label = {repr(self.id2label)}\n"
                f"base_model = {auto_cls_name}.from_pretrained({repr(self.model_name)},"
                f" revision={repr(self.revision)}, num_labels=len(label2id),"
                f" label2id=label2id, id2label=id2label{problem_type}) # Load base model\n"
                f"{function_to_apply}"
                f"model = PeftModel.from_pretrained(base_model, model_id={repr(repo_id)},"
                f" revision={repr(branch)}) # Apply adapter\n"
            )
        else:
            body = (
                "## Example Usage\n\n```python3\n"
                f"from transformers import {auto_cls_name}, AutoTokenizer, pipeline\n"
                "\n"
                f"tokenizer = AutoTokenizer.from_pretrained({repr(repo_id)},"
                f" revision={repr(branch)}) # Load tokenizer\n"
                f"model = {auto_cls_name}.from_pretrained({repr(repo_id)},"
                f" revision={repr(branch)}) # Load model\n"
            )
        if len(self.label2id) == 1:
            body += (
                f"pipe = pipeline({repr(publish_info['pipeline_tag'])}, model=model,"
                " tokenizer=tokenizer, function_to_apply='none')\n\n"
                f"inputs = {repr(self._examples['Train Input'][:1])}\n"
                f"print(pipe(inputs))\n"
            )
        else:
            if self.is_multi_target:
                pred = (
                    f"print([sorted([c['label'] for c in r if c['score'] >= 0.5])"
                    f" for r in pipe(inputs, top_k={len(self.label2id)})])\n"
                )
            else:
                pred = "print(pipe(inputs))\n"
            body += (
                f"pipe = pipeline({repr(publish_info['pipeline_tag'])}, model=model,"
                " tokenizer=tokenizer)\n\n"
                f"inputs = {repr(self._examples['Train Input'][:1])}\n"
                f"{pred}"
            )
        body += "```"
        publish_info["body"] = body
        return publish_info

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


__all__ = ["TrainHFClassifier"]
