import json
import os
from functools import cached_property, partial
from itertools import islice
from typing import Any
from uuid import uuid4

import evaluate
import numpy as np
import torch
from datasets import Dataset, Value
from torch.nn import functional as F

from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..embedders.sentence_transformers_embedder import _normalize_model_name
from ..utils.arg_utils import AUTO, DEFAULT, Default, default_to
from ..utils.background_utils import RunIfTimeout
from ..utils.hf_model_utils import (
    get_base_model_from_peft_model,
    get_model_max_context_length,
    get_tokenizer,
    validate_peft_config,
)
from ..utils.import_utils import ignore_transformers_warnings
from ._train_hf_base import (
    CustomDataCollatorWithPadding,
    TrainingArguments,
    _prepare_inputs_and_outputs,
    _start_hf_trainer,
    _TrainHFBase,
    _wrap_trainer_cls,
    get_logging_callback,
)
from ._vendored import _sentence_transformer_helper
from .trainer import JointMetric, _monkey_patch_TrainerState__post_init__

with ignore_transformers_warnings():
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.models.Transformer import Transformer
    from transformers import EarlyStoppingCallback, PreTrainedModel
    from transformers.trainer_callback import PrinterCallback
    from transformers.training_args import OptimizerNames, SchedulerType


class SentenceTransformerWrapper:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.results: dict[str, Any] = {}

    def _save_return_value(self, return_value, *args, **kwargs):
        _uniq_id: None | str = None
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, dict) and "_uniq_id" in arg:
                _uniq_id = arg.pop("_uniq_id")
        if _uniq_id:
            self.results[_uniq_id] = return_value
        return return_value

    def forward(self, *args, **kwargs):  # pragma: no cover
        return self._save_return_value(
            self.model.forward(*args, **kwargs), *args, **kwargs
        )

    def __call__(self, *args, **kwargs):
        if isinstance(self.model, SentenceTransformer):
            # Handle regular forward
            return self._save_return_value(
                self.model.forward(*args, **kwargs), *args, **kwargs
            )
        else:
            # Handle PEFT forward
            return self._save_return_value(
                self.model.model.forward(*args, **kwargs), *args, **kwargs
            )


class SentenceTransformerLossWrapper(torch.nn.Module):
    def __init__(
        self,
        orig_model: SentenceTransformer,
        wrapped_model: SentenceTransformerWrapper,
        loss_module: torch.nn.Module,
        _is_peft: bool,
    ):
        torch.nn.Module.__init__(self)
        self.orig_model = orig_model
        self.wrapped_model = wrapped_model
        self.loss_module = loss_module
        self._is_peft = _is_peft

    def __getattr__(self, name):
        if name == "config":
            if self._is_peft:
                sentence_transformer_model = get_base_model_from_peft_model(
                    self.orig_model
                )
            else:
                sentence_transformer_model = self.orig_model
            has_transformer_module = (
                "0" in sentence_transformer_model._modules
                and isinstance(sentence_transformer_model._modules["0"], Transformer)
            )
            if has_transformer_module:
                transformer_module = sentence_transformer_model._modules["0"]
                has_auto_model = (
                    "auto_model" in transformer_module._modules
                    and isinstance(
                        transformer_module._modules["auto_model"], PreTrainedModel
                    )
                )
                if has_auto_model:
                    return transformer_module._modules["auto_model"].config
        return super().__getattr__(name)

    def forward(
        self,
        anchor_input_ids: None | torch.Tensor = None,
        anchor_attention_mask: None | torch.Tensor = None,
        positive_input_ids: None | torch.Tensor = None,
        positive_attention_mask: None | torch.Tensor = None,
        negative_input_ids: None | torch.Tensor = None,
        negative_attention_mask: None | torch.Tensor = None,
        labels: None | torch.Tensor = None,
    ):
        _uniq_ids = []
        sentence_features = []
        _uniq_ids.append(uuid4().hex)
        sentence_features.append(
            {
                "_uniq_id": _uniq_ids[-1],
                "input_ids": anchor_input_ids,
                "attention_mask": anchor_attention_mask,
            }
        )
        if positive_input_ids is not None:
            _uniq_ids.append(uuid4().hex)
            sentence_features.append(
                {
                    "_uniq_id": _uniq_ids[-1],
                    "input_ids": positive_input_ids,
                    "attention_mask": positive_attention_mask,
                }
            )
        if negative_input_ids is not None:
            _uniq_ids.append(uuid4().hex)
            sentence_features.append(
                {
                    "_uniq_id": _uniq_ids[-1],
                    "input_ids": negative_input_ids,
                    "attention_mask": negative_attention_mask,
                }
            )
        loss = self.loss_module(sentence_features=sentence_features, labels=labels)
        return {
            "loss": loss,
            "embeddings": [
                self.wrapped_model.results[_uniq_id]["sentence_embedding"].detach()
                for _uniq_id in _uniq_ids
            ],
            "loss_for_joint_metric": loss,
        }


class TrainSentenceTransformer(_TrainHFBase):
    def __init__(
        self,
        name: str,
        model_name: str,
        trust_remote_code: bool = False,
        device: None | int | str | torch.device | list[int | str | torch.device] = None,
        dtype: None | str | torch.dtype = None,
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
            revision=None,
            trust_remote_code=trust_remote_code,
            device=device or "cpu",
            dtype=dtype,
            quantization_config=None,
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

            self.peft_config.task_type = TaskType.FEATURE_EXTRACTION

    def _create_model(
        self,
        label2id: None | dict[int, Any] = None,
        id2label: None | dict[int, Any] = None,
        is_multi_target: bool = False,
        device: None
        | int
        | str
        | torch.device
        | list[int | str | torch.device]
        | Default = DEFAULT,
        is_ref_model: bool = False,
    ) -> SentenceTransformer:
        # Seed
        if self.seed:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():  # pragma: no cover
                torch.cuda.manual_seed_all(self.seed)

        # Load model
        log_if_timeout = RunIfTimeout(
            partial(lambda self: self.logger.info("Loading model..."), self),
            timeout=10.0,
        )
        model_device = default_to(device, self.device)
        model = SentenceTransformer(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            device="cpu" if isinstance(model_device, list) else model_device,
            **self.kwargs,
        )
        model[0].tokenizer = get_tokenizer(
            _normalize_model_name(self.model_name),
            revision=None,
            trust_remote_code=self.trust_remote_code,
        )
        model.max_seq_length = (
            get_model_max_context_length(
                model_name=self.model_name, config=model[0].auto_model.config
            )
            if model.max_seq_length is None
            else model.max_seq_length
        )
        self.max_seq_length = model.max_seq_length

        # Set model dtype
        model = model.to(self.dtype)

        # Create PeftModel if peft_config
        if self.peft_config:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import get_peft_model, prepare_model_for_kbit_training

            if self.quantization_config:  # pragma: no cover
                model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, validate_peft_config(model, self.peft_config))

        # Switch model to train mode
        model.train()

        # Finished loading
        log_if_timeout.stop(
            partial(lambda self: self.logger.info("Finished loading."), self)
        )

        return model

    def _save_resource(self, resource: Any, path: str):
        if isinstance(resource, SentenceTransformerLossWrapper):
            resource = resource.wrapped_model.model
        if isinstance(resource, SentenceTransformer):
            resource.save(path=path, create_model_card=False)
        else:
            return _TrainHFBase._save_resource(self, resource=resource, path=path)

    def _publish_resource(
        self,
        resource: Any,
        repo_id: str,
        branch: None | str = None,
        private: bool = False,
        **kwargs,
    ):  # pragma: no cover
        if isinstance(resource, SentenceTransformer):
            _sentence_transformer_helper.save_to_hub(  # type:ignore[attr-defined]
                resource, repo_id=repo_id, private=private, exist_ok=True, **kwargs
            )
        else:
            return super()._publish_resource(
                resource=resource, repo_id=repo_id, branch=branch, private=private
            )

    def _train(  # type:ignore[override] # noqa: C901
        self,
        train_anchors: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_positives: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_negatives: None | OutputDatasetColumn | OutputIterableDatasetColumn,
        train_labels: None | OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_anchors: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_positives: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_negatives: None | OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_labels: None | OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        margin: float | Default = DEFAULT,
        epochs: float = 3.0,
        batch_size: int = 8,
        loss: type[torch.nn.Module] | Default = AUTO,
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

        # Validate arguments
        assert (train_negatives is None) == (validation_negatives is None), (
            "You must either specify both `train_negatives` and `validation_negatives`"
            " or set both to `None` if you only have positive examples."
        )
        assert (train_labels is None) == (validation_labels is None), (
            "You must either specify both `train_labels` and `validation_labels`"
            " or set both to `None` if you don't have labels."
        )
        has_negative_examples = (
            train_negatives is not None and validation_negatives is not None
        )
        has_labels = train_labels is not None and validation_labels is not None
        assert not (
            has_negative_examples and has_labels
        ), "You cannot specify both negative examples and labels."

        # Detect type of training
        loss_cls: Any
        if has_negative_examples:
            # Training with (anchor, positive, negative) triplets
            positive_name = "Positive"
            loss_cls = default_to(
                loss,
                lambda model: losses.TripletLoss(
                    model,
                    distance_metric=losses.TripletDistanceMetric.COSINE,
                    triplet_margin=default_to(margin, 0.5),
                ),
            )
            metric_for_best_model = "eval_joint_metric"
            greater_is_better = True
        elif has_labels:
            # Training with (anchor, other) labeled pairs
            positive_name = "Other"
            loss_cls = default_to(
                loss, lambda model: losses.CosineSimilarityLoss(model)
            )
            metric_for_best_model = "eval_loss"
            greater_is_better = True
        else:
            assert margin == DEFAULT, (
                "`margin` parameter is not supported when training"
                " with only positive pairs."
            )
            # Training with (anchor, positive) pairs
            positive_name = "Positive"
            loss_cls = default_to(loss, losses.MultipleNegativesSymmetricRankingLoss)
            metric_for_best_model = "eval_loss"
            greater_is_better = False

        # Prepare datasets
        train_columns = {
            ("anchor_input_ids", "Train Anchors"): train_anchors,
            ("positive_input_ids", f"Train {positive_name}s"): train_positives,
        }
        if has_negative_examples and train_negatives is not None:
            train_columns[("negative_input_ids", "Train Negatives")] = train_negatives
        if has_labels and train_labels is not None:
            train_columns[("labels", "Train Labels")] = train_labels
        validation_columns = {
            ("anchor_input_ids", "Validation Anchors"): validation_anchors,
            (
                "positive_input_ids",
                f"Validation {positive_name}s",
            ): validation_positives,
        }
        if has_negative_examples and validation_negatives is not None:
            validation_columns[
                ("negative_input_ids", "Validation Negatives")
            ] = validation_negatives
        if has_labels and validation_labels is not None:
            validation_columns[("labels", "Validation Labels")] = validation_labels
        train_dataset, validation_dataset, _, _ = _prepare_inputs_and_outputs(
            self,
            train_columns=train_columns,
            validation_columns=validation_columns,
            truncate=truncate,
        )
        if has_labels:
            train_dataset = train_dataset.cast_column("labels", Value("float64"))
            validation_dataset = validation_dataset.cast_column(
                "labels", Value("float64")
            )

        # Prepare data collator
        fields_to_pad = [
            {
                "name": "anchor_input_ids",
                "output_name": "anchor_input_ids",
                "output_attention_mask_name": "anchor_attention_mask",
            },
            {
                "name": "positive_input_ids",
                "output_name": "positive_input_ids",
                "output_attention_mask_name": "positive_attention_mask",
            },
        ]
        if has_negative_examples:
            fields_to_pad.append(
                {
                    "name": "negative_input_ids",
                    "output_name": "negative_input_ids",
                    "output_attention_mask_name": "negative_attention_mask",
                }
            )
        data_collator = kwargs.pop(
            "data_collator", None
        ) or CustomDataCollatorWithPadding(
            tokenizer=self.tokenizer,
            fields_to_pad=fields_to_pad,
            fields_to_keep=["labels"] if has_labels else [],
            extra_column_names_to_add={"labels": torch.tensor(-1)}
            if not has_labels
            else None,
        )

        # Prepare compute metrics
        def compute_accuracy_metrics(accuracy, f1, eval_pred):
            (all_embeddings, loss), labels = eval_pred
            if isinstance(loss, np.ndarray):  # pragma: no cover
                loss = np.mean(loss)
            if has_negative_examples:
                (
                    anchor_embeddings,
                    positive_embeddings,
                    negative_embeddings,
                ) = all_embeddings
                preds = []
                labels = []
                bz = 128
                idx_iter = iter(range(anchor_embeddings.shape[0]))
                while True:
                    idx_batch = list(islice(idx_iter, bz))
                    if len(idx_batch) == 0:
                        break
                    anchor_embeddings_batch = torch.tensor(
                        anchor_embeddings[idx_batch[0] : idx_batch[-1] + 1]
                    )
                    positive_embeddings_batch = torch.tensor(
                        positive_embeddings[idx_batch[0] : idx_batch[-1] + 1]
                    )
                    negative_embeddings_batch = torch.tensor(
                        negative_embeddings[idx_batch[0] : idx_batch[-1] + 1]
                    )
                    pos_sims = F.cosine_similarity(
                        anchor_embeddings_batch, positive_embeddings_batch
                    )
                    neg_sims = F.cosine_similarity(
                        anchor_embeddings_batch, negative_embeddings_batch
                    )
                    preds.extend(
                        [int(p_s > n_s) for p_s, n_s in zip(pos_sims, neg_sims)]
                    )
                    labels.extend([1] * len(pos_sims))
                accuracy_metrics = accuracy.compute(
                    predictions=preds, references=labels
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
            else:
                return {}

        compute_metrics = kwargs.pop("compute_metrics", None) or partial(
            compute_accuracy_metrics, evaluate.load("accuracy"), evaluate.load("f1")
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
            for kwarg in ["optimizers", "optimizer", "lr_scheduler"]
            if kwarg in kwargs
        }

        # Prepare preprocess_logits_for_metrics
        preprocess_logits_for_metrics = kwargs.pop(
            "preprocess_logits_for_metrics", None
        )

        # Prepare model
        self.seed = seed
        model = self._create_model()
        wrapped_model = SentenceTransformerWrapper(model)
        loss_module = loss_cls(wrapped_model)
        loss_wrapper_cls = SentenceTransformerLossWrapper
        if self.peft_config:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import PeftModel

            class PeftSentenceTransformerLossWrapper(
                SentenceTransformerLossWrapper, PeftModel
            ):
                @property
                def module(self):
                    return self.orig_model

                def __getattr__(self, name):
                    if name not in [
                        "orig_model",
                        "wrapped_model",
                        "loss_module",
                        "forward",
                        "config",
                    ]:
                        return getattr(self.orig_model, name)
                    else:
                        return super().__getattr__(name)

            loss_wrapper_cls = PeftSentenceTransformerLossWrapper
        wrapped_model_with_loss = loss_wrapper_cls(
            orig_model=model,
            wrapped_model=wrapped_model,
            loss_module=loss_module,
            _is_peft=self.peft_config is not None,
        )

        # Prepare training arguments
        training_args = TrainingArguments(
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
            # save_safetensors=True breaks if True when using "t5-base" for
            # SentenceTransformers
            save_safetensors=False,
            metric_for_best_model=kwargs.pop("metric_for_best_model", None)
            or metric_for_best_model,
            greater_is_better=kwargs.pop("greater_is_better", greater_is_better),
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
            model=wrapped_model_with_loss,
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

        # Clean up resources after training
        self.unload_model()

    def train(self, *args, **kwargs) -> "TrainSentenceTransformer":
        raise RuntimeError(
            "Do not use `.train()` for `TrainSentenceTransformer`. Instead, use"
            " `.train_with_triplets()`, `.train_with_positive_pairs()`,"
            " `.train_with_labeled_pairs()`."
        )

    def train_with_triplets(
        self,
        train_anchors: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_positives: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_negatives: None | OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_anchors: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_positives: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_negatives: None | OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        margin: float | Default = DEFAULT,
        epochs: float = 3.0,
        batch_size: int = 8,
        loss: type[torch.nn.Module] | Default = AUTO,
        optim: OptimizerNames | str = "adamw_torch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler_type: SchedulerType | str = "linear",
        warmup_steps: int = 0,
        neftune_noise_alpha: None | float = None,
        seed: int = 42,
        **kwargs,
    ) -> "TrainSentenceTransformer":
        self._setup_folder_and_resume(
            train_anchors=train_anchors,
            train_positives=train_positives,
            train_negatives=train_negatives,
            train_labels=None,
            validation_anchors=validation_anchors,
            validation_positives=validation_positives,
            validation_negatives=validation_negatives,
            validation_labels=None,
            truncate=truncate,
            margin=margin,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
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

    def train_with_positive_pairs(
        self,
        train_anchors: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_positives: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_anchors: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_positives: OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        loss: type[torch.nn.Module] | Default = AUTO,
        optim: OptimizerNames | str = "adamw_torch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler_type: SchedulerType | str = "linear",
        warmup_steps: int = 0,
        neftune_noise_alpha: None | float = None,
        seed: int = 42,
        **kwargs,
    ) -> "TrainSentenceTransformer":
        self._setup_folder_and_resume(
            train_anchors=train_anchors,
            train_positives=train_positives,
            train_negatives=None,
            train_labels=None,
            validation_anchors=validation_anchors,
            validation_positives=validation_positives,
            validation_negatives=None,
            validation_labels=None,
            truncate=truncate,
            margin=DEFAULT,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
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

    def train_with_labeled_pairs(
        self,
        train_anchors: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_others: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_labels: None | OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_anchors: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_others: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_labels: None | OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        loss: type[torch.nn.Module] | Default = AUTO,
        optim: OptimizerNames | str = "adamw_torch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        lr_scheduler_type: SchedulerType | str = "linear",
        warmup_steps: int = 0,
        neftune_noise_alpha: None | float = None,
        seed: int = 42,
        **kwargs,
    ) -> "TrainSentenceTransformer":
        self._setup_folder_and_resume(
            train_anchors=train_anchors,
            train_positives=train_others,
            train_negatives=None,
            train_labels=train_labels,
            validation_anchors=validation_anchors,
            validation_positives=validation_others,
            validation_negatives=None,
            validation_labels=validation_labels,
            truncate=truncate,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
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

    def _load_model(
        self,
        label2id: None | dict[Any, int] = None,
        id2label: None | dict[int, Any] = None,
        is_multi_target: bool = False,
        with_optimizations: bool = True,
    ) -> SentenceTransformer:
        # Load model metadata
        self._load_model_metadata()

        # Load model
        log_if_timeout = RunIfTimeout(
            partial(
                lambda self: self.logger.info("Loading trained model from disk..."),
                self,
            ),
            timeout=10.0,
        )
        if self.peft_config:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import PeftModel

            model = SentenceTransformer(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                device="cpu" if isinstance(self.device, list) else self.device,
                **self.kwargs,
            )
            model[0].tokenizer = get_tokenizer(
                _normalize_model_name(self.model_name),
                revision=None,
                trust_remote_code=self.trust_remote_code,
            )
            model.max_seq_length = (
                get_model_max_context_length(
                    model_name=self.model_name, config=model[0].auto_model.config
                )
                if model.max_seq_length is None
                else model.max_seq_length
            )
            model = PeftModel.from_pretrained(
                model,
                model_id=os.path.join(self._output_folder_path, "_model"),
                torch_dtype=self.dtype,
                **self.kwargs,
            )
        else:
            model = SentenceTransformer(
                os.path.join(self._output_folder_path, "_model"),
                trust_remote_code=self.trust_remote_code,
                device="cpu" if isinstance(self.device, list) else self.device,
                **self.kwargs,
            )
        self.max_seq_length = model.max_seq_length

        # Set model dtype
        model = model.to(self.dtype)

        # Switch model to eval mode
        model.eval()

        if with_optimizations:
            # Torch compile
            # torch._dynamo.config.suppress_errors = True
            # model = torch.compile(model)
            pass

        # Finished loading
        log_if_timeout.stop(
            partial(lambda self: self.logger.info("Finished loading."), self)
        )

        return model

    def export_to_disk(
        self, path: str, adapter_only: bool = False
    ) -> SentenceTransformer:
        return super().export_to_disk(path=path, adapter_only=adapter_only)

    def _publish_info(
        self, repo_id: str, branch: None | str = None, adapter_only: bool = False
    ) -> dict[str, Any]:  # pragma: no cover
        publish_info = super()._publish_info(
            repo_id=repo_id, branch=branch, adapter_only=adapter_only
        )
        publish_info["pipeline_tag"] = "sentence-similarity"
        publish_info["library_name"] = (
            "peft" if (self.peft_config and adapter_only) else "sentence-transformers"
        )
        publish_info["tags"] += ["sentence-transformers", "feature-extraction"]
        examples = [
            (
                ex["Train Anchors"],
                [
                    ex[c]
                    for c in ["Train Positives", "Train Negatives", "Train Others"]
                    if c in ex
                ],
            )
            for ex in list(Dataset.from_dict(self._examples))
        ]
        first_example = examples[0]
        if self.peft_config and adapter_only:
            body = (
                "## Example Usage\n\n```python3\n"
                f"from sentence_transformers import SentenceTransformer\n"
                f"from sentence_transformers.util import cos_sim\n"
                f"from peft import PeftModel\n"
                "\n"
                f"base_model = SentenceTransformer({repr(self.model_name)})"
                f" # Load base model\n"
                f"base_model.max_seq_length = {self.max_seq_length}\n"
                f"base_model.tokenizer.pad_token = base_model.tokenizer.pad_token or base_model.tokenizer.eos_token\n"
                f"model = PeftModel.from_pretrained(base_model, model_id={repr(repo_id)},"
                f" revision={repr(branch)}) # Apply adapter\n\n"
            )
        else:
            body = (
                "## Example Usage\n\n```python3\n"
                f"from sentence_transformers import SentenceTransformer\n"
                f"from sentence_transformers.util import cos_sim\n"
                "\n"
                f"model = SentenceTransformer({repr(repo_id)})"
                f" # Load model\n\n"
            )
        body += (
            f"input = model.encode({repr(first_example[0])})\n"
            f"others = model.encode({repr(first_example[1])})\n"
            "print(cos_sim(input, others))\n"
        )
        body += "```"
        publish_info["body"] = body
        widget_examples = [
            (
                f'example_title: "Example {str(example_idx + 1)}"\n'
                f"    source_sentence: {json.dumps(str(example[0]))}\n"
                f"    sentences:\n"
            )
            + "\n".join([f"      - {json.dumps(str(x))}" for x in example[1]])
            for example_idx, example in enumerate(examples)
        ]
        publish_info["widget_examples"] = widget_examples
        return publish_info

    def publish_to_hf_hub(  # type:ignore[override]
        self,
        repo_id: str,
        private: bool = False,
        token: None | str = None,
        adapter_only: bool = False,
        is_synthetic: bool = True,
        **kwargs,
    ) -> str:  # pragma: no cover
        return super().publish_to_hf_hub(
            repo_id=repo_id,
            branch=None,
            private=private,
            token=token,
            adapter_only=adapter_only,
            **kwargs,
        )

    @property
    def model(self) -> SentenceTransformer:
        return super().model

    @cached_property
    def citation(self) -> None | list[str]:
        citations = _TrainHFBase.citation.func(self) or []  # type: ignore[attr-defined]
        citations.append(
            """
@inproceedings{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural"""
            """ Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
}
        """.strip()
        )
        return citations


__all__ = ["TrainSentenceTransformer"]
