import dataclasses
import json
import os
import sys
from copy import deepcopy
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Type

import torch
from datasets import IterableDataset

from .. import DataDreamer
from .._cachable._cachable import _is_primitive
from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..utils.arg_utils import AUTO, DEFAULT, Default, default_to
from ..utils.background_utils import RunIfTimeout
from ..utils.hf_model_utils import get_base_model_from_peft_model, validate_peft_config
from ..utils.import_utils import ignore_transformers_warnings
from ._train_hf_base import (
    _prepare_inputs_and_outputs,
    _start_hf_trainer,
    _TrainHFBase,
    get_logging_callback,
)
from ._vendored._setfit_helper import get_peft_model_cls  # type:ignore[attr-defined]
from .train_hf_classifier import TrainHFClassifier
from .train_sentence_transformer import TrainSentenceTransformer

with ignore_transformers_warnings():
    from sentence_transformers import SentenceTransformer
    from setfit import SetFitModel, logging as setfit_logging
    from transformers import PreTrainedModel
    from transformers.trainer_callback import EarlyStoppingCallback, PrinterCallback


class SetFitModelWrapper(SetFitModel):
    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "SetFitModelWrapper":
        model = SetFitModel.from_pretrained(*args, **kwargs)
        parent_fields = set([f.name for f in dataclasses.fields(SetFitModel) if f.init])
        return SetFitModelWrapper(
            **{k: v for k, v in model.__dict__.items() if k in parent_fields}
        )

    def _save_pretrained(self, save_directory: Path | str) -> None:
        import joblib

        with ignore_transformers_warnings():
            from setfit.modeling import CONFIG_NAME, MODEL_HEAD_NAME

        save_directory = str(save_directory)
        # Save the config
        config_path = os.path.join(save_directory, CONFIG_NAME)
        with open(config_path, "w") as f:
            json.dump(
                {
                    attr_name: getattr(self, attr_name)
                    for attr_name in self.attributes_to_save
                    if hasattr(self, attr_name)
                },
                f,
                indent=2,
            )
        # Save the body
        if not isinstance(self.model_body, SentenceTransformer):  # Must be PEFT
            # Save both adapter and base model
            TrainSentenceTransformer._save_resource(
                self, resource=self.model_body, path=save_directory
            )
            os.rename(
                os.path.join(save_directory, "adapter_config.json"),
                os.path.join(save_directory, "adapter_config.json.hidden"),
            )  # Temporarily hide it, until we can reload it
            TrainSentenceTransformer._save_resource(
                self,
                resource=get_base_model_from_peft_model(self.model_body),
                path=save_directory,
            )
        else:
            TrainSentenceTransformer._save_resource(
                self, resource=self.model_body, path=save_directory
            )

        # Move the head to the CPU before saving
        if self.has_differentiable_head:
            self.model_head.to("cpu")
        # Save the classification head
        joblib.dump(self.model_head, str(Path(save_directory) / MODEL_HEAD_NAME))
        if self.has_differentiable_head:
            self.model_head.to(self.device)


class TrainSetFitClassifier(TrainHFClassifier):
    def __init__(
        self,
        name: str,
        model_name: str,
        multi_target_strategy: None | str | Default = AUTO,
        device: None | int | str | torch.device = None,
        dtype: None | str | torch.dtype = None,
        peft_config: None | Any = None,
        force: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
        **kwargs,
    ):
        cls_name = self.__class__.__name__
        assert not isinstance(
            device, list
        ), f"Training on multiple devices is not supported for {cls_name}."
        _TrainHFBase.__init__(
            self,
            name=name,
            model_name=model_name,
            chat_prompt_template=None,
            system_prompt=None,
            revision=None,
            trust_remote_code=False,
            device=device or "cpu",
            dtype=dtype,
            quantization_config=None,
            peft_config=peft_config,
            force=force,
            verbose=verbose,
            log_level=log_level,
            **kwargs,
        )
        self.multi_target_strategy = multi_target_strategy
        self.chat_prompt_template = None
        self.system_prompt = None
        if self.peft_config:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import TaskType

            self.peft_config.task_type = TaskType.FEATURE_EXTRACTION

    @property
    def resumable(self) -> bool:
        return False

    @property
    def auto_cls(self) -> Type:
        return SetFitModelWrapper

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
    ) -> SetFitModel:
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
        assert label2id is not None
        model = self.auto_cls.from_pretrained(
            self.model_name,
            device=default_to(device, self.device),
            multi_target_strategy=default_to(
                self.multi_target_strategy, "one-vs-rest" if is_multi_target else None
            ),
            use_differentiable_head=True,
            head_params={"out_features": len(label2id)},
            labels=list(label2id.keys()),
            **self.kwargs,
        )

        # Set model dtype
        model.model_body = model.model_body.to(self.dtype)
        model.model_head = model.model_head.to(self.dtype)

        # Create PeftModel if peft_config
        if self.peft_config:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import prepare_model_for_kbit_training

            if self.quantization_config:  # pragma: no cover
                model.model_body = prepare_model_for_kbit_training(model.model_body)
            model.model_body = get_peft_model_cls()(
                model=model.model_body,
                peft_config=validate_peft_config(model.model_body, self.peft_config),
            )

        # Finished loading
        log_if_timeout.stop(
            partial(lambda self: self.logger.info("Finished loading."), self)
        )

        return model

    def _publish_resource(
        self,
        resource: Any,
        repo_id: str,
        branch: None | str = None,
        private: bool = False,
        **kwargs,
    ):  # pragma: no cover
        resource.push_to_hub(repo_id=repo_id, branch=branch, private=private, **kwargs)

    def _train(  # type:ignore[override]
        self,
        train_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_output: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_output: OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        sampling_strategy: str = "oversampling",
        end_to_end: bool = False,
        epochs: float | tuple[float, float] = 3.0,
        batch_size: int | tuple[int, int] = 8,
        body_learning_rate: float | tuple[float, float] = 1e-3,
        head_learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        seed: int = 42,
        **kwargs,
    ):
        with ignore_transformers_warnings():
            from setfit import Trainer, TrainingArguments
            from transformers.trainer_callback import ProgressCallback

        # Prepare datasets
        (
            train_dataset,
            validation_dataset,
            label2id,
            is_multi_target,
        ) = _prepare_inputs_and_outputs(
            self,
            train_columns={
                ("text", "Train Input"): train_input,
                ("label", "Train Output"): train_output,
            },
            validation_columns={
                ("text", "Validation Input"): validation_input,
                ("label", "Validation Output"): validation_output,
            },
            truncate=truncate,
        )
        id2label = {v: k for k, v in label2id.items()}
        assert (
            len(id2label) > 1
        ), "There must be at least 2 output labels in your dataset."

        # Prepare metrics
        metric = kwargs.pop("metric", "f1")
        metric_kwargs = kwargs.pop(
            "metric_kwargs", {"average": "micro"} if metric == "f1" else {}
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

        # Prepare training arguments
        training_args = TrainingArguments(
            report_to=kwargs.pop("report_to", None),
            run_name=f"DataDreamer - {self.name}",
            show_progress_bar=True,
            output_dir=os.path.join(self._output_folder_path, "_checkpoints"),
            num_epochs=epochs,
            batch_size=batch_size,
            body_learning_rate=body_learning_rate,
            head_learning_rate=head_learning_rate,
            l2_weight=weight_decay,
            warmup_proportion=0,  # Placeholder, gets set later
            sampling_strategy=sampling_strategy,
            end_to_end=end_to_end,
            max_length=sys.maxsize if not truncate else self.tokenizer.model_max_length,
            logging_strategy=kwargs.pop("logging_strategy", None) or "steps",
            logging_steps=kwargs.pop("logging_steps", 1),
            evaluation_strategy=kwargs.pop("evaluation_strategy", None) or "epoch",
            save_strategy=kwargs.pop("save_strategy", None) or "epoch",
            save_total_limit=kwargs.pop("save_total_limit", 1),
            load_best_model_at_end=kwargs.pop("load_best_model_at_end", True),
            seed=seed,
            **kwargs,
        )
        if kwargs.get("max_steps", None) is not None:  # pragma: no cover
            total_train_steps = kwargs["max_steps"]
        else:
            if warmup_steps > 0:  # pragma: no cover
                assert not isinstance(train_dataset, IterableDataset), (
                    "The train input columns must be of known length. Use `total_num_rows` if"
                    " using iterable datasets."
                )
                total_train_steps = (
                    len(train_dataset) * training_args.embedding_num_epochs
                )
            else:
                total_train_steps = 1
        training_args.warmup_proportion = min(
            max(warmup_steps / total_train_steps, 0.0), 1.0
        )

        # Prepare model
        self.seed = training_args.seed
        model = self._create_model(
            label2id=label2id, id2label=id2label, is_multi_target=is_multi_target
        )

        # Setup trainer
        class CustomTrainer(Trainer):
            def train_classifier(trainer, *args, **kwargs):
                self.logger.info("Finished training SetFit model body (embeddings).")

                # Reload back PEFT adapter if it got removed when loading the best
                # checkpoint at the end of SetFit model body training
                if self.peft_config and isinstance(
                    model.model_body, SentenceTransformer
                ):
                    # Two warnings we can't silence are thrown by peft at import-time so
                    # we import this library only when needed
                    with ignore_transformers_warnings():
                        from peft import PeftModel

                    best_model_checkpoint = trainer.state.best_model_checkpoint
                    os.rename(
                        os.path.join(
                            best_model_checkpoint, "adapter_config.json.hidden"
                        ),
                        os.path.join(best_model_checkpoint, "adapter_config.json"),
                    )  # Un-hide
                    model.model_body = PeftModel.from_pretrained(
                        model.model_body,
                        model_id=best_model_checkpoint,
                        torch_dtype=self.dtype,
                        **self.kwargs,
                    )
                    model.model_body.forward = partial(
                        get_peft_model_cls().forward, model.model_body
                    )

                self.logger.info("Training SetFit model head (classifier)...")
                if not DataDreamer.ctx.hf_log:
                    setfit_logging_prog_bar = setfit_logging.is_progress_bar_enabled()
                    setfit_logging.enable_progress_bar()
                results = super().train_classifier(*args, **kwargs)
                if not DataDreamer.ctx.hf_log:
                    if setfit_logging_prog_bar:  # pragma: no cover
                        setfit_logging.enable_progress_bar()
                    else:
                        setfit_logging.disable_progress_bar()
                return results

            def evaluate(self, *args, **kwargs):
                metrics = {
                    "epoch": "Final"
                    if "final" in kwargs
                    else round(self.state.epoch or 0.0, 2)
                }
                kwargs.pop("final", None)
                for k, v in super().evaluate(*args, **kwargs).items():
                    metrics[f"eval_{k}"] = v
                self.callback_handler.on_log(
                    training_args, self.state, self.control, metrics
                )
                return metrics

        trainer = CustomTrainer(
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            model=model,
            metric=metric,
            metric_kwargs=metric_kwargs,
            callbacks=callbacks,
            args=training_args,
        )
        trainer.remove_callback(PrinterCallback)
        trainer.remove_callback(ProgressCallback)

        # Start the trainer
        self.logger.info("Training SetFit model body (embeddings)...")
        _start_hf_trainer(self, trainer)
        self.logger.info("Running final trained SetFit model evaluation...")
        trainer.evaluate(final=True)  # Run a final evaluation

        # Save the model to disk
        json_safe_training_args = deepcopy(training_args)
        json_safe_training_args.__dict__.update(
            {
                k: (v if _is_primitive(v) else str(v))
                for k, v in training_args.__dict__.items()
            }
        )

        self._save_model(
            training_args=json_safe_training_args,
            model=trainer.model,
            tokenizer=self.tokenizer,
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
        sampling_strategy: str = "oversampling",
        end_to_end: bool = False,
        epochs: float | tuple[float, float] = 3.0,
        batch_size: int | tuple[int, int] = 8,
        body_learning_rate: float | tuple[float, float] = 1e-3,
        head_learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        seed: int = 42,
        **kwargs,
    ) -> "TrainSetFitClassifier":
        self._setup_folder_and_resume(
            train_input=train_input,
            train_output=train_output,
            validation_input=validation_input,
            validation_output=validation_output,
            truncate=truncate,
            sampling_strategy=sampling_strategy,
            end_to_end=end_to_end,
            epochs=epochs,
            batch_size=batch_size,
            body_learning_rate=body_learning_rate,
            head_learning_rate=head_learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
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
    ) -> SetFitModel:
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
        assert label2id is not None
        if self.peft_config:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import PeftModel

            if os.path.exists(
                os.path.join(
                    os.path.join(self._output_folder_path, "_model"),
                    "adapter_config.json",
                )
            ):
                os.rename(
                    os.path.join(
                        os.path.join(self._output_folder_path, "_model"),
                        "adapter_config.json",
                    ),
                    os.path.join(
                        os.path.join(self._output_folder_path, "_model"),
                        "adapter_config.json.hidden",
                    ),
                )  # Hide
            model = self.auto_cls.from_pretrained(
                os.path.join(self._output_folder_path, "_model"),
                device=self.device,
                multi_target_strategy=default_to(
                    self.multi_target_strategy,
                    "one-vs-rest" if is_multi_target else None,
                ),
                use_differentiable_head=True,
                labels=list(label2id.keys()),
                **self.kwargs,
            )
            if os.path.exists(
                os.path.join(
                    os.path.join(self._output_folder_path, "_model"),
                    "adapter_config.json.hidden",
                )
            ):
                os.rename(
                    os.path.join(
                        os.path.join(self._output_folder_path, "_model"),
                        "adapter_config.json.hidden",
                    ),
                    os.path.join(
                        os.path.join(self._output_folder_path, "_model"),
                        "adapter_config.json",
                    ),
                )  # Un-hide
            model.model_body = PeftModel.from_pretrained(
                model.model_body,
                model_id=os.path.join(self._output_folder_path, "_model"),
                torch_dtype=self.dtype,
                **self.kwargs,
            )
            model.model_body = model.model_body.merge_and_unload()
        else:
            model = self.auto_cls.from_pretrained(
                os.path.join(self._output_folder_path, "_model"),
                device=self.device,
                multi_target_strategy=default_to(
                    self.multi_target_strategy,
                    "one-vs-rest" if is_multi_target else None,
                ),
                use_differentiable_head=True,
                labels=list(label2id.keys()),
                **self.kwargs,
            )

        # Set model dtype
        model.model_body = model.model_body.to(self.dtype)
        model.model_head = model.model_head.to(self.dtype)

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

    def export_to_disk(  # type:ignore[override]
        self, path: str
    ) -> PreTrainedModel:
        return super().export_to_disk(path=path, adapter_only=False)

    def _publish_info(
        self, repo_id: str, branch: None | str = None, adapter_only: bool = False
    ) -> dict[str, Any]:  # pragma: no cover
        publish_info = super()._publish_info(
            repo_id=repo_id, branch=branch, adapter_only=adapter_only
        )
        publish_info["library_name"] = "setfit"
        publish_info["tags"] += [
            "setfit",
            "sentence-transformers",
            "generated_from_setfit_trainer",
        ]

        if self.is_multi_target:
            pred = (
                "print([[label for score, label in zip(r, model.labels)"
                " if score == 1] for r in"
                " model.predict(inputs).detach().cpu().tolist()])\n"
            )
        else:
            pred = "print(model.predict(inputs))\n"
        body = (
            "## Example Usage\n\n```python3\n"
            f"from setfit import SetFitModel\n"
            "\n"
            f"model = SetFitModel.from_pretrained({repr(repo_id)},"
            f" revision={repr(branch)}) # Load model\n"
            f"inputs = {repr(self._examples['Train Input'][:1])}\n"
            f"{pred}"
            "```"
        )
        publish_info["body"] = body
        return publish_info

    def publish_to_hf_hub(  # type:ignore[override]
        self,
        repo_id: str,
        branch: None | str = None,
        private: bool = False,
        token: None | str = None,
        is_synthetic: bool = True,
        **kwargs,
    ) -> str:  # pragma: no cover
        return super().publish_to_hf_hub(
            repo_id=repo_id,
            branch=branch,
            private=private,
            token=token,
            adapter_only=False,
            **kwargs,
        )

    @property
    def model(self) -> SetFitModel:
        return super().model

    @cached_property
    def citation(self) -> None | list[str]:
        citations = TrainSentenceTransformer.citation.func(self) or []  # type: ignore[attr-defined]
        citations.append(
            """
@misc{https://doi.org/10.48550/arxiv.2209.11055,
  doi = {10.48550/ARXIV.2209.11055},
  url = {https://arxiv.org/abs/2209.11055},
  author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and"
  " Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences,"
  " FOS: Computer and information sciences},
  title = {Efficient Few-Shot Learning Without Prompts},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
        """.strip()
        )
        return citations


__all__ = ["TrainSetFitClassifier"]
