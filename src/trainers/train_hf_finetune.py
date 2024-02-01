import os
from typing import Any

import torch

from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..utils.arg_utils import AUTO, Default
from ..utils.import_utils import ignore_transformers_warnings
from ._train_hf_base import (
    CustomDataCollatorWithPadding,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    _prepare_inputs_and_outputs,
    _start_hf_trainer,
    _TrainHFBase,
    _wrap_trainer_cls,
    get_logging_callback,
)

with ignore_transformers_warnings():
    from transformers import (
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        PreTrainedModel,
        Trainer,
    )
    from transformers.trainer_callback import PrinterCallback
    from transformers.training_args import OptimizerNames, SchedulerType
    from transformers.utils.quantization_config import QuantizationConfigMixin


class TrainHFFineTune(_TrainHFBase):
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
        if self.peft_config:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import TaskType

            if self._is_encoder_decoder:  # pragma: no cover
                self.peft_config.task_type = TaskType.SEQ_2_SEQ_LM
            else:
                self.peft_config.task_type = TaskType.CAUSAL_LM

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
        data_collator = kwargs.pop("data_collator", None)

        with ignore_transformers_warnings():
            from transformers import Seq2SeqTrainer

        # Prepare datasets
        assert (
            self._is_encoder_decoder or truncate
        ), "`truncate=False` is not supported for this model."
        train_dataset, validation_dataset, _, _ = _prepare_inputs_and_outputs(
            self,
            train_columns={
                (
                    "input_ids" if self._is_encoder_decoder else "train_input",
                    "Train Input",
                ): train_input,
                (
                    "decoder_labels" if self._is_encoder_decoder else "train_output",
                    "Train Output",
                ): train_output,
            },
            validation_columns={
                (
                    "input_ids" if self._is_encoder_decoder else "validation_input",
                    "Validation Input",
                ): validation_input,
                (
                    "decoder_labels"
                    if self._is_encoder_decoder
                    else "validation_output",
                    "Validation Output",
                ): validation_output,
            },
            truncate=truncate,
            causal=(not self._is_encoder_decoder),
        )

        # Prepare compute metrics
        def compute_perplexity_metrics(eval_pred):
            preds, labels = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = torch.tensor(preds)
            labels = torch.tensor(labels)
            if self._is_encoder_decoder:
                nll = torch.nn.functional.cross_entropy(
                    input=preds.view(-1, preds.size(-1)), target=labels.view(-1)
                )
            else:
                shift_preds = preds[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                nll = torch.nn.functional.cross_entropy(
                    input=shift_preds.view(-1, shift_preds.size(-1)),
                    target=shift_labels.view(-1),
                )
            return {"perplexity": torch.exp(nll)}

        compute_metrics = (
            kwargs.pop("compute_metrics", None) or compute_perplexity_metrics
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
        model = self._create_model()

        # Prepare training arguments
        if self._is_encoder_decoder:
            training_args_cls = Seq2SeqTrainingArguments
        else:
            training_args_cls = TrainingArguments
        training_args = training_args_cls(
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
            or "eval_perplexity",
            greater_is_better=kwargs.pop("greater_is_better", False),
            load_best_model_at_end=kwargs.pop("load_best_model_at_end", True),
            seed=seed,
            neftune_noise_alpha=neftune_noise_alpha,
            **kwargs,
        )

        # Setup trainer
        if self._is_encoder_decoder:
            # Prepare data collator
            data_collator = data_collator or DataCollatorForSeq2Seq(
                model=model, tokenizer=self.tokenizer, return_tensors="pt"
            )
            trainer_cls = trainer_cls or Seq2SeqTrainer
            trainer_args = {"data_collator": data_collator}
        else:
            data_collator = data_collator or CustomDataCollatorWithPadding(
                tokenizer=self.tokenizer,
                fields_to_pad=[
                    {"name": "input_ids", "output_name": "input_ids"},
                    {
                        "name": "labels",
                        "output_name": "labels",
                        "pad_token_id": -100,
                        "keep_first_pad_token": True,
                    },
                ],
            )
            trainer_cls = trainer_cls or Trainer
            trainer_args = {"data_collator": data_collator}
        trainer = _wrap_trainer_cls(trainer_cls=trainer_cls, **trainer_override_kwargs)(
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            model=model,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=training_args,
            **trainer_args,
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
    ) -> "TrainHFFineTune":
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

    def export_to_disk(self, path: str, adapter_only: bool = False) -> PreTrainedModel:
        return super().export_to_disk(path=path, adapter_only=adapter_only)

    def _publish_info(
        self, repo_id: str, branch: None | str = None, adapter_only: bool = False
    ) -> dict[str, Any]:  # pragma: no cover
        publish_info = super()._publish_info(
            repo_id=repo_id, branch=branch, adapter_only=adapter_only
        )
        if self.chat_prompt_template:
            # Disabling using this widget for now because it is buggy on the HF Hub
            # site.
            # publish_info["pipeline_tag"] = "conversational"
            publish_info["pipeline_tag"] = (
                "text2text-generation"
                if self._is_encoder_decoder
                else "text-generation"
            )
        else:
            publish_info["pipeline_tag"] = (
                "text2text-generation"
                if self._is_encoder_decoder
                else "text-generation"
            )
        auto_cls_name = self.auto_cls.__name__
        if self.chat_prompt_template:
            body = (
                "## Example Usage\n\n```python3\n"
                f"from transformers import {auto_cls_name}, AutoTokenizer, pipeline,"
                " Conversation\n"
            )
        else:
            body = (
                "## Example Usage\n\n```python3\n"
                f"from transformers import {auto_cls_name}, AutoTokenizer, pipeline\n"
            )
        if self._is_encoder_decoder:
            padding_side = ""
            return_full_text = ""
        else:
            padding_side = "tokenizer.padding_side = 'left'\n"
            return_full_text = ", return_full_text=False"
        if self.peft_config and adapter_only:
            body += (
                f"from peft import PeftModel\n"
                "\n"
                f"tokenizer = AutoTokenizer.from_pretrained({repr(repo_id)},"
                f" revision={repr(branch)}) # Load tokenizer\n"
                f"{padding_side}"
                f"base_model = {auto_cls_name}.from_pretrained({repr(self.model_name)},"
                f" revision={repr(self.revision)}) # Load base model\n"
                f"model = PeftModel.from_pretrained(base_model, model_id={repr(repo_id)},"
                f" revision={repr(branch)}) # Apply adapter\n"
            )
        else:
            body += (
                "\n"
                f"tokenizer = AutoTokenizer.from_pretrained({repr(repo_id)},"
                f" revision={repr(branch)}) # Load tokenizer\n"
                f"{padding_side}"
                f"model = {auto_cls_name}.from_pretrained({repr(repo_id)},"
                f" revision={repr(branch)}) # Load model\n"
            )
        if self.chat_prompt_template:
            # Disabling using this widget for now because it is buggy on the HF Hub
            # site.
            # body += (
            #     f"pipe = pipeline('conversational', model=model,"
            #     " tokenizer=tokenizer)\n\n"
            #     f"conversation = Conversation({repr(self._examples['Train Input'][0])})\n"
            #     f"print(pipe(conversation, max_length={self.tokenizer.model_max_length},"
            #     f" do_sample=False).messages[-1]['content'])\n"
            # )
            body += (
                f"pipe = pipeline({repr(publish_info['pipeline_tag'])}, model=model,"
                " tokenizer=tokenizer,"
                f" pad_token_id=tokenizer.pad_token_id{return_full_text})\n\n"
                f"inputs = {repr(self._examples['Train Input'][:1])}\n"
                f"prompts = [tokenizer.apply_chat_template([{{'role': 'user',"
                f" 'content': i}}], tokenize=False, add_generation_prompt=True)"
                f" for i in inputs]\n"
                f"print(pipe(prompts, max_length={self.tokenizer.model_max_length},"
                f" do_sample=False))\n"
            )
        else:
            body += (
                f"pipe = pipeline({repr(publish_info['pipeline_tag'])}, model=model,"
                " tokenizer=tokenizer,"
                f" pad_token_id=tokenizer.pad_token_id{return_full_text})\n\n"
                f"inputs = {repr(self._examples['Train Input'][:1])}\n"
                f"print(pipe(inputs, max_length={self.tokenizer.model_max_length},"
                f" do_sample=False))\n"
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


__all__ = ["TrainHFFineTune"]
