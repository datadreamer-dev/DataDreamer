import json
import logging
import os
import sys
from copy import copy
from functools import cached_property, partial
from io import BytesIO
from itertools import chain
from shutil import copy2
from typing import TYPE_CHECKING, Any, Callable, Type, cast

import numpy as np
import torch
from datasets import Dataset, IterableDataset, Value, concatenate_datasets
from datasets.fingerprint import Hasher
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .. import DataDreamer
from ..datasets import (
    OutputDatasetColumn,
    OutputIterableDataset,
    OutputIterableDatasetColumn,
)
from ..datasets.datasets import _SizedIterableDataset, get_sized_dataset
from ..logging import logger
from ..steps import DataSource
from ..utils.arg_utils import AUTO, DEFAULT, Default, default_to
from ..utils.background_utils import RunIfTimeout
from ..utils.device_utils import (
    _TrainingArgumentDeviceOverrideMixin,
    get_device_memory_monitoring_callback,
    validate_device,
)
from ..utils.distributed_utils import (
    get_global_rank,
    get_num_nodes_from_distributed_config,
    is_distributed,
    not_distributed_or_main_process,
    save_distributed_model,
    validate_distributed_config,
)
from ..utils.fingerprint_utils import stable_fingerprint
from ..utils.fs_utils import clear_dir
from ..utils.hf_chat_prompt_templates import set_hf_chat_template
from ..utils.hf_hub_utils import (
    get_citation_info,
    get_license_info,
    get_model_card_url,
    get_readme_contents,
    hf_hub_login,
    prepare_to_publish,
)
from ..utils.hf_model_utils import (
    HF_TRANSFORMERS_CITATION,
    PEFT_CITATION,
    convert_dtype,
    get_config,
    get_model_prompt_template,
    get_tokenizer,
    is_encoder_decoder,
    validate_peft_config,
)
from ..utils.import_utils import ignore_transformers_warnings
from .trainer import Trainer as DataDreamerTrainer

with ignore_transformers_warnings():
    from optimum.bettertransformer import BetterTransformer
    from optimum.bettertransformer.models import BetterTransformerManager
    from setfit import logging as setfit_logging
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        PreTrainedModel,
        PreTrainedTokenizer,
        TrainerCallback,
        logging as hf_transformers_logging,
    )
    from transformers.utils.quantization_config import QuantizationConfigMixin

from transformers import (
    Seq2SeqTrainingArguments as _Seq2SeqTrainingArguments,
    TrainingArguments as _TrainingArguments,
)

if TYPE_CHECKING:  # pragma: no cover
    with ignore_transformers_warnings():
        from transformers import Trainer


class TrainingArguments(_TrainingArgumentDeviceOverrideMixin, _TrainingArguments):
    pass


class Seq2SeqTrainingArguments(
    _TrainingArgumentDeviceOverrideMixin, _Seq2SeqTrainingArguments
):
    pass


def _wrap_trainer_cls(
    trainer_cls: Type["Trainer"],
    optimizers: tuple[None | Optimizer, None | LambdaLR] = (None, None),
    optimizer: None | Optimizer = None,
    lr_scheduler: None | LambdaLR = None,
    compute_loss: None | Callable = None,
) -> Type["Trainer"]:
    class WrappedTrainer(trainer_cls):
        def create_optimizer(self):
            final_optimizer = optimizer or optimizers[0]
            if final_optimizer is not None:  # pragma: no cover
                self.optimizer = final_optimizer
            else:
                super().create_optimizer()

        def create_scheduler(
            self, num_training_steps: int, optimizer: None | Optimizer = None
        ):
            final_lr_scheduler = lr_scheduler or optimizers[1]
            if final_lr_scheduler is not None:  # pragma: no cover
                self.lr_scheduler = final_lr_scheduler
            else:
                super().create_scheduler(
                    num_training_steps=num_training_steps, optimizer=optimizer
                )

        def compute_loss(self, model, inputs, return_outputs=False):
            if compute_loss is not None:  # pragma: no cover
                return compute_loss(model, inputs, return_outputs=return_outputs)
            else:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs
                )

    return WrappedTrainer


def _prepare_inputs_and_outputs(  # noqa: C901
    self: "_TrainHFBase",
    train_columns: dict[
        tuple[str, str], OutputDatasetColumn | OutputIterableDatasetColumn
    ],
    validation_columns: dict[
        tuple[str, str], OutputDatasetColumn | OutputIterableDatasetColumn
    ],
    truncate: bool = False,
    causal: bool = False,
    dpo: bool = False,
    reward_pairs: bool = False,
    reward_scores: bool = False,
) -> tuple[
    Dataset | IterableDataset | _SizedIterableDataset,
    Dataset | IterableDataset | _SizedIterableDataset,
    dict[Any, int],
    bool,
]:
    num_proc = (
        (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        )
        if sys.platform != "darwin"
        else 1
    )
    label2id: dict[Any, int] = {}
    is_multi_target: bool = False

    def get_train_column(
        column_name: str,
    ) -> OutputDatasetColumn | OutputIterableDatasetColumn:
        for (train_column_name, _), train_column in train_columns.items():
            if train_column_name == column_name:
                return train_column
        raise KeyError(f"Train column {column_name} not found.")  # pragma: no cover

    def get_validation_column(
        column_name: str,
    ) -> OutputDatasetColumn | OutputIterableDatasetColumn:
        for (
            validation_column_name,
            _,
        ), validation_column in validation_columns.items():
            if validation_column_name == column_name:
                return validation_column
        raise KeyError(
            f"Validation column {column_name} not found."
        )  # pragma: no cover

    def apply_chat_prompt_template(prompt: str) -> str:
        return (
            cast(str, self.chat_prompt_template)
            .replace("{{system_prompt}}", self.system_prompt or "")
            .replace("{{prompt}}", prompt)
        )

    def tokenize_function(
        examples,
        column_name: str,
        new_column_name: str,
        causal: bool,
        reward_scores: bool,
    ):  # pragma: no cover
        if reward_scores:
            prompt, completion = examples[column_name]
            if self.chat_prompt_template:
                prompt = apply_chat_prompt_template(prompt)
            input_ids = self.tokenizer(
                prompt + completion,
                truncation=truncate,
                padding=False,
                add_special_tokens=True,
            )["input_ids"]
            return {
                "input_ids": input_ids[: self.tokenizer.model_max_length]
                if truncate
                else input_ids,
                "labels": examples["label"],
            }
        elif causal:
            prompt, completion = examples[column_name]
            if self.chat_prompt_template:
                prompt = apply_chat_prompt_template(prompt)
            prompt_input_ids = self.tokenizer(
                prompt, truncation=truncate, padding=False, add_special_tokens=True
            )["input_ids"]
            completion_input_ids = self.tokenizer(
                completion, truncation=truncate, padding=False, add_special_tokens=False
            )["input_ids"] + [self.tokenizer.eos_token_id]
            prompt_labels = [-100] * len(prompt_input_ids)
            input_ids = prompt_input_ids + completion_input_ids
            labels = prompt_labels + completion_input_ids
            return {
                "input_ids": input_ids[: self.tokenizer.model_max_length]
                if truncate
                else input_ids,
                "labels": labels[: self.tokenizer.model_max_length]
                if truncate
                else labels,
            }
        elif new_column_name in ["decoder_labels"]:
            return {
                "labels": self.tokenizer(
                    examples[column_name],
                    truncation=truncate,
                    padding=False,
                    add_special_tokens=True,
                )["input_ids"]
            }
        else:
            prompts = examples[column_name]
            if self.chat_prompt_template:
                prompts = list(map(apply_chat_prompt_template, prompts))
            tokenizer_results = self.tokenizer(
                prompts, truncation=truncate, padding=False, add_special_tokens=True
            )
            return {
                new_column_name: tokenizer_results["input_ids"],
                f"{new_column_name.replace('input_ids', '')}attention_mask": tokenizer_results[
                    "attention_mask"
                ],
            }

    def tokenize_column_name(
        column_name: str,
        new_column_name: str,
        causal: bool,
        reward_scores: bool = False,
    ) -> Callable:
        return partial(
            tokenize_function,
            column_name=column_name,
            new_column_name=new_column_name,
            causal=causal,
            reward_scores=reward_scores,
        )

    def tokenize_column(
        column: OutputDatasetColumn | OutputIterableDatasetColumn,
        new_column_name: str,
        name: str,
        causal: bool = False,
        reward_scores: bool = False,
    ) -> Dataset | IterableDataset:
        column_name = column.column_names[0]
        return column.step.map(
            name=f"Tokenize {name}",
            function=tokenize_column_name(
                column_name,
                new_column_name=new_column_name,
                causal=causal,
                reward_scores=reward_scores,
            ),
            batched=not causal and not reward_scores,
            remove_columns=column.step.output.column_names,
            total_num_rows=column.num_rows,
            auto_progress=column.num_rows is not None,
            lazy=isinstance(column, OutputIterableDatasetColumn),
            progress_interval=sys.maxsize
            if isinstance(column, OutputIterableDatasetColumn)
            else 120,
            save_num_proc=num_proc,
        ).output.dataset

    def rename_column(
        column: OutputDatasetColumn | OutputIterableDatasetColumn, new_column_name: str
    ) -> Dataset | IterableDataset:
        column_name = column.column_names[0]
        column_dataset = column.step.output.dataset.select_columns(column.column_names)
        return (
            column_dataset.rename_column(column_name, new_column_name)
            if column_name != new_column_name
            else column_dataset
        )

    def label_encode_function(
        _, column_name: str, example: dict[str, Any]
    ) -> dict[str, Any]:  # pragma: no cover
        if isinstance(example[column_name], list):
            row_labels = set(str(label) for label in example[column_name])
            return {
                column_name: [1 if label in row_labels else 0 for label in label2id]
            }
        else:
            return {column_name: label2id[str(example[column_name])]}

    def label2id_column(
        column: OutputDatasetColumn | OutputIterableDatasetColumn,
        new_column_name: str,
        name: str,
    ) -> Dataset | IterableDataset:
        column_name = column.column_names[0]
        return rename_column(
            column.step.map(
                name=f"Encode {name} labels",
                function=partial(
                    label_encode_function, sorted(label2id.keys()), column_name
                ),
                batched=False,
                remove_columns=list(
                    set(column.step.output.column_names).difference(set([column_name]))
                ),
                total_num_rows=column.num_rows,
                auto_progress=column.num_rows is not None,
                lazy=isinstance(column, OutputIterableDatasetColumn),
                progress_interval=sys.maxsize
                if isinstance(column, OutputIterableDatasetColumn)
                else 120,
                save_num_proc=num_proc,
            ).output[column_name],
            new_column_name,
        )

    def process_column(
        column: OutputDatasetColumn | OutputIterableDatasetColumn,
        new_column_name: str,
        name: str,
    ) -> Dataset | IterableDataset:
        if new_column_name == "label" and reward_scores is False:
            return label2id_column(
                column=column, new_column_name=new_column_name, name=name
            )
        else:  # pragma: no cover
            return rename_column(column=column, new_column_name=new_column_name)

    def concatenate_prompts_and_completions(
        dataset: Dataset | IterableDataset,
    ) -> IterableDataset:
        iterable_dataset = (
            dataset.to_iterable_dataset() if isinstance(dataset, Dataset) else dataset
        )
        return iterable_dataset.map(
            lambda row: {"text": [row["prompt"], row["completion"]]},
            remove_columns=["prompt", "completion"],
        )

    # Calculate label2id
    uniq_labels = []
    for (new_column_name, name), column in list(train_columns.items()) + list(
        validation_columns.items()
    ):
        column_name = column.column_names[0]

        def uniqify_labels(labels: set[Any], column_name, example):
            nonlocal is_multi_target
            if isinstance(example[column_name], list):
                is_multi_target = True
                is_new = False
                for label in example[column_name]:
                    if label not in labels:
                        is_new = True
                        labels.add(label)
                return is_new
            else:
                is_new = example[column_name] not in labels
                labels.add(example[column_name])
                return is_new

        if new_column_name == "label" and reward_scores is False:
            uniq_labels_column = column.step.filter(
                name=f"Get all {name} label names",
                function=partial(uniqify_labels, set(), column_name),
                batched=False,
                total_num_rows=column.num_rows,
                auto_progress=column.num_rows is not None,
                lazy=False,
                progress_interval=sys.maxsize
                if isinstance(column, OutputIterableDatasetColumn)
                else 120,
            ).output[column_name]
            uniq_labels_from_column = list(uniq_labels_column)
            uniq_labels += (
                list(chain.from_iterable(uniq_labels_column))
                if len(uniq_labels_from_column) > 0
                and isinstance(uniq_labels_from_column[0], list)
                else uniq_labels_column
            )
    uniq_labels = sorted(set(uniq_labels))
    for label in uniq_labels:
        label2id[str(label)] = len(label2id)

    # Create train and validation datasets
    train_dataset: Dataset | IterableDataset
    validation_dataset: Dataset | IterableDataset
    if reward_pairs:
        # Check if scores are provided
        try:
            get_train_column("train_chosen_scores")
            has_scores = True
        except KeyError:
            has_scores = False

        # Get data collator
        def prepare_for_reward_pairs(row):  # pragma: no cover
            row = row.copy()
            if self.chat_prompt_template:
                row["prompt"] = apply_chat_prompt_template(row["prompt"])
            row["chosen"] = row["prompt"] + row["chosen"]
            row["rejected"] = row["prompt"] + row["rejected"]
            reward_results = {}
            chosen_tokenizer_results = self.tokenizer(
                row["chosen"],
                truncation=truncate,
                padding=False,
                add_special_tokens=True,
            )
            reward_results["input_ids_chosen"] = chosen_tokenizer_results["input_ids"]
            rejected_tokenizer_results = self.tokenizer(
                row["rejected"],
                truncation=truncate,
                padding=False,
                add_special_tokens=True,
            )
            reward_results["input_ids_rejected"] = rejected_tokenizer_results[
                "input_ids"
            ]
            if "chosen_scores" in row and "rejected_scores" in row:
                reward_results["margin"] = row["chosen_scores"] - row["rejected_scores"]
            return reward_results

        # Run data collator
        train_columns_to_combine = [
            rename_column(get_train_column("train_prompts"), "prompt"),
            rename_column(get_train_column("train_chosen"), "chosen"),
            rename_column(get_train_column("train_rejected"), "rejected"),
        ]
        if has_scores:
            train_columns_to_combine.extend(
                [
                    rename_column(
                        get_train_column("train_chosen_scores"), "chosen_scores"
                    ),
                    rename_column(
                        get_train_column("train_rejected_scores"), "rejected_scores"
                    ),
                ]
            )
        train_combine_step = DataSource(
            "Combine Train Prompts, Chosen Generations, and Rejected Generations",
            data=concatenate_datasets(train_columns_to_combine, axis=1),
            total_num_rows=get_train_column("train_prompts").num_rows,
            auto_progress=get_train_column("train_prompts").num_rows is not None,
        )
        train_dataset = train_combine_step.map(
            name="Prepare Train Dataset for Reward Model Training",
            function=prepare_for_reward_pairs,
            batched=False,
            remove_columns=train_combine_step.output.column_names,
            total_num_rows=get_train_column("train_prompts").num_rows,
            auto_progress=get_train_column("train_prompts").num_rows is not None,
            lazy=isinstance(train_combine_step.output, OutputIterableDataset),
            progress_interval=sys.maxsize
            if isinstance(train_combine_step.output, OutputIterableDataset)
            else 120,
            save_num_proc=num_proc,
        ).output.dataset
        validation_columns_to_combine = [
            rename_column(get_validation_column("validation_prompts"), "prompt"),
            rename_column(get_validation_column("validation_chosen"), "chosen"),
            rename_column(get_validation_column("validation_rejected"), "rejected"),
        ]
        if has_scores:
            validation_columns_to_combine.extend(
                [
                    rename_column(
                        get_validation_column("validation_chosen_scores"),
                        "chosen_scores",
                    ),
                    rename_column(
                        get_validation_column("validation_rejected_scores"),
                        "rejected_scores",
                    ),
                ]
            )
        validation_combine_step = DataSource(
            "Combine Validation Prompts, Chosen Generations, and Rejected Generations",
            data=concatenate_datasets(validation_columns_to_combine, axis=1),
            total_num_rows=get_validation_column("validation_prompts").num_rows,
            auto_progress=get_validation_column("validation_prompts").num_rows
            is not None,
        )
        validation_dataset = validation_combine_step.map(
            name="Prepare Validation Dataset for Reward Model Training",
            function=prepare_for_reward_pairs,
            batched=False,
            remove_columns=validation_combine_step.output.column_names,
            total_num_rows=get_validation_column("validation_prompts").num_rows,
            auto_progress=get_validation_column("validation_prompts").num_rows
            is not None,
            lazy=isinstance(validation_combine_step.output, OutputIterableDataset),
            progress_interval=sys.maxsize
            if isinstance(validation_combine_step.output, OutputIterableDataset)
            else 120,
            save_num_proc=num_proc,
        ).output.dataset
    elif dpo:
        if TYPE_CHECKING:  # pragma: no cover
            DPODataCollatorWithPadding: Any = None
        else:
            from ._vendored._dpo_helper import DPODataCollatorWithPadding

        # Get data collator
        data_collator = DPODataCollatorWithPadding(
            tokenizer=self.tokenizer,
            max_length=self.tokenizer.model_max_length if truncate else sys.maxsize,
            max_prompt_length=self.tokenizer.model_max_length
            if truncate
            else sys.maxsize,
            label_pad_token_id=-100,
            padding_value=0,
            truncation_mode="keep_end",
            is_encoder_decoder=self._is_encoder_decoder,
            max_target_length=self.tokenizer.model_max_length
            if truncate
            else sys.maxsize,
        )

        def run_data_collator(row):  # pragma: no cover
            if self.chat_prompt_template:
                row["prompt"] = apply_chat_prompt_template(row["prompt"])
            dpo_results = data_collator.__call__([row])
            for key, value in list(dpo_results.items()):
                if "attention_mask" in key:
                    del dpo_results[key]
                elif isinstance(value, list) and len(value) == 1:
                    dpo_results[key] = value[0]
                elif isinstance(value, torch.Tensor) and len(value.shape) == 2:
                    value = value[0]
                    if truncate:
                        dpo_results[key] = value[: self.tokenizer.model_max_length]
            return dpo_results

        # Run data collator
        train_combine_step = DataSource(
            "Combine Train Prompts, Chosen Generations, and Rejected Generations",
            data=concatenate_datasets(
                [
                    rename_column(get_train_column("train_prompts"), "prompt"),
                    rename_column(get_train_column("train_chosen"), "chosen"),
                    rename_column(get_train_column("train_rejected"), "rejected"),
                ],
                axis=1,
            ),
            total_num_rows=get_train_column("train_prompts").num_rows,
            auto_progress=get_train_column("train_prompts").num_rows is not None,
        )
        train_dataset = train_combine_step.map(
            name="Prepare Train Dataset for DPO",
            function=run_data_collator,
            batched=False,
            total_num_rows=get_train_column("train_prompts").num_rows,
            auto_progress=get_train_column("train_prompts").num_rows is not None,
            lazy=isinstance(train_combine_step.output, OutputIterableDataset),
            progress_interval=sys.maxsize
            if isinstance(train_combine_step.output, OutputIterableDataset)
            else 120,
            save_num_proc=num_proc,
        ).output.dataset
        validation_combine_step = DataSource(
            "Combine Validation Prompts, Chosen Generations, and Rejected Generations",
            data=concatenate_datasets(
                [
                    rename_column(
                        get_validation_column("validation_prompts"), "prompt"
                    ),
                    rename_column(get_validation_column("validation_chosen"), "chosen"),
                    rename_column(
                        get_validation_column("validation_rejected"), "rejected"
                    ),
                ],
                axis=1,
            ),
            total_num_rows=get_validation_column("validation_prompts").num_rows,
            auto_progress=get_validation_column("validation_prompts").num_rows
            is not None,
        )
        validation_dataset = validation_combine_step.map(
            name="Prepare Validation Dataset for DPO",
            function=run_data_collator,
            batched=False,
            total_num_rows=get_validation_column("validation_prompts").num_rows,
            auto_progress=get_validation_column("validation_prompts").num_rows
            is not None,
            lazy=isinstance(validation_combine_step.output, OutputIterableDataset),
            progress_interval=sys.maxsize
            if isinstance(validation_combine_step.output, OutputIterableDataset)
            else 120,
            save_num_proc=num_proc,
        ).output.dataset
    elif reward_scores:
        train_combined = concatenate_datasets(
            [
                rename_column(get_train_column("train_input"), "prompt"),
                rename_column(get_train_column("train_output"), "completion"),
                rename_column(get_train_column("label"), "label").cast_column(
                    "label", Value("float64")
                ),
            ],
            axis=1,
        )
        train_dataset = tokenize_column(
            DataSource(
                "Concatenate Train Prompts and Generations",
                data=concatenate_prompts_and_completions(train_combined),
                total_num_rows=get_train_column("train_input").num_rows,
                auto_progress=get_train_column("train_input").num_rows is not None,
                save=not isinstance(train_combined, IterableDataset),
            ).output["text"],
            "input_ids",
            "Train Dataset",
            reward_scores=True,
        )
        validation_combined = concatenate_datasets(
            [
                rename_column(get_validation_column("validation_input"), "prompt"),
                rename_column(get_validation_column("validation_output"), "completion"),
                rename_column(get_validation_column("label"), "label").cast_column(
                    "label", Value("float64")
                ),
            ],
            axis=1,
        )
        validation_dataset = tokenize_column(
            DataSource(
                "Concatenate Validation Prompts and Generations",
                data=concatenate_prompts_and_completions(validation_combined),
                total_num_rows=get_validation_column("validation_input").num_rows,
                auto_progress=get_validation_column("validation_input").num_rows
                is not None,
                save=not isinstance(validation_combined, IterableDataset),
            ).output["text"],
            "input_ids",
            "Validation Dataset",
            reward_scores=True,
        )
    elif causal:
        train_combined = concatenate_datasets(
            [
                rename_column(get_train_column("train_input"), "prompt"),
                rename_column(get_train_column("train_output"), "completion"),
            ],
            axis=1,
        )
        train_dataset = tokenize_column(
            DataSource(
                "Concatenate Train Input and Output",
                data=concatenate_prompts_and_completions(train_combined),
                total_num_rows=get_train_column("train_input").num_rows,
                auto_progress=get_train_column("train_input").num_rows is not None,
                save=not isinstance(train_combined, IterableDataset),
            ).output["text"],
            "input_ids",
            "Train Dataset",
            causal=True,
        )
        validation_combined = concatenate_datasets(
            [
                rename_column(get_validation_column("validation_input"), "prompt"),
                rename_column(get_validation_column("validation_output"), "completion"),
            ],
            axis=1,
        )
        validation_dataset = tokenize_column(
            DataSource(
                "Concatenate Validation Input and Output",
                data=concatenate_prompts_and_completions(validation_combined),
                total_num_rows=get_validation_column("validation_input").num_rows,
                auto_progress=get_validation_column("validation_input").num_rows
                is not None,
                save=not isinstance(validation_combined, IterableDataset),
            ).output["text"],
            "input_ids",
            "Validation Dataset",
            causal=True,
        )
    else:
        train_dataset = concatenate_datasets(
            [
                tokenize_column(train_column, train_column_name, name)
                if train_column_name in ["input_ids", "decoder_labels"]
                or train_column_name.endswith("_input_ids")
                else process_column(train_column, train_column_name, name)
                for (train_column_name, name), train_column in train_columns.items()
            ],
            axis=1,
        )
        validation_dataset = concatenate_datasets(
            [
                tokenize_column(validation_column, validation_column_name, name)
                if validation_column_name in ["input_ids", "decoder_labels"]
                or validation_column_name.endswith("_input_ids")
                else process_column(validation_column, validation_column_name, name)
                for (
                    validation_column_name,
                    name,
                ), validation_column in validation_columns.items()
            ],
            axis=1,
        )

    # Save information for publishing
    train_step = list(train_columns.values())[0].step
    self._step_metadata = train_step._get_metadata(train_step.output)

    # Save information for publishing
    self._examples = {
        name: (
            train_column.dataset[:3][train_column.column_names[0]]
            if isinstance(train_column.dataset, Dataset)
            else list(
                map(
                    lambda row: row[train_column.column_names[0]],
                    train_column.dataset.take(3),
                )
            )
        )
        for (_, name), train_column in train_columns.items()
    }
    if reward_scores:
        if self.chat_prompt_template:
            self._examples["Train Prompts"] = [
                apply_chat_prompt_template(prompt)
                for prompt in self._examples["Train Prompts"]
            ]
        self._examples["Train Input"] = [
            prompt + generation
            for prompt, generation in zip(
                self._examples["Train Prompts"], self._examples["Train Generations"]
            )
        ]
    elif reward_pairs:
        if self.chat_prompt_template:
            self._examples["Train Prompts"] = [
                apply_chat_prompt_template(prompt)
                for prompt in self._examples["Train Prompts"]
            ]
        chosen_examples = [
            prompt + generation
            for prompt, generation in zip(
                self._examples["Train Prompts"],
                self._examples["Train Chosen Generations"],
            )
        ]
        rejected_examples = [
            prompt + generation
            for prompt, generation in zip(
                self._examples["Train Prompts"],
                self._examples["Train Rejected Generations"],
            )
        ]
        self._examples["Train Input"] = list(
            chain.from_iterable(zip(chosen_examples, rejected_examples))
        )
    elif dpo:
        self._examples["Train Input"] = self._examples["Train Prompts"]

    # Return datasets
    return (
        get_sized_dataset(
            dataset=train_dataset,
            total_num_rows=list(train_columns.values())[0].num_rows,
        ),
        get_sized_dataset(
            dataset=validation_dataset,
            total_num_rows=list(validation_columns.values())[0].num_rows,
        ),
        label2id,
        is_multi_target,
    )


def _start_hf_trainer(self: "_TrainHFBase", trainer: Any):  # noqa: C901
    # Setup loggers the way we need them to be
    if not DataDreamer.ctx.hf_log:
        if self.logger.level <= logging.NOTSET:  # pragma: no cover
            hf_transformers_trainer_logger = logging.getLogger("transformers.trainer")
            if (
                not hf_transformers_trainer_logger.level
                or hf_transformers_trainer_logger.level > logging.INFO
            ):
                hf_transformers_trainer_logger.level = logging.INFO
                hf_transformers_trainer_logger.propagate = True
            DataDreamer._enable_hf_transformers_logging(progress_bars=False)
            DataDreamer._enable_setfit_logging(progress_bars=False)
            hf_transformers_logging.set_verbosity_info()
            setfit_logging.set_verbosity_info()

    # Add GPU monitoring if distributed
    device_memory_monitoring_callback = get_device_memory_monitoring_callback(
        trainer=self
    )
    trainer.add_callback(device_memory_monitoring_callback)

    # Run training
    try:
        # Try to resume
        if self.resumable:
            trainer.train(resume_from_checkpoint=True)
        else:
            raise ValueError()
    except ValueError:
        # Nothing to resume from, so start a new training run

        # Evaluate before starting training so we can see how the model
        # performs before any weight updates
        if device_memory_monitoring_callback:
            device_memory_monitoring_callback()._log_device_memory_usage()
        if is_distributed() and trainer.is_fsdp_enabled:  # pragma: no cover
            from transformers.trainer import logger as trainer_logger

            # This is a hack to run .evaluate() before training happens on FSDP
            # but after the FSDP is set up
            old_info = trainer_logger.info

            def _info(old_info, *args, **kwargs):
                if len(args) > 0 and args[0].startswith("***** Running training *****"):
                    trainer.evaluate()
                    trainer.model.train()  # Switch the model back to train mode
                    trainer_logger.info = old_info  # Undo the monkey-patch
                return old_info(*args, **kwargs)

            trainer_logger.info = partial(_info, old_info)
        else:
            trainer.evaluate()

        # Start training
        trainer.train()
    if not DataDreamer.ctx.hf_log:
        if self.logger.level <= logging.NOTSET:  # pragma: no cover
            logging.getLogger(
                "transformers.trainer"
            ).level = DataDreamer.ctx._transformers_trainer_verbosity
            DataDreamer._disable_hf_transformers_logging()
            DataDreamer._disable_setfit_logging()


class CustomDataCollatorWithPadding:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        fields_to_pad: list[dict[str, Any]],
        fields_to_keep: None | list[str] = None,
        extra_column_names_to_add: None | dict[str, Any] = None,
    ):
        self.tokenizer = tokenizer
        self.fields_to_pad = fields_to_pad
        self.fields_to_keep = fields_to_keep
        self.extra_column_names_to_add = extra_column_names_to_add

    def update_pad_token_id(
        self, tensor: torch.Tensor, pad_token_id: int, keep_first_pad_token: bool
    ):
        # Find where the pad tokens are
        pad_token_mask = tensor == self.tokenizer.pad_token_id
        if keep_first_pad_token:
            # Find the indices of the left-most pad token in each row
            leftmost_true_indices = pad_token_mask.to(torch.int32).argmax(dim=1)
            # Create a mask to help keep the left-most pad_token value
            keep_leftmost_mask = (
                torch.arange(pad_token_mask.size(1)) <= leftmost_true_indices[:, None]
            )
            # Apply the mask to the original mask
            pad_token_mask = pad_token_mask & ~keep_leftmost_mask
        # Update the pad token IDs
        tensor[pad_token_mask] = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        result = {}
        for field in self.fields_to_pad:
            tokenizer = field.get("tokenizer", self.tokenizer)
            pad_results = tokenizer.pad(
                [{"input_ids": feature[field["name"]]} for feature in features],
                padding=True,
                return_tensors="pt",
            )
            result[field["output_name"]] = pad_results["input_ids"]
            if "pad_token_id" in field:
                self.update_pad_token_id(
                    tensor=result[field["output_name"]],
                    pad_token_id=field["pad_token_id"],
                    keep_first_pad_token=field.get("keep_first_pad_token", False),
                )
            if "output_attention_mask_name" in field:  # pragma: no cover
                result[field["output_attention_mask_name"]] = pad_results[
                    "attention_mask"
                ]
            if isinstance(self.extra_column_names_to_add, dict):
                for (
                    column_name,
                    default_value,
                ) in self.extra_column_names_to_add.items():
                    result[column_name] = default_value
        if self.fields_to_keep is not None:
            for field_name in self.fields_to_keep:
                result[field_name] = [
                    feature[field_name] for feature in features if field_name in feature
                ]
                if len(result[field_name]) > 0 and isinstance(
                    result[field_name][0], (bool, int, float, np.ndarray, torch.Tensor)
                ):
                    result[field_name] = torch.tensor(result[field_name])
                elif len(result[field_name]) == 0:
                    del result[field_name]
        return result


def get_logging_callback(trainer: "_TrainHFBase", log_loss: bool = True) -> Type:
    class LoggingCallback(TrainerCallback):
        def on_log(self_, args, state, control, logs=None, **kwargs):
            if is_distributed() and get_global_rank() != 0:  # pragma: no cover
                return
            logs = logs.copy()
            if "eval_progress" in logs and logs["eval_progress"] == "100%":
                return
            _ = logs.pop("total_flos", None)
            _ = logs.pop("eval_joint_metric", None)
            if state.is_local_process_zero:
                epoch = logs.pop("epoch", 0.0)
                if any([metric.startswith("eval_") for metric in logs.keys()]):
                    logs = {k.replace("eval_", ""): v for k, v in logs.items()}
                    if not log_loss:
                        logs.pop("loss")
                    trainer.logger.info(f"Eval Epoch: {epoch} -- {logs}")
                else:
                    logs = {k.replace("train_", ""): v for k, v in logs.items()}
                    if not log_loss:
                        logs.pop("loss")
                    trainer.logger.info(f"Train Epoch: {epoch} -- {logs}")

    return LoggingCallback


class _TrainHFBase(DataDreamerTrainer):
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
        self.model_name = model_name
        super().__init__(name=name, force=force, verbose=verbose, log_level=log_level)
        self.chat_prompt_template, self.system_prompt = get_model_prompt_template(
            model_name=self.model_name,
            revision=revision,
            chat_prompt_template=chat_prompt_template,
            system_prompt=system_prompt,
        )
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.device = validate_device(device=device)
        self.dtype = convert_dtype(dtype)
        self.quantization_config = quantization_config
        self.peft_config = copy(peft_config)
        self.distributed_config = validate_distributed_config(distributed_config)
        self.fsdp = fsdp
        self.fsdp_config = fsdp_config
        self.kwargs = kwargs

        # Load config
        self.config = get_config(
            model_name=self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
        )

        # Initialize variables assigned later
        self.seed: None | int = None

        # Initalize variables used for publishing
        self._examples: dict[str, list[str]] = {}

    @cached_property
    def _is_encoder_decoder(self) -> bool:
        return is_encoder_decoder(self.config)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = get_tokenizer(
            model_name=self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            **self.kwargs,
        )
        if self.chat_prompt_template:
            set_hf_chat_template(
                tokenizer=tokenizer,
                chat_prompt_template=self.chat_prompt_template,
                system_prompt=self.system_prompt,
            )
        return tokenizer

    @property
    def auto_cls(self) -> Type:
        if self._is_encoder_decoder:
            auto_cls = AutoModelForSeq2SeqLM
        else:
            auto_cls = AutoModelForCausalLM
        return auto_cls

    def _create_model(
        self,
        label2id: None | dict[Any, int] = None,
        id2label: None | dict[int, Any] = None,
        is_multi_target: bool = False,
        device: None
        | int
        | str
        | torch.device
        | list[int | str | torch.device]
        | Default = DEFAULT,
        is_ref_model: bool = False,
    ) -> PreTrainedModel:
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
        classification_kwargs = {}
        if label2id is not None:
            classification_kwargs = {
                "num_labels": len(label2id),
                "label2id": label2id,
                "id2label": id2label,
                "problem_type": "multi_label_classification"
                if is_multi_target
                else "single_label_classification",
            }
        model = self.auto_cls.from_pretrained(
            self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            quantization_config=self.quantization_config,
            torch_dtype=self.dtype,
            **self.kwargs,
            **classification_kwargs,
        )

        # Optionally add tags if the user has the appropriate transformers
        # version. That way the tag will be pushed automatically even if the
        # users do not call `trainer.push_to_hub()` but e.g. `model.push_to_hub()`
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(self._trainer_tags)

        from .train_hf_classifier import TrainHFClassifier
        from .train_setfit_classifier import TrainSetFitClassifier

        if isinstance(self, TrainHFClassifier) and not isinstance(
            self, TrainSetFitClassifier
        ):
            model.config.eos_token_id = self.tokenizer.eos_token_id
            model.config.pad_token_id = self.tokenizer.pad_token_id

        # Send model to accelerator device
        model_device = default_to(device, self.device)
        model = model.to("cpu" if isinstance(model_device, list) else model_device)

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
        resource.save_pretrained(path, safe_serialization=True)

    def _publish_resource(
        self,
        resource: Any,
        repo_id: str,
        branch: None | str = None,
        private: bool = False,
        **kwargs,
    ):  # pragma: no cover
        resource.push_to_hub(
            repo_id=repo_id,
            branch=branch,
            private=private,
            safe_serialization=True,
            **kwargs,
        )

    def _save_model(
        self,
        training_args: None | TrainingArguments,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        accelerator: Any = None,
        fsdp: bool = False,
    ):
        MODEL_DIR = os.path.join(self._output_folder_path, "_model")

        # Save the model
        if is_distributed():  # pragma: no cover
            save_distributed_model(
                trainer=self,
                accelerator=accelerator,
                fsdp=fsdp,
                peft=self.peft_config is not None,
                model=model,
                output_dir=MODEL_DIR,
            )
        else:
            # Save a normal model
            self._save_resource(model, MODEL_DIR)

        # Save other metadata
        if not_distributed_or_main_process():
            self._save_resource(tokenizer, MODEL_DIR)
            with open(os.path.join(MODEL_DIR, "step_metadata.json"), "w+") as f:
                json.dump(self._step_metadata, f, indent=4)
            with open(os.path.join(MODEL_DIR, "widget_examples.json"), "w+") as f:
                json.dump(self._examples, f, indent=4)
            with open(os.path.join(MODEL_DIR, "seed.json"), "w+") as f:
                json.dump(self.seed, f, indent=4)
            if training_args:
                training_args_dict = training_args.to_dict()
            else:  # pragma: no cover
                training_args_dict = {}
            with open(os.path.join(MODEL_DIR, "training_args.json"), "w+") as f:
                json.dump(training_args_dict, f, indent=4)

    def _load_model_metadata(self):
        MODEL_DIR = os.path.join(self._output_folder_path, "_model")

        # Seed
        with open(os.path.join(MODEL_DIR, "step_metadata.json"), "r") as f:
            self._step_metadata = json.load(f)
        with open(os.path.join(MODEL_DIR, "widget_examples.json"), "r") as f:
            self._examples = json.load(f)
        with open(os.path.join(MODEL_DIR, "seed.json"), "r") as f:
            self.seed = json.load(f)
        if self.seed:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():  # pragma: no cover
                torch.cuda.manual_seed_all(self.seed)

    def _load_model(
        self,
        label2id: None | dict[Any, int] = None,
        id2label: None | dict[int, Any] = None,
        is_multi_target: bool = False,
        with_optimizations: bool = True,
    ) -> PreTrainedModel:
        MODEL_DIR = os.path.join(self._output_folder_path, "_model")

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

            classification_kwargs = {}
            if label2id is not None:
                classification_kwargs = {
                    "num_labels": len(label2id),
                    "label2id": label2id,
                    "id2label": id2label,
                    "problem_type": "multi_label_classification"
                    if is_multi_target
                    else "single_label_classification",
                }
            model = self.auto_cls.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                quantization_config=self.quantization_config,
                torch_dtype=self.dtype,
                **self.kwargs,
                **classification_kwargs,
            )
            model = PeftModel.from_pretrained(
                model,
                model_id=MODEL_DIR,
                quantization_config=self.quantization_config,
                torch_dtype=self.dtype,
                **self.kwargs,
            )
        else:
            model = self.auto_cls.from_pretrained(
                MODEL_DIR,
                quantization_config=self.quantization_config,
                torch_dtype=self.dtype,
                **self.kwargs,
            )

        # Send model to accelerator device
        model = model.to("cpu" if isinstance(self.device, list) else self.device)

        # Switch model to eval mode
        model.eval()

        if with_optimizations:
            # Apply BetterTransformer
            if self.auto_cls in [AutoModelForSeq2SeqLM, AutoModelForCausalLM]:
                if BetterTransformerManager.cannot_support(
                    model.config.model_type
                ) or not BetterTransformerManager.supports(model.config.model_type):
                    model = model  # pragma: no cover
                else:
                    model = BetterTransformer.transform(model)

            # Torch compile
            #
            # Note: Disabling due to a bug in PyTorch where encoder-decoder (T5)
            # models get compiled over an over again, making it slow. If enabled in the
            # future, it would be better to use TrainingArguments()'s torch_compile
            # arg.
            #
            # torch._dynamo.config.suppress_errors = True
            # model = torch.compile(model)

        # Finished loading
        log_if_timeout.stop(
            partial(lambda self: self.logger.info("Finished loading."), self)
        )

        return model

    def _load(self, with_optimizations: bool = True):
        model = self._load_model(with_optimizations=with_optimizations)
        return model

    @property
    def model(self) -> PreTrainedModel:
        return super().model

    @property
    def model_path(self) -> str:  # type:ignore[return]
        """The path to the trained model after training."""
        if self.model:
            return os.path.join(self._output_folder_path, "_model")

    def export_to_disk(self, path: str, adapter_only: bool = False) -> PreTrainedModel:
        """Export the trained model to disk.

        Args:
            path: The path to export the model to.
            adapter_only: Whether to export only the adapter.

        Returns:
            The exported model.
        """
        from .train_hf_finetune import TrainHFFineTune
        from .train_setfit_classifier import TrainSetFitClassifier

        assert (
            not adapter_only or self.peft_config
        ), "`adapter_only` can only be used if a `peft_config` was provided."

        # Clear the directory
        clear_dir(path)

        # Create a copy of the model & merge adapter if requested
        self.unload_model()
        model = self._load(with_optimizations=False)
        if not isinstance(self, TrainSetFitClassifier):
            if self.peft_config and not adapter_only:
                model = model.merge_and_unload()
                if hasattr(self, "label2id") and hasattr(self, "id2label"):
                    model.config.num_labels = len(self.label2id)
                    model.config.label2id = self.label2id
                    model.config.id2label = self.id2label
                if hasattr(self, "is_multi_target") and self.is_multi_target:
                    model.config.problem_type = (
                        "multi_label_classification"
                        if self.is_multi_target
                        else "single_label_classification"
                    )
        # Set generation configuration
        if isinstance(self, TrainHFFineTune) and (
            self.peft_config is None or not adapter_only
        ):
            model.generation_config._from_model_config = False
            model.generation_config.max_length = self.tokenizer.model_max_length

        # Save model and tokenizer
        self._save_resource(model, path)
        self._save_resource(self.tokenizer, path)

        # Copy labels and IDs
        if os.path.exists(
            os.path.join(self._output_folder_path, "_model", "label2id.json")
        ):
            copy2(
                os.path.join(self._output_folder_path, "_model", "label2id.json"),
                os.path.join(path, "label2id.json"),
            )
        if os.path.exists(
            os.path.join(self._output_folder_path, "_model", "id2label.json")
        ):
            copy2(
                os.path.join(self._output_folder_path, "_model", "id2label.json"),
                os.path.join(path, "id2label.json"),
            )

        # Copy training args
        if os.path.exists(
            os.path.join(self._output_folder_path, "_model", "training_args.json")
        ):
            copy2(
                os.path.join(self._output_folder_path, "_model", "training_args.json"),
                os.path.join(path, "training_args.json"),
            )

        # Get README contents
        publish_info = self._publish_info(
            repo_id=path, branch=None, adapter_only=adapter_only
        )

        # Save README.md
        readme_contents = """# Model Card

{body}

---
This model was trained with [DataDreamer 🤖💤](https://datadreamer.dev)."""
        readme_contents = readme_contents.replace("{body}", publish_info["body"])

        with open(os.path.join(path, "README.md"), "w+") as f:
            f.write(readme_contents)

        # Clear memory
        self.unload_model()

        # Return model
        logger.info(f"Trainer '{self.name}' exported to disk 💫 : {path}")
        return model

    def _publish_info(
        self, repo_id: str, branch: None | str = None, adapter_only: bool = False
    ) -> dict[str, Any]:  # pragma: no cover
        def apply_chat_prompt_template(prompt: str) -> str:
            return (
                cast(str, self.chat_prompt_template)
                .replace("{{system_prompt}}", self.system_prompt or "")
                .replace("{{prompt}}", prompt)
            )

        widget_examples = [
            f"text: {json.dumps(str(apply_chat_prompt_template(example) if self.chat_prompt_template else example))}\n"
            f'    example_title: "Example {str(example_idx + 1)}"'
            for example_idx, example in enumerate(self._examples.get("Train Input", []))
        ]
        return {
            "body": None,
            "tags": [],
            "pipeline_tag": None,
            "library_name": "peft" if self.peft_config and adapter_only else None,
            "widget_examples": widget_examples,
        }

    def publish_to_hf_hub(  # noqa: C901
        self,
        repo_id: str,
        branch: None | str = None,
        private: bool = False,
        token: None | str = None,
        adapter_only: bool = False,
        is_synthetic: bool = True,
        **kwargs,
    ) -> str:  # pragma: no cover
        """Publish the model to the Hugging Face Hub.

        Args:
            repo_id: The repository ID to publish the model to.
            branch: The branch to push the model to.
            private: Whether to make the model private.
            token: The Hugging Face API token to use for authentication.
            adapter_only: Whether to publish only the adapter.
            is_synthetic: Whether the dataset is synthetic (applies certain metadata
                when publishing).
            **kwargs: Additional keyword arguments to pass to
                :py:meth:`~transformers.PreTrainedModel.push_to_hub`.

        Returns:
            The URL to the published model.
        """
        from .train_hf_finetune import TrainHFFineTune
        from .train_setfit_classifier import TrainSetFitClassifier

        assert (
            not adapter_only or self.peft_config
        ), "`adapter_only` can only be used if a `peft_config` was provided."

        # Login
        api = hf_hub_login(token=token)
        if "/" not in repo_id:
            repo_id = f"{api.whoami()['name']}/{repo_id}"

        # Create a copy of the model & merge adapter if requested
        self.unload_model()
        model = self._load(with_optimizations=False)
        if not isinstance(self, TrainSetFitClassifier):
            if self.peft_config and not adapter_only:
                model = model.merge_and_unload()
                if hasattr(self, "label2id") and hasattr(self, "id2label"):
                    model.config.num_labels = len(self.label2id)
                    model.config.label2id = self.label2id
                    model.config.id2label = self.id2label
                if hasattr(self, "is_multi_target") and self.is_multi_target:
                    model.config.problem_type = (
                        "multi_label_classification"
                        if self.is_multi_target
                        else "single_label_classification"
                    )

        # Set generation configuration
        if isinstance(self, TrainHFFineTune) and (
            self.peft_config is None or not adapter_only
        ):
            model.generation_config._from_model_config = False
            model.generation_config.max_length = self.tokenizer.model_max_length

        # Prepare for publishing
        (tags, dataset_names, model_names, upload_metadata) = prepare_to_publish(
            step_metadata=self._step_metadata,
            api=api,
            repo_id=repo_id,
            repo_type="model",
            branch=branch,
            is_synthetic=is_synthetic,
        )
        publish_info = self._publish_info(
            repo_id=repo_id, branch=branch, adapter_only=adapter_only
        )

        # Push model and tokenizer
        DataDreamer._enable_hf_transformers_logging()
        DataDreamer._enable_hf_huggingface_hub_logging(logs=True)
        self._publish_resource(
            model, repo_id=repo_id, branch=branch, private=private, **kwargs
        )
        self._publish_resource(
            self.tokenizer, repo_id=repo_id, branch=branch, private=private, **kwargs
        )
        DataDreamer._disable_hf_huggingface_hub_logging()
        DataDreamer._disable_hf_transformers_logging()

        # Upload labels and IDs
        if os.path.exists(
            os.path.join(self._output_folder_path, "_model", "label2id.json")
        ):
            api.upload_file(
                path_or_fileobj=os.path.join(
                    self._output_folder_path, "_model", "label2id.json"
                ),
                path_in_repo="label2id.json",
                repo_id=repo_id,
                repo_type="model",
                revision=branch,
                commit_message="Pushed by DataDreamer",
                commit_description="Update label2id.json",
            )
        if os.path.exists(
            os.path.join(self._output_folder_path, "_model", "id2label.json")
        ):
            api.upload_file(
                path_or_fileobj=os.path.join(
                    self._output_folder_path, "_model", "id2label.json"
                ),
                path_in_repo="id2label.json",
                repo_id=repo_id,
                repo_type="model",
                revision=branch,
                commit_message="Pushed by DataDreamer",
                commit_description="Update id2label.json",
            )

        # Clear memory
        del model
        self.unload_model()

        # Push datadreamer.json
        upload_metadata(trainer=self)

        # Upload training_args.json
        api.upload_file(
            path_or_fileobj=os.path.join(
                self._output_folder_path, "_model", "training_args.json"
            ),
            path_in_repo="training_args.json",
            repo_id=repo_id,
            repo_type="model",
            revision=branch,
            commit_message="Pushed by DataDreamer",
            commit_description="Update training_args.json",
        )

        # Upload README.md
        readme_contents = (
            """
---
base_model: {base_model}
{datasets}
tags:
{tags}
{library_name}
{widget}
{pipeline_tag}
---
# Model Card

[Add more information here](https://huggingface.co/templates/model-card-example)

{body}

---
This model was trained with"""
            f""" a {'synthetic ' if is_synthetic else ''}dataset with"""
            f""" [DataDreamer 🤖💤](https://datadreamer.dev)."""
            f""" The {'synthetic ' if is_synthetic else ''}dataset card and model"""
            f""" card can be found [here](datadreamer.json)."""
            f""" The training arguments can be found [here](training_args.json)."""
        )
        readme_contents = readme_contents.replace("{base_model}", self.model_name)
        tags = tags + model_names
        tags = (
            tags
            + publish_info["tags"]
            + (
                [publish_info["pipeline_tag"]]
                if publish_info["pipeline_tag"] is not None
                else []
            )
        )
        readme_contents = readme_contents.replace("{tags}", "- " + ("\n- ".join(tags)))
        if len(dataset_names) > 0:
            readme_contents = readme_contents.replace(
                "{datasets}", "datasets:\n- " + ("\n- ".join(dataset_names))
            )
        else:
            readme_contents = readme_contents.replace("{datasets}", "")
        if publish_info["library_name"] is not None:
            readme_contents = readme_contents.replace(
                "{library_name}", f"library_name: {publish_info['library_name']}"
            )
        else:
            readme_contents = readme_contents.replace("{library_name}", "")
        if len(publish_info["widget_examples"]) > 0:
            readme_contents = readme_contents.replace(
                "{widget}",
                "widget:\n  - " + ("\n  - ".join(publish_info["widget_examples"])),
            )
        else:
            readme_contents = readme_contents.replace("{widget}", "")
        if publish_info["pipeline_tag"] is not None:
            readme_contents = readme_contents.replace(
                "{pipeline_tag}", f"pipeline_tag: {publish_info['pipeline_tag']}"
            )
        else:
            readme_contents = readme_contents.replace("{pipeline_tag}", "")
        if publish_info["body"] is not None:
            readme_contents = readme_contents.replace("{body}", publish_info["body"])
        else:
            readme_contents = readme_contents.replace("{body}", "")
        current_readme_contents = get_readme_contents(
            repo_id, repo_type="model", revision=branch
        )
        if "DataDreamer" not in current_readme_contents:
            api.upload_file(
                path_or_fileobj=BytesIO(bytes(readme_contents, "utf8")),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                revision=branch,
                commit_message="Pushed by DataDreamer",
                commit_description="Update README.md",
            )

        # Construct and return URL
        url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Trainer '{self.name}' published to HF Hub 💫 : {url}")
        return url

    @cached_property
    def base_model_card(self) -> None | str:
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
        if self.peft_config:
            citations.append(PEFT_CITATION)
        if isinstance(model_citations, list):
            citations.extend(model_citations)
        return citations

    @cached_property
    def display_name(self) -> str:
        return f"{self.name} ({self.model_name})"

    def compute_fingerprint(self, **kwargs) -> str:
        def filter_kwargs(arg_name: str) -> bool:
            return arg_name not in ["precompute_ref_log_probs"]

        def map_kwarg(arg_name: str, value: Any) -> Any:
            if arg_name == "batch_size" and isinstance(
                self.device, list
            ):  # pragma: no cover
                return (
                    value
                    * len(self.device)
                    * get_num_nodes_from_distributed_config(self.distributed_config)
                )
            else:
                return value

        column_fingerprints = {}
        for kwarg in sorted(kwargs.keys()):
            if isinstance(
                kwargs[kwarg], OutputDatasetColumn | OutputIterableDatasetColumn
            ):
                column = kwargs.pop(kwarg)
                column_fingerprints[kwarg] = (
                    column.step.fingerprint,
                    column.column_names,
                )

        to_hash = [
            str(type(self).__name__),
            self.name,
            self.version,
            self.model_name,
            self.chat_prompt_template,
            self.system_prompt,
            self.revision,
            self.dtype,
            False,  # Legacy fingerprint, prev was: load_in_4bit
            False,  # Legacy fingerprint, prev was: load_in_8bit
            self.quantization_config,
            self.peft_config,
            column_fingerprints,
            stable_fingerprint(
                {
                    kwarg: map_kwarg(kwarg, val)
                    for kwarg, val in kwargs.items()
                    if filter_kwargs(kwarg)
                }
            ),
        ]
        if isinstance(self.device, list):  # pragma: no cover
            to_hash.extend([self.fsdp, self.fsdp_config])
        fingerprint = Hasher.hash(to_hash)
        self.fingerprint = fingerprint
        return fingerprint

    def unload_model(self):
        super().unload_model()

        # Clear CUDA cache
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.empty_cache()


__all__ = ["_TrainHFBase"]
