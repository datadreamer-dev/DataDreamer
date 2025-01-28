import logging
import os
import sys
from contextlib import nullcontext
from functools import cache, partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Type, cast
from unittest.mock import patch

import dill
import numpy as np
import torch
from datasets import Dataset, IterableDataset, Value, concatenate_datasets
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .. import DataDreamer
from ..datasets import (
    OutputDatasetColumn,
    OutputIterableDataset,
    OutputIterableDatasetColumn,
)
from ..datasets.datasets import _SizedIterableDataset, get_sized_dataset
from ..steps import DataSource
from ..trainers.trainer import JointMetric
from ..utils.device_utils import (
    _TrainingArgumentDeviceOverrideMixin,
    get_device_memory_monitoring_callback,
)
from ..utils.distributed_utils import (
    get_current_accelerator,
    get_global_rank,
    get_local_world_size,
    is_distributed,
    not_distributed_or_main_process,
    set_current_accelerator,
)
from ..utils.import_utils import ignore_transformers_warnings

with ignore_transformers_warnings():
    from setfit import logging as setfit_logging
    from transformers import (
        PreTrainedTokenizer,
        Seq2SeqTrainingArguments as _Seq2SeqTrainingArguments,
        TrainerCallback,
        TrainerState,
        TrainingArguments as _TrainingArguments,
        logging as hf_transformers_logging,
    )
    from transformers.trainer_pt_utils import EvalLoopContainer

if TYPE_CHECKING:  # pragma: no cover
    from ..trainers.train_hf_classifier import _TrainHFBase

    with ignore_transformers_warnings():
        from transformers import Trainer


_old_TrainerState__post_init__ = TrainerState.__post_init__


def _deserialize_join_metric__TrainerState__post_init__(self, *args, **kwargs):
    _old_TrainerState__post_init__(self, *args, **kwargs)
    if (
        hasattr(self, "best_metric")
        and isinstance(self.best_metric, dict)
        and "is_joint_metric" in self.best_metric
    ):
        self.best_metric = JointMetric(**self.best_metric)


@cache
def _monkey_patch_TrainerState__post_init__():
    TrainerState.__post_init__ = _deserialize_join_metric__TrainerState__post_init__


class TrainingArguments(_TrainingArgumentDeviceOverrideMixin, _TrainingArguments):
    pass


class Seq2SeqTrainingArguments(
    _TrainingArgumentDeviceOverrideMixin, _Seq2SeqTrainingArguments
):
    pass


def wrap_trainer_cls(
    trainer_cls: Type["Trainer"],
    optimizers: tuple[None | Optimizer, None | LambdaLR] = (None, None),
    optimizer: None | Optimizer = None,
    lr_scheduler: None | LambdaLR = None,
    compute_loss: None | Callable = None,
    trainer: "None | _TrainHFBase" = None,
) -> Type["Trainer"]:
    class WrappedTrainer(trainer_cls):
        def __init__(self, *args, **kwargs):
            cls_names = [b.__name__ for b in WrappedTrainer.__bases__]
            if "tokenizer" in kwargs and "RewardTrainer" not in cls_names:
                kwargs["processing_class"] = kwargs["tokenizer"]
                del kwargs["tokenizer"]
            super().__init__(*args, **kwargs)
            set_current_accelerator(self.accelerator)

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

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            if compute_loss is not None:  # pragma: no cover
                return compute_loss(model, inputs, return_outputs=return_outputs)
            else:
                kwargs = {"num_items_in_batch": num_items_in_batch}
                if num_items_in_batch is None:
                    del kwargs["num_items_in_batch"]
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs, **kwargs
                )

        def visualize_samples(self, *args, **kwargs):
            if (
                not_distributed_or_main_process()
                and trainer is not None
                and trainer.logger.level <= logging.DEBUG
            ):  # pragma: no cover
                return super().visualize_samples(*args, **kwargs)

    return WrappedTrainer


def prepare_inputs_and_outputs(  # noqa: C901
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
            from ..trainers._vendored._dpo_helper import DPODataCollatorWithPadding

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


def start_hf_trainer(self: "_TrainHFBase", trainer: Any):  # noqa: C901
    patches = nullcontext()
    if is_distributed():  # pragma: no cover
        patches = patch(  # type:ignore[assignment]
            "transformers.trainer.unwrap_model", lambda model, *args, **kwargs: model
        )

    with patches:
        # Do monkey patches
        _monkey_patch_EvalLoopContainer_add()

        # Setup loggers the way we need them to be
        if not DataDreamer.ctx.hf_log:
            if self.logger.level <= logging.NOTSET:  # pragma: no cover
                hf_transformers_trainer_logger = logging.getLogger(
                    "transformers.trainer"
                )
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
            try:
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
                        if len(args) > 0 and args[0].startswith(
                            "***** Running training *****"
                        ):
                            trainer.evaluate()
                            trainer.model.train()  # Switch the model back to train mode
                            trainer_logger.info = old_info  # Undo the monkey-patch
                        return old_info(*args, **kwargs)

                    trainer_logger.info = partial(_info, old_info)
                else:
                    trainer.evaluate()

                # Start training
                trainer.train()
            except Exception as e:
                raise e from None
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


def wrap_compute_metrics(compute_metrics, training_args: "TrainingArguments"):
    def _wrapped_compute_metrics(*args, compute_result: None | bool = None, **kwargs):
        if not_distributed_or_main_process():
            if compute_result is not None:
                computed_metrics = compute_metrics(
                    *args, compute_result=compute_result, **kwargs
                )
            else:  # pragma: no cover
                computed_metrics = compute_metrics(*args, **kwargs)
            if is_distributed():  # pragma: no cover
                for _ in range(get_local_world_size() - 1):
                    DataDreamer.ctx.distributed_pipe.put(dill.dumps(computed_metrics))
                get_current_accelerator().wait_for_everyone()
            return computed_metrics
        else:  # pragma: no cover
            get_current_accelerator().wait_for_everyone()
            return dill.loads(DataDreamer.ctx.distributed_pipe.get())

    return _wrapped_compute_metrics if compute_metrics is not None else None


_old_EvalLoopContainer_add = EvalLoopContainer.add


def _save_memory_in__EvalLoopContainer_add(self, *args, **kwargs):
    if DataDreamer.initialized() and not_distributed_or_main_process():
        _old_EvalLoopContainer_add(self, *args, **kwargs)
    elif not DataDreamer.initialized():  # pragma: no cover
        _old_EvalLoopContainer_add(self, *args, **kwargs)
    else:  # pragma: no cover
        # Don't save when distributed and not the main process to save memory
        self.tensors = torch.tensor([0.0])  # Dummy list


@cache
def _monkey_patch_EvalLoopContainer_add():
    EvalLoopContainer.add = _save_memory_in__EvalLoopContainer_add


class ComputeMetricsState:
    def __init__(self):
        self.metrics = []

    def add_metrics(self, batch_size, metrics_dict, compute_result: None | bool = None):
        if compute_result is None:  # pragma: no cover
            return metrics_dict
        elif compute_result is False:
            self.metrics.append({"weight": batch_size, "metrics": metrics_dict})
            return metrics_dict
        elif compute_result is True:
            self.metrics.append({"weight": batch_size, "metrics": metrics_dict})

            # Compute total weight
            total_weight = sum([m["weight"] for m in self.metrics])

            # Initialize a dictionary to store the weighted sums of metrics
            weighted_sums = {}

            # Accumulate the weighted sum for each metric
            for entry in self.metrics:
                weight = entry["weight"]
                metrics = entry["metrics"]
                for key, value in metrics.items():
                    if not (
                        isinstance(value, int)
                        or isinstance(value, float)
                        or isinstance(value, JointMetric)
                        or isinstance(value, torch.Tensor)
                        or isinstance(value, np.ndarray)
                        or isinstance(value, np.floating)
                        or isinstance(value, np.integer)
                    ):  # pragma: no cover
                        value = 0
                    if key not in weighted_sums:
                        weighted_sums[key] = value * weight
                    else:
                        weighted_sums[key] += value * weight

            # Compute the weighted average for each metric
            averaged_metrics = {
                key: weighted_sums[key] / total_weight for key in weighted_sums
            }

            # Reset the metrics state
            self.metrics.clear()

            return averaged_metrics
