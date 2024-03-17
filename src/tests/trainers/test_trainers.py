import json
import os
import sys
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import jsonlines
import pytest
import torch
from jsonlines import Reader
from openai import BadRequestError, NotFoundError, PermissionDeniedError

from ... import DataDreamer
from ...llms import OpenAI
from ...llms._chat_prompt_templates import CHAT_PROMPT_TEMPLATES, SYSTEM_PROMPTS
from ...steps import DataSource
from ...trainers import (
    TrainHFClassifier,
    TrainHFDPO,
    TrainHFFineTune,
    TrainHFPPO,
    TrainHFRewardModel,
    TrainOpenAIFineTune,
    TrainSentenceTransformer,
    TrainSetFitClassifier,
)
from ...trainers._train_hf_base import CustomDataCollatorWithPadding
from ...utils.fs_utils import clear_dir
from ...utils.hf_model_utils import get_orig_model, validate_peft_config
from ...utils.import_utils import ignore_transformers_warnings

with ignore_transformers_warnings():
    from transformers import (
        DataCollatorForSeq2Seq,
        DataCollatorWithPadding,
        TrainerCallback,
        pipeline,
    )


class TestTrainer:
    def test_trainer_not_in_datadreamer(self, create_datadreamer):
        with pytest.raises(RuntimeError):
            TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")

    def test_empty_name(self, create_datadreamer):
        with create_datadreamer():
            with pytest.raises(ValueError):
                TrainHFClassifier("", model_name="google/flan-t5-small")

    @pytest.mark.parametrize("verbose,num_logs", [(True, 22), (False, 17)])
    def test_logging(self, verbose, num_logs, create_datadreamer, caplog):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={"inputs": ["in", "in"], "outputs": ["out1", "out2"]},
            )
            trainer = TrainHFClassifier(
                "T5 Trainer", model_name="google/flan-t5-small", verbose=verbose
            )
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=1,
                batch_size=1,
            )
            logs = [rec.message for rec in caplog.records]
            caplog.clear()
            assert len(logs) == num_logs

    def test_train_on_iterable_column(self, create_datadreamer):
        with create_datadreamer():

            def dataset_generator():
                yield {"inputs": "in", "outputs": "out1"}
                yield {"inputs": "in", "outputs": "out2"}

            dataset = DataSource(
                "Training Data", data=dataset_generator, total_num_rows=2
            )
            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=1,
                batch_size=1,
            )


class TestTrainHFBase:
    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none"
            )
            trainer = TrainHFClassifier(
                "T5 Trainer", model_name="google/flan-t5-small", peft_config=peft_config
            )
            assert trainer.base_model_card is not None
            assert trainer.license is not None
            assert isinstance(trainer.citation, list)
            assert len(trainer.citation) == 3

    @pytest.mark.skipif(
        "HUGGING_FACE_HUB_TOKEN" not in os.environ, reason="requires HF Hub token"
    )
    def test_hf_chat_template(self, create_datadreamer):
        with create_datadreamer():
            trainer = TrainHFFineTune(
                "Falcon Trainer", model_name="tiiuae/falcon-180B-chat"
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "A"},
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "C"},
                    ],
                    tokenize=False,
                )
                == "A\nUser: B\nAssistant: C<|endoftext|>"
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "C"},
                    ],
                    tokenize=False,
                )
                == "You are a helpful assistant.\nUser: B\nAssistant: C<|endoftext|>"
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "A"},
                        {"role": "user", "content": "B"},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                == "A\nUser: B\nAssistant: "
            )
        with create_datadreamer():
            trainer = TrainHFFineTune(
                "OpenBuddy Trainer",
                model_name="OpenBuddy/openbuddy-mistral-7b-v13",
                chat_prompt_template="{{system_prompt}}\n\nUser: {{prompt}}\nAssistant: ",
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "A"},
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "C"},
                    ],
                    tokenize=False,
                )
                == "<s>A\n\nUser: B\nAssistant: C</s>"
            )
        with create_datadreamer():
            trainer = TrainHFFineTune(
                "Zephyr Trainer", model_name="HuggingFaceH4/zephyr-7b-beta"
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": "A"},
                        {"role": "assistant", "content": "B"},
                    ],
                    tokenize=False,
                )
                == "<s><|system|>\nYou are a helpful assistant.</s>\n<|user|>\nA</s>\n<|assistant|>\nB</s>"
            )
        with create_datadreamer():
            trainer = TrainHFFineTune(
                "Zephyr (System) Trainer",
                model_name="HuggingFaceH4/zephyr-7b-beta",
                chat_prompt_template="<|system|>\n{{system_prompt}}</s>\n<|user|>\n{{prompt}}</s>\n<|assistant|>\n",
                system_prompt="You are a helpful assistant.",
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "A"},
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "C"},
                    ],
                    tokenize=False,
                )
                == "<s><|system|>\nA</s>\n<|user|>\nB</s>\n<|assistant|>\nC</s>"
            )
        with create_datadreamer():
            trainer = TrainHFFineTune(
                "LLaMa-2 (Falcon) Trainer",
                model_name="meta-llama/Llama-2-7b-chat-hf",
                chat_prompt_template=CHAT_PROMPT_TEMPLATES["falcon"],
                system_prompt=SYSTEM_PROMPTS["llama_system"],
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "A"},
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "C"},
                    ],
                    tokenize=False,
                )
                == "<s>A\nUser: B\nAssistant: C</s>"
            )
        with create_datadreamer():
            trainer = TrainHFFineTune(
                "LLaMa-2 (Custom) Trainer",
                model_name="meta-llama/Llama-2-7b-chat-hf",
                chat_prompt_template=(
                    "Custom\n\n{{system_prompt}}\nUser: {{prompt}}" "\nAssistant: "
                ),
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "A"},
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "C"},
                    ],
                    tokenize=False,
                )
                == "<s>Custom\n\nA\nUser: B\nAssistant: C</s>"
            )
        with create_datadreamer():
            trainer = TrainHFFineTune(
                "RedPajama Trainer",
                model_name="togethercomputer/RedPajama-INCITE-7B-Chat",
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": "A"},
                        {"role": "assistant", "content": "B"},
                    ],
                    tokenize=False,
                )
                == "<human>: A\n<bot>: B<|endoftext|>"
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "A"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                == "<human>: A\n<bot>: "
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "A"},
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "C"},
                    ],
                    tokenize=False,
                )
                == "<human>: B\n<bot>: C<|endoftext|>"
            )
        with create_datadreamer():
            trainer = TrainHFFineTune(
                "LLaMa-2 Trainer", model_name="meta-llama/Llama-2-7b-chat-hf"
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "A"},
                        {"role": "user", "content": "B"},
                        {"role": "assistant", "content": "C"},
                    ],
                    tokenize=False,
                )
                == "<s>[INST] <<SYS>>\nA\n<</SYS>>\n\nB [/INST] C</s>"
            )
            assert trainer.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": "B"},
                    {"role": "assistant", "content": "C"},
                ],
                tokenize=False,
            ) == (
                "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>"
                "\n\nB [/INST] C</s>"
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "A"},
                        {"role": "user", "content": "B"},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                == "<s>[INST] <<SYS>>\nA\n<</SYS>>\n\nB [/INST] "
            )
        with create_datadreamer():
            trainer = TrainHFFineTune(
                "CodeLLaMa Trainer", model_name="codellama/CodeLlama-7b-Instruct-hf"
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": "A"},
                        {"role": "assistant", "content": "B"},
                    ],
                    tokenize=False,
                )
                == "<s>[INST] A [/INST] B</s>"
            )
            assert (
                trainer.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "A"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                == "<s>[INST] A [/INST] "
            )
            with pytest.raises(Exception, match=r"Conversation roles must alternate.*"):
                assert (
                    trainer.tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": "A"},
                            {"role": "user", "content": "B"},
                            {"role": "assistant", "content": "C"},
                        ],
                        tokenize=False,
                    )
                    == "<s>[INST] B [/INST] C</s>"
                )

    def test_validate_peft_config(self, create_datadreamer):
        # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
        with ignore_transformers_warnings():
            from peft import LoraConfig

        with create_datadreamer():
            classification_trainer = TrainHFClassifier(
                "T5 Trainer", model_name="google/flan-t5-small"
            )
            assert set(
                validate_peft_config(
                    classification_trainer._create_model(), LoraConfig()
                ).target_modules
            ) == {"wi_1", "wi_0", "wo", "v", "out_proj", "q", "o", "k", "dense"}
            assert not validate_peft_config(
                classification_trainer._create_model(), LoraConfig()
            ).fan_in_fan_out

        with create_datadreamer():
            finetune_trainer = TrainHFFineTune("GPT-2 Trainer", model_name="gpt2")
            assert set(
                validate_peft_config(
                    finetune_trainer._create_model(), LoraConfig()
                ).target_modules
            ) == {"c_attn", "c_fc", "c_proj"}
            assert validate_peft_config(
                finetune_trainer._create_model(), LoraConfig()
            ).fan_in_fan_out

    def test_resume(self, create_datadreamer):
        exit_on_epoch: None | int = None
        data = {
            "inputs": ["Bad", "Good", "Bad", "Good", "Bad"],
            "outputs": ["negative", "positive", "negative", "positive", "negative"],
        }

        class ExitCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                if state.epoch == exit_on_epoch:
                    raise ReferenceError("CustomError")

        with create_datadreamer():
            dataset = DataSource("Training Data", data=data)
            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            exit_on_epoch = 1
            with pytest.raises(ReferenceError):
                trainer.train(
                    train_input=dataset.output["inputs"],
                    train_output=dataset.output["outputs"],
                    validation_input=dataset.output["inputs"],
                    validation_output=dataset.output["outputs"],
                    epochs=3,
                    batch_size=8,
                    callbacks=[ExitCallback],
                )
            assert not trainer._resumed
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert not os.path.isdir(os.path.join(trainer_path, "_model"))
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            dataset = DataSource("Training Data", data=data)
            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            exit_on_epoch = None
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=3,
                batch_size=8,
                callbacks=[ExitCallback],
            )
            assert trainer._resumed
            assert trainer.seed == 42
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(trainer_path, "_model"))

        with create_datadreamer(resume_path):
            dataset = DataSource("Training Data", data=data)
            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            exit_on_epoch = 1
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=3,
                batch_size=8,
                callbacks=[ExitCallback],
            )
            assert trainer._resumed
            assert trainer.seed == 42
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(trainer_path, "_model"))

        with create_datadreamer(resume_path):
            dataset = DataSource("Training Data", data=data)
            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            exit_on_epoch = None
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=3,
                batch_size=10,
                callbacks=[ExitCallback],
            )
            assert not trainer._resumed
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(trainer_path, "_model"))
            backup_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                "_backups",
                "t5-trainer",
                "af8fe7c4d25eacaa",
            )
            assert os.path.isdir(backup_path)
            assert os.path.isfile(os.path.join(backup_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(backup_path, "_model"))

        with create_datadreamer(resume_path):
            dataset = DataSource("Training Data", data=data)
            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            exit_on_epoch = 1
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=3,
                batch_size=8,
                callbacks=[ExitCallback],
            )
            assert trainer._resumed
            assert trainer.seed == 42
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(trainer_path, "_model"))
            old_backup_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                "_backups",
                "t5-trainer",
                "af8fe7c4d25eacaa",
            )
            assert os.path.isdir(old_backup_path)
            assert not os.path.isfile(os.path.join(old_backup_path, "fingerprint.json"))
            assert not os.path.isdir(os.path.join(old_backup_path, "_checkpoints"))
            new_backup_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                "_backups",
                "t5-trainer",
                "05cba1c44718d59a",
            )
            assert os.path.isdir(new_backup_path)
            assert os.path.isfile(os.path.join(new_backup_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(new_backup_path, "_model"))

    def test_peft_resume(self, create_datadreamer):
        exit_on_epoch: None | int = None
        data = {
            "inputs": ["Bad", "Good", "Bad", "Good", "Bad"],
            "outputs": ["negative", "positive", "negative", "positive", "negative"],
        }

        # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
        with ignore_transformers_warnings():
            from peft import LoraConfig

        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none")

        class ExitCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                if state.epoch == exit_on_epoch:
                    raise ReferenceError("CustomError")

        with create_datadreamer():
            dataset = DataSource("Training Data", data=data)
            trainer = TrainHFClassifier(
                "T5 Trainer", model_name="google/flan-t5-small", peft_config=peft_config
            )
            exit_on_epoch = 1
            with pytest.raises(ReferenceError):
                trainer.train(
                    train_input=dataset.output["inputs"],
                    train_output=dataset.output["outputs"],
                    validation_input=dataset.output["inputs"],
                    validation_output=dataset.output["outputs"],
                    epochs=3,
                    batch_size=8,
                    callbacks=[ExitCallback],
                )
            assert not trainer._resumed
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert not os.path.isdir(os.path.join(trainer_path, "_model"))
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            dataset = DataSource("Training Data", data=data)
            trainer = TrainHFClassifier(
                "T5 Trainer", model_name="google/flan-t5-small", peft_config=peft_config
            )
            exit_on_epoch = None
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=3,
                batch_size=8,
                callbacks=[ExitCallback],
            )
            assert trainer._resumed
            assert trainer.seed == 42
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(trainer_path, "_model"))

    def test_model_card(self, create_datadreamer, capsys):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["Bad", "Good", "Bad", "Good", "Bad"],
                    "outputs": [
                        "negative",
                        "positive",
                        "negative",
                        "positive",
                        "negative",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert trainer._model_card["model_card"]["Model Card"] == [
                "https://huggingface.co/google/flan-t5-small"
            ]
            assert trainer._model_card["data_card"] == dataset._data_card
            capsys.readouterr()
            trainer.model_card()
            captured = capsys.readouterr()
            assert "model_card" in captured.out and "data_card" in captured.out


class TestTrainHFClassifier:
    def test_init(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["Bad", "Good", "Bad", "Good", "Bad"],
                    "outputs": [
                        "negative",
                        "positive",
                        "negative",
                        "positive",
                        "negative",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            data_collator_spy = mocker.spy(DataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            assert (
                data_collator_spy.spy_return["input_ids"]
                == torch.tensor([[3862, 1], [1804, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["attention_mask"]
                == torch.tensor([[1, 1], [1, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["labels"] == torch.tensor([0, 1])
            ).all()
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "6e5e0736fa8a8ef5"
            assert train_result is trainer
            assert trainer.id2label == {0: "negative", 1: "positive"}
            assert trainer.label2id == {"negative": 0, "positive": 1}
            assert not trainer.is_multi_target
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "T5ForSequenceClassification"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "spiece.model"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "T5ForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "spiece.model"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_classifier",
            #     private=True,
            # )

    def test_int_labels(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["Bad", "Good", "Bad", "Good", "Bad"],
                    "outputs": [0, 1, 0, 1, 0],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "8ea8a72037c731c0"
            assert train_result is trainer
            assert trainer.id2label == {0: "0", 1: "1"}
            assert trainer.label2id == {"0": 0, "1": 1}

    def test_multi_target(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["Zero", "One", "Two", "Three"],
                    "outputs": [[], ["A"], ["A", "B"], ["A", "B", "C"]],
                },
            )
            val_dataset = dataset

            trainer = TrainHFClassifier("T5 Trainer", model_name="google/flan-t5-small")
            data_collator_spy = mocker.spy(DataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            assert (
                data_collator_spy.spy_return["input_ids"]
                == torch.tensor([[17971, 1], [555, 1], [2759, 1], [5245, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["attention_mask"]
                == torch.tensor([[1, 1], [1, 1], [1, 1], [1, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["labels"]
                == torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
                )
            ).all()
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "6e330059977a4d84"
            assert train_result is trainer
            assert trainer.id2label == {0: "A", 1: "B", 2: "C"}
            assert trainer.label2id == {"A": 0, "B": 1, "C": 2}
            assert trainer.is_multi_target
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "T5ForSequenceClassification"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "is_multi_target.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "spiece.model"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "T5ForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "spiece.model"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_classifier_multi_target",
            #     private=True,
            # )

    def test_peft(self, create_datadreamer):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["Bad", "Good", "Bad", "Good", "Bad"],
                    "outputs": [
                        "negative",
                        "positive",
                        "negative",
                        "positive",
                        "negative",
                    ],
                },
            )

            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none"
            )

            trainer = TrainHFClassifier(
                "T5 Trainer", model_name="google/flan-t5-small", peft_config=peft_config
            )
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "5720e11579d49bf1"
            assert train_result is trainer
            assert trainer.id2label == {0: "negative", 1: "positive"}
            assert trainer.label2id == {"negative": 0, "positive": 1}
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "PeftModelForSequenceClassification"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(
                    trainer_path,
                    "_checkpoints",
                    "checkpoint-1",
                    "adapter_model.safetensors",
                )
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "spiece.model"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert export_result.config.num_labels == 2
            assert export_result.config.id2label == {0: "negative", 1: "positive"}
            assert export_result.config.label2id == {"negative": 0, "positive": 1}
            assert type(export_result).__name__ == "T5ForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "spiece.model"))
            clear_dir(export_path)
            export_result = trainer.export_to_disk(path=export_path, adapter_only=True)
            assert type(export_result).__name__ == "PeftModelForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(export_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(export_path, "spiece.model"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_classifier_peft_merged",
            #     private=True,
            # )
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_classifier_peft",
            #     private=True,
            #     adapter_only=True,
            # )

    def test_peft_multi_target(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["Zero", "One", "Two", "Three"],
                    "outputs": [[], ["A"], ["A", "B"], ["A", "B", "C"]],
                },
            )
            val_dataset = dataset

            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none"
            )

            trainer = TrainHFClassifier(
                "T5 Trainer", model_name="google/flan-t5-small", peft_config=peft_config
            )
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert export_result.config.num_labels == 3
            assert export_result.config.id2label == {0: "A", 1: "B", 2: "C"}
            assert export_result.config.label2id == {"A": 0, "B": 1, "C": 2}
            assert export_result.config.problem_type == "multi_label_classification"


class TestTrainHFFineTune:
    def test_seq2seq(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": [
                        "A founder of Microsoft is",
                        "A founder of Apple is",
                        "A founder of Y Combinator is",
                        "A founder of Twitter is",
                        "A founder of Tesla is",
                        "A founder of Pixar is",
                    ],
                    "outputs": [
                        "Bill Gates",
                        "Steve Jobs",
                        "Paul Graham",
                        "Jack Dorsey",
                        "Elon Musk",
                        "Ed Catmull",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFFineTune("T5 Trainer", model_name="google/flan-t5-small")
            data_collator_spy = mocker.spy(DataCollatorForSeq2Seq, "__call__")
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            assert (
                data_collator_spy.spy_return["input_ids"]
                == torch.tensor(
                    [[71, 7174, 13, 2803, 19, 1], [71, 7174, 13, 2184, 19, 1]]
                )
            ).all()
            assert (
                data_collator_spy.spy_return["attention_mask"]
                == torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["labels"]
                == torch.tensor([[3259, 11118, 7, 1], [5659, 15106, 1, -100]])
            ).all()
            assert (
                data_collator_spy.spy_return["decoder_input_ids"]
                == torch.tensor([[0, 3259, 11118, 7], [0, 5659, 15106, 1]])
            ).all()
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "f3f60921d65d633e"
            assert train_result is trainer
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "T5ForConditionalGeneration"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "spiece.model"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "T5ForConditionalGeneration"
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "spiece.model"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_finetune_seq2seq",
            #     private=True,
            # )

    def test_causal(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": [
                        "A founder of Microsoft is",
                        "A founder of Apple is",
                        "A founder of Y Combinator is",
                        "A founder of Twitter is",
                        "A founder of Tesla is",
                        "A founder of Pixar is",
                    ],
                    "outputs": [
                        " William Henry Gates (Bill Gates)",
                        " Steve Jobs",
                        " Paul Graham",
                        " Jack Dorsey",
                        " Elon Musk",
                        " Ed Catmull",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFFineTune("GPT-2 Trainer", model_name="gpt2")
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            assert (data_collator_spy.spy_return["input_ids"] == torch.tensor(
                [
                    [32, 9119, 286, 5413, 318, 3977, 8616, 15953, 357, 17798, 15953, 8, 50256],  # noqa: B950
                    [32, 9119, 286, 4196, 318, 6542, 19161, 50256, 50256, 50256, 50256, 50256, 50256],  # noqa: B950
                ]
            )).all()  # fmt: skip
            assert (data_collator_spy.spy_return["labels"] == torch.tensor(
                [
                    [-100, -100, -100, -100, -100, 3977, 8616, 15953, 357, 17798, 15953, 8, 50256],  # noqa: B950
                    [-100, -100, -100, -100, -100, 6542, 19161, 50256, -100, -100, -100, -100, -100],  # noqa: B950
                ]
            )).all()  # fmt: skip
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "e60ef6c220a9d9e2"
            assert train_result is trainer
            assert type(get_orig_model(trainer.model)).__name__ == "GPT2LMHeadModel"
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2LMHeadModel"
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_finetune_causal",
            #     private=True,
            # )

    @pytest.mark.skipif(
        "HUGGING_FACE_HUB_TOKEN" not in os.environ, reason="requires HF Hub token"
    )
    def test_instruction_tune(self, create_datadreamer, mocker):
        # Test instruction-tuning seq2seq
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={"inputs": ["A founder of Apple is"], "outputs": ["Steve Jobs"]},
            )

            trainer = TrainHFFineTune(
                "T5 Trainer",
                model_name="google/flan-t5-small",
                chat_prompt_template=CHAT_PROMPT_TEMPLATES["guanaco_system"],
                system_prompt=SYSTEM_PROMPTS["llama_system"],
            )
            data_collator_spy = mocker.spy(DataCollatorForSeq2Seq, "__call__")
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            assert (
                data_collator_spy.spy_return["input_ids"]
                == torch.tensor(
                    [
                        [
                            1713, 30345, 2149, 10, 148, 33, 3, 9, 2690, 6165, 5, 1713,
                            30345, 3892, 10, 71, 7174, 13, 2184, 19, 1713, 30345, 9255,
                            10, 3, 1
                        ]
                    ]
                )
            ).all()  # fmt: skip
            assert (
                data_collator_spy.spy_return["attention_mask"]
                == torch.tensor(
                    [
                        [
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1
                        ]
                    ]
                )
            ).all()  # fmt: skip
            assert (
                data_collator_spy.spy_return["labels"]
                == torch.tensor([[5659, 15106, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["decoder_input_ids"]
                == torch.tensor([[0, 5659, 15106]])
            ).all()

        # Test instruction-tuning causal
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={"inputs": ["A founder of Apple is"], "outputs": ["Steve Jobs"]},
            )

            trainer = TrainHFFineTune(
                "GPT-2 Trainer",
                model_name="gpt2",
                chat_prompt_template=CHAT_PROMPT_TEMPLATES["guanaco_system"],
                system_prompt=SYSTEM_PROMPTS["llama_system"],
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            assert (data_collator_spy.spy_return["input_ids"] == torch.tensor(
                [
                    [
                        21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524,
                        25, 317, 9119, 286, 4196, 318, 198, 21017, 15286, 25, 220, 19206,
                        19161, 50256
                    ]
                ]
            )).all()  # fmt: skip
            assert (data_collator_spy.spy_return["labels"] == torch.tensor(
                [
                    [
                        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                        -100, -100, -100, 19206, 19161, 50256
                    ]
                ]
            )).all()  # fmt: skip

        # Test LLaMa-2 prompt
        with create_datadreamer():
            llama_trainer = TrainHFFineTune(
                "LLaMa-2 Trainer", model_name="meta-llama/Llama-2-7b-chat-hf"
            )
            assert (
                llama_trainer.chat_prompt_template
                == CHAT_PROMPT_TEMPLATES["llama_system"]
            )
            assert llama_trainer.system_prompt == SYSTEM_PROMPTS["llama_system"]


class TestTrainSentenceTransformer:
    def test_do_not_use_train(self, create_datadreamer):
        with create_datadreamer():
            trainer = TrainSentenceTransformer(
                "Distilbert NLI Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
            )
            with pytest.raises(RuntimeError):
                trainer.train()

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            trainer = TrainSentenceTransformer(
                "Distilbert NLI Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
            )
            assert isinstance(trainer.citation, list)
            assert len(trainer.citation) == 3

    def test_triplets(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "anchors": ["Base", "Base", "Base", "Base", "Base", "Base"],
                    "positives": [
                        "Wonderful",
                        "Great",
                        "Excellent",
                        "Amazing",
                        "Magnificent",
                        "Fantastic",
                    ],
                    "negatives": [
                        "Bad",
                        "Terrible",
                        "Horrible",
                        "Awful",
                        "Atrocious",
                        "Abhorrent",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainSentenceTransformer(
                "Distilbert NLI Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train_with_triplets(
                train_anchors=dataset.output["anchors"],
                train_positives=dataset.output["positives"],
                train_negatives=dataset.output["negatives"],
                validation_anchors=val_dataset.output["anchors"],
                validation_positives=val_dataset.output["positives"],
                validation_negatives=val_dataset.output["negatives"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            assert (
                data_collator_spy.spy_return["anchor_input_ids"]
                == torch.tensor([[101, 2918, 102], [101, 2918, 102]])
            ).all()
            assert (
                data_collator_spy.spy_return["anchor_attention_mask"]
                == torch.tensor([[1, 1, 1], [1, 1, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["positive_input_ids"]
                == torch.tensor([[101, 6919, 102], [101, 2307, 102]])
            ).all()
            assert (
                data_collator_spy.spy_return["positive_attention_mask"]
                == torch.tensor([[1, 1, 1], [1, 1, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["negative_input_ids"]
                == torch.tensor([[101, 2919, 102], [101, 6659, 102]])
            ).all()
            assert (
                data_collator_spy.spy_return["negative_attention_mask"]
                == torch.tensor([[1, 1, 1], [1, 1, 1]])
            ).all()
            assert data_collator_spy.spy_return["labels"] == torch.tensor(-1)
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "0f545a22e318701e"
            assert train_result is trainer
            assert type(get_orig_model(trainer.model)).__name__ == "SentenceTransformer"
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "1_Pooling", "config.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "tokenizer.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "SentenceTransformer"
            assert os.path.isfile(os.path.join(export_path, "1_Pooling", "config.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "tokenizer.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_sentence_transformers_triplets",
            #     private=True,
            # )

    def test_peft(self, create_datadreamer):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "anchors": ["Base", "Base", "Base", "Base", "Base", "Base"],
                    "positives": [
                        "Wonderful",
                        "Great",
                        "Excellent",
                        "Amazing",
                        "Magnificent",
                        "Fantastic",
                    ],
                    "negatives": [
                        "Bad",
                        "Terrible",
                        "Horrible",
                        "Awful",
                        "Atrocious",
                        "Abhorrent",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["lin1", "lin2"],
            )

            trainer = TrainSentenceTransformer(
                "Distilbert NLI Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
                peft_config=peft_config,
            )
            train_result = trainer.train_with_triplets(
                train_anchors=dataset.output["anchors"],
                train_positives=dataset.output["positives"],
                train_negatives=dataset.output["negatives"],
                validation_anchors=val_dataset.output["anchors"],
                validation_positives=val_dataset.output["positives"],
                validation_negatives=val_dataset.output["negatives"],
                epochs=1,
                batch_size=8,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "d7919437f0b2e066"
            assert train_result is trainer
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "PeftModelForFeatureExtraction"
            )
            assert trainer.model.encode(["Sentence A.", "Sentence B."]).shape == (
                2,
                768,
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(
                    trainer_path, "_checkpoints", "checkpoint-1", "adapter_model.bin"
                )
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "tokenizer.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "SentenceTransformer"
            assert os.path.isfile(os.path.join(export_path, "1_Pooling", "config.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "tokenizer.json"))
            clear_dir(export_path)
            export_result = trainer.export_to_disk(path=export_path, adapter_only=True)
            assert type(export_result).__name__ == "PeftModelForFeatureExtraction"
            assert os.path.isfile(
                os.path.join(export_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(export_path, "tokenizer.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_sentence_transformers_triplets_peft_merged",
            #     private=True,
            # )
            # trainer.publish_to_hf_hub(
            #     repo_id="test_sentence_transformers_triplets_peft",
            #     private=True,
            #     adapter_only=True,
            # )

    def test_peft_resume(self, create_datadreamer):
        exit_on_epoch: None | int = None
        data = {
            "anchors": ["Base", "Base", "Base", "Base", "Base", "Base"],
            "positives": [
                "Wonderful",
                "Great",
                "Excellent",
                "Amazing",
                "Magnificent",
                "Fantastic",
            ],
            "negatives": [
                "Bad",
                "Terrible",
                "Horrible",
                "Awful",
                "Atrocious",
                "Abhorrent",
            ],
        }

        class ExitCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                if state.epoch == exit_on_epoch:
                    raise ReferenceError("CustomError")

        # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
        with ignore_transformers_warnings():
            from peft import LoraConfig

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["lin1", "lin2"],
        )

        with create_datadreamer():
            dataset = DataSource("Training Data", data=data)
            trainer = TrainSentenceTransformer(
                "Distilbert NLI Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
                peft_config=peft_config,
            )
            exit_on_epoch = 1
            with pytest.raises(ReferenceError):
                trainer.train_with_triplets(
                    train_anchors=dataset.output["anchors"],
                    train_positives=dataset.output["positives"],
                    train_negatives=dataset.output["negatives"],
                    validation_anchors=dataset.output["anchors"],
                    validation_positives=dataset.output["positives"],
                    validation_negatives=dataset.output["negatives"],
                    epochs=3,
                    batch_size=8,
                    callbacks=[ExitCallback],
                )
            assert not trainer._resumed
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert not os.path.isdir(os.path.join(trainer_path, "_model"))
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            dataset = DataSource("Training Data", data=data)
            trainer = TrainSentenceTransformer(
                "Distilbert NLI Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
                peft_config=peft_config,
            )
            exit_on_epoch = None
            trainer.train_with_triplets(
                train_anchors=dataset.output["anchors"],
                train_positives=dataset.output["positives"],
                train_negatives=dataset.output["negatives"],
                validation_anchors=dataset.output["anchors"],
                validation_positives=dataset.output["positives"],
                validation_negatives=dataset.output["negatives"],
                epochs=3,
                batch_size=8,
                callbacks=[ExitCallback],
            )
            assert trainer._resumed
            assert trainer.seed == 42
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(trainer_path, "_model"))

    def test_positive_pairs(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "anchors": ["Apple", "Sky", "Grass", "Carrots", "Sun", "Flamingo"],
                    "positives": ["Red", "Blue", "Green", "Orange", "Yellow", "Pink"],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainSentenceTransformer(
                "Distilbert NLI Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train_with_positive_pairs(
                train_anchors=dataset.output["anchors"],
                train_positives=dataset.output["positives"],
                validation_anchors=val_dataset.output["anchors"],
                validation_positives=val_dataset.output["positives"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            assert (
                data_collator_spy.spy_return["anchor_input_ids"]
                == torch.tensor([[101, 6207, 102], [101, 3712, 102]])
            ).all()
            assert (
                data_collator_spy.spy_return["anchor_attention_mask"]
                == torch.tensor([[1, 1, 1], [1, 1, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["positive_input_ids"]
                == torch.tensor([[101, 2417, 102], [101, 2630, 102]])
            ).all()
            assert (
                data_collator_spy.spy_return["positive_attention_mask"]
                == torch.tensor([[1, 1, 1], [1, 1, 1]])
            ).all()
            assert data_collator_spy.spy_return["labels"] == torch.tensor(-1)
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "7e69fc96c61a4b57"
            assert train_result is trainer
            assert type(get_orig_model(trainer.model)).__name__ == "SentenceTransformer"
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "1_Pooling", "config.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "tokenizer.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "SentenceTransformer"
            assert os.path.isfile(os.path.join(export_path, "1_Pooling", "config.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "tokenizer.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_sentence_transformers_positive_pairs",
            #     private=True,
            # )

    def test_labeled_pairs(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "anchors": ["Base", "Base", "Base", "Base", "Base", "Base"],
                    "others": [
                        "Wonderful",
                        "Bad",
                        "Great",
                        "Terrible",
                        "Excellent",
                        "Horrible",
                    ],
                    "labels": [1, -1, 1, -1, 1, -1],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainSentenceTransformer(
                "Distilbert NLI Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train_with_labeled_pairs(
                train_anchors=dataset.output["anchors"],
                train_others=dataset.output["others"],
                train_labels=dataset.output["labels"],
                validation_anchors=val_dataset.output["anchors"],
                validation_others=val_dataset.output["others"],
                validation_labels=val_dataset.output["labels"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            assert (
                data_collator_spy.spy_return["anchor_input_ids"]
                == torch.tensor([[101, 2918, 102], [101, 2918, 102]])
            ).all()
            assert (
                data_collator_spy.spy_return["anchor_attention_mask"]
                == torch.tensor([[1, 1, 1], [1, 1, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["positive_input_ids"]
                == torch.tensor([[101, 6919, 102], [101, 2919, 102]])
            ).all()
            assert (
                data_collator_spy.spy_return["positive_attention_mask"]
                == torch.tensor([[1, 1, 1], [1, 1, 1]])
            ).all()
            assert (
                data_collator_spy.spy_return["labels"] == torch.tensor([1, -1])
            ).all()
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "e9156c7c3b03b42e"
            assert train_result is trainer
            assert type(get_orig_model(trainer.model)).__name__ == "SentenceTransformer"
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "1_Pooling", "config.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "tokenizer.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "SentenceTransformer"
            assert os.path.isfile(os.path.join(export_path, "1_Pooling", "config.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "tokenizer.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_sentence_transformers_labeled_pairs",
            #     private=True,
            # )


class TestTrainHFDPO:
    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            trainer = TrainHFDPO("GPT-2 Trainer", model_name="gpt2")
            assert isinstance(trainer.citation, list)
            assert len(trainer.citation) == 4

    def test_seq2seq(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "prompts": [
                        "The color of the sky is",
                        "Firetrucks are",
                        "The color of an apple is",
                        "The color of grass is",
                        "The color of clouds are",
                        "The color of the sun is",
                    ],
                    "chosen": [
                        " purple",
                        " bright yellow",
                        " orange",
                        " blue",
                        " red",
                        " green",
                    ],
                    "rejected": [
                        " blue",
                        " red",
                        " red",
                        " green",
                        " white",
                        " yellow",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFDPO("T5 Trainer", model_name="google/flan-t5-small")
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_prompts=dataset.output["prompts"],
                train_chosen=dataset.output["chosen"],
                train_rejected=dataset.output["rejected"],
                validation_prompts=val_dataset.output["prompts"],
                validation_chosen=val_dataset.output["chosen"],
                validation_rejected=val_dataset.output["rejected"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 5
            spy_return_value = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in data_collator_spy.spy_return.items()
            }
            assert "reference_chosen_logps" in spy_return_value
            assert "reference_rejected_logps" in spy_return_value
            del spy_return_value["reference_chosen_logps"]
            del spy_return_value["reference_rejected_logps"]
            assert spy_return_value == {
                "prompt_input_ids": [
                    [37, 945, 13, 8, 5796, 19, 1],
                    [3655, 28105, 7, 33, 1, 0, 0],
                ],
                "prompt_attention_mask": [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]],
                "chosen_labels": [[11094, 1, -100], [2756, 4459, 1]],
                "rejected_labels": [[1692, 1], [1131, 1]],
                "prompt": ["The color of the sky is", "Firetrucks are"],
                "chosen": [
                    "The color of the sky is purple",
                    "Firetrucks are bright yellow",
                ],
                "rejected": ["The color of the sky is blue", "Firetrucks are red"],
                "chosen_response_only": [" purple", " bright yellow"],
                "rejected_response_only": [" blue", " red"],
            }
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "f35a4a241a62b91f"
            assert train_result is trainer
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "T5ForConditionalGeneration"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "spiece.model"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "T5ForConditionalGeneration"
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "spiece.model"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_dpo_seq2seq",
            #     private=True,
            # )

    def test_causal(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "prompts": [
                        "The color of the sky is",
                        "Firetrucks are",
                        "The color of an apple is",
                        "The color of grass is",
                        "The color of clouds are",
                        "The color of the sun is",
                    ],
                    "chosen": ["purple", "yellow", "orange", "blue", "red", "green"],
                    "rejected": ["blue", "red", "red", "green", "white", "yellow"],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFDPO(
                "GPT-2 Trainer",
                model_name="gpt2",
                chat_prompt_template=CHAT_PROMPT_TEMPLATES["guanaco_system"],
                system_prompt=SYSTEM_PROMPTS["llama_system"],
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_prompts=dataset.output["prompts"],
                train_chosen=dataset.output["chosen"],
                train_rejected=dataset.output["rejected"],
                validation_prompts=val_dataset.output["prompts"],
                validation_chosen=val_dataset.output["chosen"],
                validation_rejected=val_dataset.output["rejected"],
                epochs=1,
                batch_size=8,
                precompute_ref_log_probs=False,  # We test precompute_ref_log_probs here
            )
            assert data_collator_spy.call_count == 3
            spy_return_value = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in data_collator_spy.spy_return.items()
            }
            assert spy_return_value == {
                "prompt_input_ids": [
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 383, 3124, 286, 262, 6766, 318, 198, 21017, 15286, 25, 220],  # noqa: B950
                    [50256, 21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 3764, 83, 622, 4657, 389, 198, 21017, 15286, 25, 220],  # noqa: B950
                ],
                "prompt_attention_mask": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa: B950
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa: B950
                ],
                "chosen_input_ids": [
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 383, 3124, 286, 262, 6766, 318, 198, 21017, 15286, 25, 220, 14225, 1154, 50256],  # noqa: B950
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 3764, 83, 622, 4657, 389, 198, 21017, 15286, 25, 220, 36022, 50256, 50256, 50256],  # noqa: B950
                ],
                "chosen_attention_mask": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa: B950
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # noqa: B950
                ],
                "chosen_labels": [
                    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 14225, 1154, 50256],  # noqa: B950
                    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 36022, 50256, -100, -100],  # noqa: B950
                ],
                "rejected_input_ids": [
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 383, 3124, 286, 262, 6766, 318, 198, 21017, 15286, 25, 220, 17585, 50256],  # noqa: B950
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 3764, 83, 622, 4657, 389, 198, 21017, 15286, 25, 220, 445, 50256, 50256],  # noqa: B950
                ],
                "rejected_attention_mask": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa: B950
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # noqa: B950
                ],
                "rejected_labels": [
                    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 17585, 50256],  # noqa: B950
                    [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 445, 50256, -100],  # noqa: B950
                ],
                "prompt": [
                    "### System: You are a helpful assistant.\n### Human: The color of the sky is\n### Assistant: ",  # noqa: B950
                    "### System: You are a helpful assistant.\n### Human: Firetrucks are\n### Assistant: ",  # noqa: B950
                ],
                "chosen": [
                    "### System: You are a helpful assistant.\n### Human: The color of the sky is\n### Assistant: purple",  # noqa: B950
                    "### System: You are a helpful assistant.\n### Human: Firetrucks are\n### Assistant: yellow",  # noqa: B950
                ],
                "rejected": [
                    "### System: You are a helpful assistant.\n### Human: The color of the sky is\n### Assistant: blue",  # noqa: B950
                    "### System: You are a helpful assistant.\n### Human: Firetrucks are\n### Assistant: red",  # noqa: B950
                ],
                "chosen_response_only": ["purple", "yellow"],
                "rejected_response_only": ["blue", "red"],
            }  # fmt: skip
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "4a0c13237b9d4988"
            assert train_result is trainer
            assert type(get_orig_model(trainer.model)).__name__ == "GPT2LMHeadModel"
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2LMHeadModel"
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_dpo_causal",
            #     private=True,
            # )

    def test_peft(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "prompts": [
                        "The color of the sky is",
                        "Firetrucks are",
                        "The color of an apple is",
                        "The color of grass is",
                        "The color of clouds are",
                        "The color of the sun is",
                    ],
                    "chosen": [
                        " purple",
                        " yellow",
                        " orange",
                        " blue",
                        " red",
                        " green",
                    ],
                    "rejected": [
                        " blue",
                        " red",
                        " red",
                        " green",
                        " white",
                        " yellow",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["c_proj"],
                fan_in_fan_out=True,
            )

            trainer = TrainHFDPO(
                "GPT-2 Trainer", model_name="gpt2", peft_config=peft_config
            )
            train_result = trainer.train(
                train_prompts=dataset.output["prompts"],
                train_chosen=dataset.output["chosen"],
                train_rejected=dataset.output["rejected"],
                validation_prompts=val_dataset.output["prompts"],
                validation_chosen=val_dataset.output["chosen"],
                validation_rejected=val_dataset.output["rejected"],
                epochs=1,
                batch_size=8,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "199c2a7e9ac922c7"
            assert train_result is trainer
            assert (
                type(get_orig_model(trainer.model)).__name__ == "PeftModelForCausalLM"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(
                    trainer_path,
                    "_checkpoints",
                    "checkpoint-1",
                    "adapter_model.safetensors",
                )
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2LMHeadModel"
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            clear_dir(export_path)
            export_result = trainer.export_to_disk(path=export_path, adapter_only=True)
            assert type(export_result).__name__ == "PeftModelForCausalLM"
            assert os.path.isfile(
                os.path.join(export_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_dpo_peft_merged",
            #     private=True,
            # )
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_dpo_peft",
            #     private=True,
            #     adapter_only=True,
            # )


class TestTrainHFRewardModel:
    def test_do_not_use_train(self, create_datadreamer):
        with create_datadreamer():
            trainer = TrainHFRewardModel("GPT-2 Trainer", model_name="gpt2")
            with pytest.raises(RuntimeError):
                trainer.train()

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            trainer = TrainHFRewardModel("GPT-2 Trainer", model_name="gpt2")
            assert isinstance(trainer.citation, list)
            assert len(trainer.citation) == 4

    def test_pairs_seq2seq(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "prompts": [
                        "The color of the sky is",
                        "Firetrucks are",
                        "The color of an apple is",
                        "The color of grass is",
                        "The color of clouds are",
                        "The color of the sun is",
                    ],
                    "chosen": [
                        " purple",
                        " bright yellow",
                        " orange",
                        " blue",
                        " red",
                        " green",
                    ],
                    "rejected": [
                        " blue",
                        " red",
                        " red",
                        " green",
                        " white",
                        " yellow",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFRewardModel(
                "T5 Trainer", model_name="google/flan-t5-small"
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train_with_pairs(
                train_prompts=dataset.output["prompts"],
                train_chosen=dataset.output["chosen"],
                train_rejected=dataset.output["rejected"],
                validation_prompts=val_dataset.output["prompts"],
                validation_chosen=val_dataset.output["chosen"],
                validation_rejected=val_dataset.output["rejected"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            spy_return_value = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in data_collator_spy.spy_return.items()
            }
            assert spy_return_value == {
                "input_ids_chosen": [
                    [37, 945, 13, 8, 5796, 19, 11094, 1],
                    [3655, 28105, 7, 33, 2756, 4459, 1, 0],
                ],
                "attention_mask_chosen": [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0],
                ],
                "return_loss": True,
                "input_ids_rejected": [
                    [37, 945, 13, 8, 5796, 19, 1692, 1],
                    [3655, 28105, 7, 33, 1131, 1, 0, 0],
                ],
                "attention_mask_rejected": [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0, 0],
                ],
            }
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "84c8c57fbbb8e313"
            assert train_result is trainer
            assert trainer.id2label == {0: "reward"}
            assert trainer.label2id == {"reward": 0}
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "T5ForSequenceClassification"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "spiece.model"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "T5ForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "spiece.model"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_reward_pairs_seq2seq",
            #     private=True,
            # )

    def test_pairs_causal(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "prompts": [
                        "The color of the sky is",
                        "Firetrucks are",
                        "The color of an apple is",
                        "The color of grass is",
                        "The color of clouds are",
                        "The color of the sun is",
                    ],
                    "chosen": ["purple", "yellow", "orange", "blue", "red", "green"],
                    "rejected": ["blue", "red", "red", "green", "white", "yellow"],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFRewardModel(
                "GPT-2 Trainer",
                model_name="gpt2",
                chat_prompt_template=CHAT_PROMPT_TEMPLATES["guanaco_system"],
                system_prompt=SYSTEM_PROMPTS["llama_system"],
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train_with_pairs(
                train_prompts=dataset.output["prompts"],
                train_chosen=dataset.output["chosen"],
                train_rejected=dataset.output["rejected"],
                validation_prompts=val_dataset.output["prompts"],
                validation_chosen=val_dataset.output["chosen"],
                validation_rejected=val_dataset.output["rejected"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            spy_return_value = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in data_collator_spy.spy_return.items()
            }
            assert spy_return_value == {
                "input_ids_chosen": [
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 383, 3124, 286, 262, 6766, 318, 198, 21017, 15286, 25, 14032],  # noqa: B950
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 3764, 83, 622, 4657, 389, 198, 21017, 15286, 25, 7872, 50256],  # noqa: B950
                ],
                "attention_mask_chosen": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa: B950
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # noqa: B950
                ],
                "return_loss": True,
                "input_ids_rejected": [
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 383, 3124, 286, 262, 6766, 318, 198, 21017, 15286, 25, 4171],  # noqa: B950
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 3764, 83, 622, 4657, 389, 198, 21017, 15286, 25, 2266, 50256],  # noqa: B950
                ],
                "attention_mask_rejected": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa: B950
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  # noqa: B950
                ],
            }  # fmt: skip
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "66b84c7663ca340a"
            assert train_result is trainer
            assert trainer.id2label == {0: "reward"}
            assert trainer.label2id == {"reward": 0}
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "GPT2ForSequenceClassification"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2ForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_reward_pairs_causal",
            #     private=True
            # )

    def test_pairs_peft(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "prompts": [
                        "The color of the sky is",
                        "Firetrucks are",
                        "The color of an apple is",
                        "The color of grass is",
                        "The color of clouds are",
                        "The color of the sun is",
                    ],
                    "chosen": [
                        " purple",
                        " yellow",
                        " orange",
                        " blue",
                        " red",
                        " green",
                    ],
                    "rejected": [
                        " blue",
                        " red",
                        " red",
                        " green",
                        " white",
                        " yellow",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["c_proj"],
                fan_in_fan_out=True,
            )

            trainer = TrainHFRewardModel(
                "GPT-2 Trainer", model_name="gpt2", peft_config=peft_config
            )
            train_result = trainer.train_with_pairs(
                train_prompts=dataset.output["prompts"],
                train_chosen=dataset.output["chosen"],
                train_rejected=dataset.output["rejected"],
                validation_prompts=val_dataset.output["prompts"],
                validation_chosen=val_dataset.output["chosen"],
                validation_rejected=val_dataset.output["rejected"],
                epochs=1,
                batch_size=8,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "d2f8bf5c4c890ad8"
            assert train_result is trainer
            assert trainer.id2label == {0: "reward"}
            assert trainer.label2id == {"reward": 0}
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "PeftModelForSequenceClassification"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(
                    trainer_path,
                    "_checkpoints",
                    "checkpoint-1",
                    "adapter_model.safetensors",
                )
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2ForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            clear_dir(export_path)
            export_result = trainer.export_to_disk(path=export_path, adapter_only=True)
            assert type(export_result).__name__ == "PeftModelForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(export_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_reward_pairs_peft_merged",
            #     private=True
            # )
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_reward_pairs_peft",
            #     private=True,
            #     adapter_only=True
            # )

    def test_pairs_and_scores_causal(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "prompts": [
                        "The color of the sky is",
                        "Firetrucks are",
                        "The color of an apple is",
                        "The color of grass is",
                        "The color of clouds are",
                        "The color of the sun is",
                    ],
                    "chosen": [
                        " purple",
                        " yellow",
                        " orange",
                        " blue",
                        " red",
                        " green",
                    ],
                    "chosen_scores": [5, 5, 5, 5, 5, 5],
                    "rejected": [
                        " blue",
                        " red",
                        " red",
                        " green",
                        " white",
                        " yellow",
                    ],
                    "rejected_scores": [1, 1, 1, 1, 1, 1],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFRewardModel("GPT-2 Trainer", model_name="gpt2")
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train_with_pairs_and_scores(
                train_prompts=dataset.output["prompts"],
                train_chosen=dataset.output["chosen"],
                train_chosen_scores=dataset.output["chosen_scores"],
                train_rejected=dataset.output["rejected"],
                train_rejected_scores=dataset.output["rejected_scores"],
                validation_prompts=val_dataset.output["prompts"],
                validation_chosen=val_dataset.output["chosen"],
                validation_chosen_scores=val_dataset.output["chosen_scores"],
                validation_rejected=val_dataset.output["rejected"],
                validation_rejected_scores=val_dataset.output["rejected_scores"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            spy_return_value = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in data_collator_spy.spy_return.items()
            }
            assert spy_return_value == {
                "input_ids_chosen": [
                    [464, 3124, 286, 262, 6766, 318, 14032],
                    [13543, 83, 622, 4657, 389, 7872, 50256],
                ],
                "attention_mask_chosen": [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0]],
                "return_loss": True,
                "input_ids_rejected": [
                    [464, 3124, 286, 262, 6766, 318, 4171],
                    [13543, 83, 622, 4657, 389, 2266, 50256],
                ],
                "attention_mask_rejected": [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 0],
                ],
                "margin": [4, 4],
            }
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "e2e6c96e31bb1e8c"
            assert train_result is trainer
            assert trainer.id2label == {0: "reward"}
            assert trainer.label2id == {"reward": 0}
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "GPT2ForSequenceClassification"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2ForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_reward_pairs_and_scores_causal",
            #     private=True
            # )

    def test_scores_causal(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "prompts": [
                        "The color of the sky is",
                        "The color of the sky is",
                        "Firetrucks are",
                        "Firetrucks are",
                        "The color of an apple is",
                        "The color of an apple is",
                    ],
                    "generations": ["purple", "blue", "yellow", "red", "orange", "red"],
                    "scores": [5, 1, 5, 1, 5, 1],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainHFRewardModel(
                "GPT-2 Trainer",
                model_name="gpt2",
                chat_prompt_template=CHAT_PROMPT_TEMPLATES["guanaco_system"],
                system_prompt=SYSTEM_PROMPTS["llama_system"],
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train_with_scores(
                train_prompts=dataset.output["prompts"],
                train_generations=dataset.output["generations"],
                train_scores=dataset.output["scores"],
                validation_prompts=val_dataset.output["prompts"],
                validation_generations=val_dataset.output["generations"],
                validation_scores=val_dataset.output["scores"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 3
            spy_return_value = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in data_collator_spy.spy_return.items()
            }
            assert spy_return_value == {
                "input_ids": [
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 383, 3124, 286, 262, 6766, 318, 198, 21017, 15286, 25, 14032],  # noqa: B950
                    [21017, 4482, 25, 921, 389, 257, 7613, 8796, 13, 198, 21017, 5524, 25, 383, 3124, 286, 262, 6766, 318, 198, 21017, 15286, 25, 4171],  # noqa: B950
                ],
                "attention_mask": [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa: B950
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # noqa: B950
                ],
                "labels": [5.0, 1.0],
            }  # fmt: skip
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "b6306590705d3bf5"
            assert train_result is trainer
            assert trainer.id2label == {0: "reward"}
            assert trainer.label2id == {"reward": 0}
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "GPT2ForSequenceClassification"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2ForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_reward_scores_causal",
            #     private=True
            # )

    def test_scores_peft(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "prompts": [
                        "The color of the sky is",
                        "The color of the sky is",
                        "Firetrucks are",
                        "Firetrucks are",
                        "The color of an apple is",
                        "The color of an apple is",
                    ],
                    "generations": [
                        " purple",
                        " blue",
                        " yellow",
                        " red",
                        " orange",
                        " red",
                    ],
                    "scores": [5, 1, 5, 1, 5, 1],
                },
            )
            val_dataset = dataset.take(2)

            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["c_proj"],
                fan_in_fan_out=True,
            )

            trainer = TrainHFRewardModel(
                "GPT-2 Trainer", model_name="gpt2", peft_config=peft_config
            )
            train_result = trainer.train_with_scores(
                train_prompts=dataset.output["prompts"],
                train_generations=dataset.output["generations"],
                train_scores=dataset.output["scores"],
                validation_prompts=val_dataset.output["prompts"],
                validation_generations=val_dataset.output["generations"],
                validation_scores=val_dataset.output["scores"],
                epochs=1,
                batch_size=8,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "739c9b28717a62a0"
            assert train_result is trainer
            assert trainer.id2label == {0: "reward"}
            assert trainer.label2id == {"reward": 0}
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "PeftModelForSequenceClassification"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(
                    trainer_path,
                    "_checkpoints",
                    "checkpoint-1",
                    "adapter_model.safetensors",
                )
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2ForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            clear_dir(export_path)
            export_result = trainer.export_to_disk(path=export_path, adapter_only=True)
            assert type(export_result).__name__ == "PeftModelForSequenceClassification"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(export_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_reward_scores_peft_merged",
            #     private=True
            # )
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_reward_scores_peft",
            #     private=True,
            #     adapter_only=True
            # )


class TestTrainHFPPO:
    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            trainer = TrainHFPPO("GPT-2 Trainer", model_name="gpt2")
            assert isinstance(trainer.citation, list)
            assert len(trainer.citation) == 5

    def test_seq2seq(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data", data={"prompts": ["Hi hi, "] + ["Hi, "] * 5}
            )
            val_dataset = dataset.take(2)

            sentiment_model = pipeline(
                "text-classification", model="lvwerra/distilbert-imdb"
            )

            def reward_model(texts: list[str]) -> list[float]:
                return [
                    list(filter(lambda x: x["label"] == "POSITIVE", r))[0]["score"]
                    for r in sentiment_model(texts, top_k=2)
                ]

            trainer = TrainHFPPO("T5 Trainer", model_name="google/t5-small-lm-adapt")
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_prompts=dataset.output["prompts"],
                validation_prompts=val_dataset.output["prompts"],
                reward_model=reward_model,
                epochs=1,
                batch_size=6,
                max_new_tokens=5,
            )
            assert data_collator_spy.call_count == 3
            assert (
                data_collator_spy.spy_return["input_ids"]
                == torch.tensor([[2018, 7102, 6, 3, 1], [2018, 6, 3, 1, 0]])
            ).all()
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "304faae0b7c01ee0"
            assert train_result is trainer
            assert (
                type(get_orig_model(trainer.model)).__name__
                == "T5ForConditionalGeneration"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "spiece.model"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "T5ForConditionalGeneration"
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "spiece.model"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_ppo_seq2seq",
            #     private=True,
            # )

    def test_causal(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data", data={"prompts": ["Hi hi"] + ["Hi"] * 5}
            )
            val_dataset = dataset.take(2)

            sentiment_model = pipeline(
                "text-classification", model="lvwerra/distilbert-imdb"
            )

            def reward_model(texts: list[str]) -> list[float]:
                return [
                    list(filter(lambda x: x["label"] == "POSITIVE", r))[0]["score"]
                    for r in sentiment_model(texts, top_k=2)
                ]

            trainer = TrainHFPPO(
                "GPT-2 Trainer", model_name="gpt2", chat_prompt_template="{{prompt}}, "
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_prompts=dataset.output["prompts"],
                validation_prompts=val_dataset.output["prompts"],
                reward_model=reward_model,
                epochs=1,
                batch_size=6,
                max_new_tokens=4,
                # Otherwise, we get a warning 'The average ratio of batch (11.98)
                # exceeds threshold 10.00. Skipping batch.'
                ratio_threshold=12.0,
            )
            assert data_collator_spy.call_count == 3
            assert (
                data_collator_spy.spy_return["input_ids"]
                == torch.tensor([[17250, 23105, 11, 220], [50256, 17250, 11, 220]])
            ).all()
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "38f87fa4c896ffc3"
            assert train_result is trainer
            assert type(get_orig_model(trainer.model)).__name__ == "GPT2LMHeadModel"
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2LMHeadModel"
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_ppo_causal",
            #     private=True,
            # )

    def test_resume(self, create_datadreamer, mocker):
        exit_on_epoch: None | int = None
        data = {"prompts": ["Hi, "] * 6}

        class ExitCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                if state.epoch == exit_on_epoch:
                    raise ReferenceError("CustomError")

        with create_datadreamer():
            dataset = DataSource("Training Data", data=data)
            val_dataset = dataset.take(2)

            sentiment_model = pipeline(
                "text-classification", model="lvwerra/distilbert-imdb"
            )

            def reward_model(texts: list[str]) -> list[float]:
                return [
                    list(filter(lambda x: x["label"] == "POSITIVE", r))[0]["score"]
                    for r in sentiment_model(texts, top_k=2)
                ]

            trainer = TrainHFPPO("GPT-2 Trainer", model_name="gpt2")
            exit_on_epoch = 1
            with pytest.raises(ReferenceError):
                trainer.train(
                    train_prompts=dataset.output["prompts"],
                    validation_prompts=val_dataset.output["prompts"],
                    reward_model=reward_model,
                    epochs=3,
                    batch_size=6,
                    max_new_tokens=4,
                    callbacks=[ExitCallback],
                )
            assert not trainer._resumed
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert not os.path.isdir(os.path.join(trainer_path, "_model"))
            resume_path = os.path.basename(DataDreamer.get_output_folder_path())

        with create_datadreamer(resume_path):
            dataset = DataSource("Training Data", data=data)
            val_dataset = dataset.take(2)

            sentiment_model = pipeline(
                "text-classification", model="lvwerra/distilbert-imdb"
            )

            def reward_model(texts: list[str]) -> list[float]:
                return [
                    list(filter(lambda x: x["label"] == "POSITIVE", r))[0]["score"]
                    for r in sentiment_model(texts, top_k=2)
                ]

            trainer = TrainHFPPO("GPT-2 Trainer", model_name="gpt2")
            exit_on_epoch = None
            trainer.train(
                train_prompts=dataset.output["prompts"],
                validation_prompts=val_dataset.output["prompts"],
                reward_model=reward_model,
                epochs=3,
                batch_size=6,
                max_new_tokens=4,
                callbacks=[ExitCallback],
            )
            assert trainer._resumed
            assert trainer.seed == 42
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(trainer_path, "_model"))

    def test_peft(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource("Training Data", data={"prompts": ["Hi, "] * 6})
            val_dataset = dataset.take(2)

            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["c_proj"],
                fan_in_fan_out=True,
            )

            sentiment_model = pipeline(
                "text-classification", model="lvwerra/distilbert-imdb"
            )

            def reward_model(texts: list[str]) -> list[float]:
                return [
                    list(filter(lambda x: x["label"] == "POSITIVE", r))[0]["score"]
                    for r in sentiment_model(texts, top_k=2)
                ]

            trainer = TrainHFPPO(
                "GPT-2 Trainer", model_name="gpt2", peft_config=peft_config
            )
            train_result = trainer.train(
                train_prompts=dataset.output["prompts"],
                validation_prompts=val_dataset.output["prompts"],
                reward_model=reward_model,
                epochs=1,
                batch_size=6,
                max_new_tokens=4,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "54bb288a5f0215b8"
            assert train_result is trainer
            assert (
                type(get_orig_model(trainer.model)).__name__ == "PeftModelForCausalLM"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(
                    trainer_path,
                    "_checkpoints",
                    "checkpoint-1",
                    "adapter_model.safetensors",
                )
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2LMHeadModel"
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            clear_dir(export_path)
            export_result = trainer.export_to_disk(path=export_path, adapter_only=True)
            assert type(export_result).__name__ == "PeftModelForCausalLM"
            assert os.path.isfile(
                os.path.join(export_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_ppo_peft_merged",
            #     private=True,
            # )
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_ppo_peft",
            #     private=True,
            #     adapter_only=True,
            # )

    def test_reward_model_from_trainer(self, create_datadreamer, mocker):
        with create_datadreamer():
            # Train reward model
            splits = (
                DataSource(
                    "Reward Training Data",
                    data={
                        "prompts": ["Hi, "] * 10,
                        "chosen": [
                            "it so wonderful",
                            "that is great",
                            "that's excellent",
                            "wow amazing",
                            "that was magnificent",
                            "this is fantastic",
                            "that's superior",
                            "that is brilliant",
                            "i'm tremendous",
                            "you're awesome",
                        ],
                        "rejected": [
                            "that is bad",
                            "that's so terrible",
                            "you're horrible",
                            "that's just awful",
                            "that is atrocious",
                            "just abhorrent",
                            "that is apalling",
                            "he's so abominable",
                            "that's dreadful",
                            "you're detestable",
                        ],
                    },
                )
                .shuffle(seed=42)
                .splits(train_size=0.7, validation_size=0.3)
            )

            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none"
            )

            reward_trainer = TrainHFRewardModel(
                "Reward Trainer",
                model_name="google/flan-t5-small",
                peft_config=peft_config,
            )
            reward_trainer.train_with_pairs(
                train_prompts=splits["train"].output["prompts"],
                train_chosen=splits["train"].output["chosen"],
                train_rejected=splits["train"].output["rejected"],
                validation_prompts=splits["validation"].output["prompts"],
                validation_chosen=splits["validation"].output["chosen"],
                validation_rejected=splits["validation"].output["rejected"],
                epochs=1,
                batch_size=8,
            )

            # Run PPO with TrainHFRewardModel reward model
            dataset = DataSource("Training Data", data={"prompts": ["Hi, "] * 6})
            val_dataset = dataset.take(2)
            trainer = TrainHFPPO("GPT-2 Trainer", model_name="gpt2")
            trainer.train(
                train_prompts=dataset.output["prompts"],
                validation_prompts=val_dataset.output["prompts"],
                reward_model=reward_trainer,
                epochs=1,
                batch_size=6,
                max_new_tokens=4,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "8566802139e49e11"
            reward_trainer.unload_model()
            trainer.unload_model()
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_ppo_causal_with_reward_trainer",
            #     private=True,
            # )

            # Run PPO with PreTrainedModel reward model
            sentiment_model = pipeline(
                "text-classification", model="lvwerra/distilbert-imdb"
            )
            dataset = DataSource("Training Data", data={"prompts": ["Hi, "] * 6})
            val_dataset = dataset.take(2)
            trainer = TrainHFPPO("GPT-2 Trainer", model_name="gpt2")
            trainer.train(
                train_prompts=dataset.output["prompts"],
                validation_prompts=val_dataset.output["prompts"],
                reward_model=sentiment_model.model,
                reward_model_tokenizer=sentiment_model.tokenizer,
                epochs=1,
                batch_size=6,
                max_new_tokens=4,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == (
                    "f6a02f718efd4892"
                    if sys.platform == "darwin"
                    else "163e313c78643f85"
                )
            # trainer.publish_to_hf_hub(
            #     repo_id="test_hf_ppo_causal_with_pretrained_model",
            #     private=True,
            # )


class TestTrainSetFitClassifier:
    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            trainer = TrainSetFitClassifier(
                "SetFit Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
            )
            assert isinstance(trainer.citation, list)
            assert len(trainer.citation) == 4

    def test_init(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["Bad", "Good", "Bad", "Good", "Bad"],
                    "outputs": [
                        "negative",
                        "positive",
                        "negative",
                        "positive",
                        "negative",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainSetFitClassifier(
                "SetFit Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
            )
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=18,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "38729463b5ee5721"
            assert train_result is trainer
            assert trainer.id2label == {0: "negative", 1: "positive"}
            assert trainer.label2id == {"negative": 0, "positive": 1}
            assert not trainer.is_multi_target
            assert type(get_orig_model(trainer.model)).__name__ == "SetFitModelWrapper"
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(os.path.join(trainer_path, "_checkpoints", "step_1"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "1_Pooling", "config.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model_head.pkl"))
            assert os.path.isfile(os.path.join(trainer.model_path, "tokenizer.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "SetFitModelWrapper"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "1_Pooling", "config.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "model_head.pkl"))
            assert os.path.isfile(os.path.join(export_path, "tokenizer.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_setfit_classifier",
            #     private=True,
            # )

    def test_multi_target(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["Zero", "One", "Two", "Three"],
                    "outputs": [[], ["A"], ["A", "B"], ["A", "B", "C"]],
                },
            )
            val_dataset = dataset

            trainer = TrainSetFitClassifier(
                "SetFit Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
            )
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=12,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "c3357f06e6f348fd"
            assert train_result is trainer
            assert trainer.id2label == {0: "A", 1: "B", 2: "C"}
            assert trainer.label2id == {"A": 0, "B": 1, "C": 2}
            assert trainer.is_multi_target
            assert type(get_orig_model(trainer.model)).__name__ == "SetFitModelWrapper"
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(os.path.join(trainer_path, "_checkpoints", "step_1"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "is_multi_target.json")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "1_Pooling", "config.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model_head.pkl"))
            assert os.path.isfile(os.path.join(trainer.model_path, "tokenizer.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "SetFitModelWrapper"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "1_Pooling", "config.json"))
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "model_head.pkl"))
            assert os.path.isfile(os.path.join(export_path, "tokenizer.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_setfit_classifier_multi_target",
            #     private=True,
            # )

    def test_peft(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["Bad", "Good", "Bad", "Good", "Bad"],
                    "outputs": [
                        "negative",
                        "positive",
                        "negative",
                        "positive",
                        "negative",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
            with ignore_transformers_warnings():
                from peft import LoraConfig

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["lin1", "lin2"],
            )

            trainer = TrainSetFitClassifier(
                "SetFit Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
                peft_config=peft_config,
            )
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=18,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "58c3cd6fe48822c5"
            assert train_result is trainer
            assert trainer.id2label == {0: "negative", 1: "positive"}
            assert trainer.label2id == {"negative": 0, "positive": 1}
            assert not trainer.is_multi_target
            assert type(get_orig_model(trainer.model)).__name__ == "SetFitModelWrapper"
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(os.path.join(trainer_path, "_checkpoints", "step_1"))
            assert os.path.isfile(
                os.path.join(
                    trainer_path, "_checkpoints", "step_1", "adapter_model.safetensors"
                )
            )
            assert os.path.isfile(
                os.path.join(
                    trainer_path, "_checkpoints", "step_1", "model.safetensors"
                )
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "label2id.json"))
            assert os.path.isfile(os.path.join(trainer.model_path, "id2label.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "1_Pooling", "config.json")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(trainer.model_path, "model_head.pkl"))
            assert os.path.isfile(os.path.join(trainer.model_path, "tokenizer.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "SetFitModelWrapper"
            assert os.path.isfile(os.path.join(export_path, "label2id.json"))
            assert os.path.isfile(os.path.join(export_path, "id2label.json"))
            assert os.path.isfile(os.path.join(export_path, "1_Pooling", "config.json"))
            assert not os.path.isfile(
                os.path.join(export_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(export_path, "training_args.json"))
            assert os.path.isfile(os.path.join(export_path, "README.md"))
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "model_head.pkl"))
            assert os.path.isfile(os.path.join(export_path, "tokenizer.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_setfit_classifier_peft_merged",
            #     private=True,
            # )


class TestTrainOpenAIFineTune:
    def setup_method(self, _):
        self.to_cleanup = []

    def teardown_method(self, _):
        for client, model_name in self.to_cleanup:
            try:
                client.models.delete(model_name)
            except (NotFoundError, PermissionDeniedError):
                pass

    @staticmethod
    def mock_finetune_job(trainer, mocker):  # noqa: C901
        client = trainer.client
        files: dict[str, Any] = {}
        ft_jobs: dict[str, Any] = {}

        # Basic Mocks
        def unload_model_remock(orig_unload_model):
            orig_unload_model()

            # Remock after unload_model
            TestTrainOpenAIFineTune.mock_finetune_job(trainer, mocker)

        mocker.patch.object(
            trainer, "unload_model", partial(unload_model_remock, trainer.unload_model)
        )
        mocker.patch.object(client.models, "retrieve", lambda *args, **kwargs: None)
        mocker.patch.object(client.files, "delete", lambda *args, **kwargs: None)

        # Simulation Mocks
        def create_file(file: tuple[str, Any], *args, **kwargs):
            nonlocal files
            file_obj: Any = None
            if "train" in file[0]:
                file_obj = SimpleNamespace(id="train_file_id")
            elif "validation" in file[0]:
                file_obj = SimpleNamespace(id="validation_file_id")
            files[file_obj.id] = file_obj
            return file_obj

        def retrieve_file(file_id: str, *args, **kwargs):
            if file_id in files:
                return files[file_id]
            else:
                raise NotFoundError(
                    response=SimpleNamespace(request=None, status_code=None),  # type:ignore[arg-type]
                    body=None,
                    message=None,  # type:ignore[arg-type]
                )

        mocker.patch.object(client.files, "create", create_file)
        mocker.patch.object(client.files, "retrieve", retrieve_file)

        def create_fine_tune_job(*args, **kwargs):
            nonlocal ft_jobs
            ft_job: Any = SimpleNamespace(
                id="ft_job_id", status="validating_files", fine_tuned_model=None
            )
            ft_jobs[ft_job.id] = ft_job
            return ft_job

        def retrive_fine_tune_job(fine_tuning_job_id: str, *args, **kwargs):
            if fine_tuning_job_id in ft_jobs:
                if ft_jobs[fine_tuning_job_id].status == "validating_files":
                    ft_jobs[fine_tuning_job_id].status = "queued"
                elif ft_jobs[fine_tuning_job_id].status == "queued":
                    ft_jobs[fine_tuning_job_id].status = "running"
                elif ft_jobs[fine_tuning_job_id].status == "running":
                    ft_jobs[fine_tuning_job_id].status = "succeeded"
                    ft_jobs[fine_tuning_job_id].fine_tuned_model = "ft:model"
                return ft_jobs[fine_tuning_job_id]
            else:
                raise NotFoundError(
                    response=SimpleNamespace(request=None, status_code=None),  # type:ignore[arg-type]
                    body=None,
                    message=None,  # type:ignore[arg-type]
                )

        def cancel_fine_tune_job(*args, **kwargs):
            raise BadRequestError(
                response=SimpleNamespace(request=None, status_code=None),  # type:ignore[arg-type]
                body=None,
                message="Fine-tune job already completed successfully.",
            )

        def list_events_fine_tune_job(*args, **kwargs):
            return SimpleNamespace(
                data=[
                    SimpleNamespace(
                        id="event-0",
                        level="info",
                        message="Fine-tune job created.",
                        created_at=0,
                    ),
                    SimpleNamespace(
                        id="event-1",
                        level="info",
                        message="Fine-tune job completed successfully.",
                        created_at=1,
                    ),
                ],
                has_more=False,
            )

        mocker.patch.object(client.fine_tuning.jobs, "create", create_fine_tune_job)
        mocker.patch.object(client.fine_tuning.jobs, "retrieve", retrive_fine_tune_job)
        mocker.patch.object(client.fine_tuning.jobs, "cancel", cancel_fine_tune_job)
        mocker.patch.object(
            client.fine_tuning.jobs, "list_events", list_events_fine_tune_job
        )

    def test_metadata(self, create_datadreamer):
        with create_datadreamer():
            llm = OpenAI("gpt-3.5-turbo-16k")
            trainer = TrainOpenAIFineTune(
                "GPT-3.5 Turbo 16K Trainer", model_name="gpt-3.5-turbo-16k"
            )
            assert trainer.base_model_card == llm.model_card
            assert trainer.license == llm.license
            assert trainer.citation == llm.citation
            assert trainer.system_prompt == llm.system_prompt

    def test_truncate_error(self, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": ["A founder of Microsoft is"],
                    "outputs": [" William Henry Gates (Bill Gates)" * 10000],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainOpenAIFineTune("Babbage Trainer", model_name="babbage-002")
            with pytest.raises(ValueError):
                trainer.train(
                    train_input=dataset.output["inputs"],
                    train_output=dataset.output["outputs"],
                    validation_input=val_dataset.output["inputs"],
                    validation_output=val_dataset.output["outputs"],
                    truncate=False,
                )
            trainer.unload_model()

    @pytest.mark.parametrize(
        "is_chat_model,model_name",
        [(False, "babbage-002"), (True, "gpt-3.5-turbo-1106")],
    )
    def test_finetune(self, is_chat_model, model_name, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data",
                data={
                    "inputs": [
                        "A founder of Microsoft is",
                        "A founder of Apple is",
                        "A founder of Y Combinator is",
                        "A founder of Twitter is",
                        "A founder of Tesla is",
                        "A founder of Pixar is",
                        "A founder of Facebook is",
                        "A founder of OpenAI is",
                        "A founder of Amazon is",
                        "A founder of Google is",
                    ],
                    "outputs": [
                        " William Henry Gates (Bill Gates)",
                        " Steve Jobs",
                        " Paul Graham",
                        " Jack Dorsey",
                        " Elon Musk",
                        " Ed Catmull",
                        " Mark Zuckerberg",
                        " Sam Altman",
                        " Jeff Bezos",
                        " Larry Page",
                    ],
                },
            )
            val_dataset = dataset.take(2)

            trainer = TrainOpenAIFineTune(
                f"{model_name} Trainer", model_name=model_name
            )
            TestTrainOpenAIFineTune.mock_finetune_job(trainer, mocker)
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=30,
                batch_size=8,
                learning_rate_multiplier=0.5,
            )
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) in ["235c4bf5832b89f9", "416ffefdbfa0e9af"]
            assert train_result is trainer
            self.to_cleanup.append((trainer.client, trainer.model))
            assert trainer.model.startswith("ft:")
            assert trainer.model_path == trainer.model
            assert os.path.isfile(os.path.join(trainer_path, "training_args.json"))
            assert os.path.isfile(os.path.join(trainer_path, "train.jsonl"))
            assert os.path.isfile(os.path.join(trainer_path, "validation.jsonl"))
            with cast(
                Reader,
                jsonlines.open(os.path.join(trainer_path, "train.jsonl"), mode="r"),
            ) as reader:
                assert next(iter(reader)) == (
                    {
                        "messages": [
                            {"role": "system", "content": trainer.system_prompt},
                            {"role": "user", "content": dataset.output["inputs"][0]},
                            {
                                "role": "assistant",
                                "content": dataset.output["outputs"][0],
                            },
                        ]
                    }
                    if is_chat_model
                    else {
                        "prompt": dataset.output["inputs"][0],
                        "completion": dataset.output["outputs"][0],
                    }
                )
            assert os.path.isfile(os.path.join(trainer_path, "train_file_id.json"))
            assert os.path.isfile(os.path.join(trainer_path, "validation_file_id.json"))
            assert os.path.isfile(os.path.join(trainer_path, "ft_job_id.json"))
            assert os.path.isfile(os.path.join(trainer_path, "trained_model_name.json"))
            assert "data_card" in trainer._model_card
