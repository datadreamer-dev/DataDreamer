import json
import os
from typing import cast

import pytest
import torch

from ... import DataDreamer
from ...llms import HFTransformers, ParallelLLM
from ...llms._chat_prompt_templates import CHAT_PROMPT_TEMPLATES, SYSTEM_PROMPTS
from ...steps import DataSource
from ...trainers import (
    TrainHFClassifier,
    TrainHFDPO,
    TrainHFFineTune,
    TrainHFPPO,
    TrainSentenceTransformer,
)
from ...trainers._train_hf_base import CustomDataCollatorWithPadding
from ...utils.arg_utils import AUTO
from ...utils.hf_model_utils import get_orig_model
from ...utils.import_utils import ignore_transformers_warnings

with ignore_transformers_warnings():
    from transformers import DataCollatorWithPadding, TrainerCallback, pipeline


class TestInferenceInMultiGPUEnvironment:
    __test__ = (
        torch.cuda.is_available() and torch.cuda.device_count() > 1
    )  # Runs on multi-GPU only

    def test_llm_single_gpu(self, create_datadreamer):
        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small")
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [["blue"] * 2, ["green"] * 2]
            assert llm.model.device == torch.device("cpu")

        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small", device=0)
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [["blue"] * 2, ["green"] * 2]
            assert llm.model.device == torch.device(0)

    def test_llm_multi_gpu(self, create_datadreamer):
        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small", device=0)
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [["blue"] * 2, ["green"] * 2]
            assert llm.model.device == torch.device(0)

        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small", device=1)
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [["blue"] * 2, ["green"] * 2]
            assert llm.model.device == torch.device(1)

    def test_device_map_auto(self, create_datadreamer, mocker):
        with create_datadreamer():
            llm = HFTransformers("google/flan-t5-small", device_map="auto")
            generated_texts = llm.run(
                [
                    "Question: What color is the sky?\nAnswer:",
                    "Question: What color are trees?\nAnswer:",
                ],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                stop="Question:",
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert generated_texts == [["blue"] * 2, ["green"] * 2]
            assert llm.model.device == torch.device(0)

    def test_parallel_llm(self, create_datadreamer, mocker):
        prompt_1 = "This is a long prompt."
        prompt_2 = "Short prompt."

        with create_datadreamer():
            llm_1 = HFTransformers("gpt2", device=0)
            llm_2 = HFTransformers("gpt2", device=1)
            llm_1_spy = mocker.spy(llm_1, "_run_batch")
            llm_2_spy = mocker.spy(llm_2, "_run_batch")
            parallel_llm = ParallelLLM(llm_1, llm_2)
            parallel_llm.run(
                [prompt_1, prompt_2],
                max_new_tokens=25,
                temperature=0.0,
                top_p=0.0,
                n=2,
                repetition_penalty=None,
                logit_bias=None,
                batch_size=2,
            )
            assert llm_1.model.device == torch.device(0)
            assert llm_2.model.device == torch.device(1)
            assert llm_1_spy.call_count == 1
            assert llm_2_spy.call_count == 1


class TestTrainInMultiGPUEnvironment:
    __test__ = (
        torch.cuda.is_available() and torch.cuda.device_count() > 1
    )  # Runs on multi-GPU only

    def test_single_gpu_train(self, create_datadreamer, mocker):
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

            trainer = TrainHFClassifier(
                "T5 Trainer", model_name="google/flan-t5-small", device=1
            )
            trained_on_device = None
            orig_save_model = trainer._save_model

            def _save_model_spy(*args, **kwargs):
                nonlocal trained_on_device
                trained_on_device = kwargs["model"].device
                return orig_save_model(*args, **kwargs)

            trainer._save_model = _save_model_spy  # type:ignore[method-assign]
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert trainer.model.device == torch.device(1)
            assert trained_on_device == torch.device(1)
            assert (
                "ACCELERATE_TORCH_DEVICE" not in os.environ
            )  # Make sure this is cleared out


class TestTrainDistributed:
    __test__ = (
        torch.cuda.is_available() and torch.cuda.device_count() > 1
    )  # Runs on multi-GPU only

    def test_ddp(self, create_datadreamer, mocker):
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

            trainer = TrainHFFineTune(
                "GPT-2 Trainer", model_name="gpt2", device=[0, 1], fsdp=False
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 0
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "a00a3d19ff73150b"
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
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_ddp",
            #     private=True,
            # )

    def test_fsdp(self, create_datadreamer, mocker):
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

            trainer = TrainHFFineTune("GPT-2 Trainer", model_name="gpt2", device=[0, 1])
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 0
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "ab2126ca62265bc6"
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
            assert os.path.isfile(os.path.join(trainer.model_path, "pytorch_model.bin"))
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2LMHeadModel"
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_fsdp",
            #     private=True,
            # )

    def test_fsdp_peft(self, create_datadreamer, mocker):
        # TODO (fix later if transformers updates)
        # See: https://github.com/huggingface/transformers/pull/28297
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

            trainer = TrainHFFineTune(
                "GPT-2 Trainer",
                model_name="gpt2",
                peft_config=peft_config,
                device=[0, 1],
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 0
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "6b385aca0ce684b3"
            assert train_result is trainer
            assert (
                type(get_orig_model(trainer.model)).__name__ == "PeftModelForCausalLM"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isdir(
                os.path.join(trainer_path, "_checkpoints", "checkpoint-1")
            )
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            assert os.path.isfile(
                os.path.join(trainer.model_path, "adapter_model.safetensors")
            )
            for fn in [
                "pytorch_model.bin",
                "pytorch_model_fsdp.bin",
                "model.safetensors",
            ]:
                assert not os.path.isfile(os.path.join(trainer.model_path, fn))
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path, adapter_only=True)
            assert type(export_result).__name__ == "PeftModelForCausalLM"
            assert os.path.isfile(
                os.path.join(export_path, "adapter_model.safetensors")
            )
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))

    def test_fsdp_seq2seq(self, create_datadreamer, mocker):
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

            trainer = TrainHFFineTune(
                "T5 Trainer", model_name="google/flan-t5-small", device=[0, 1]
            )
            data_collator_spy = mocker.spy(DataCollatorWithPadding, "__call__")
            train_result = trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=val_dataset.output["inputs"],
                validation_output=val_dataset.output["outputs"],
                epochs=1,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 0
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "83080fe9c469f55b"
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
            assert os.path.isfile(os.path.join(trainer.model_path, "pytorch_model.bin"))
            assert os.path.isfile(os.path.join(trainer.model_path, "spiece.model"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "T5ForConditionalGeneration"
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "spiece.model"))

    @pytest.mark.parametrize(
        "fsdp,precompute_ref_log_probs",
        [(AUTO, False), (AUTO, True), (False, False), (False, True)],
    )
    def test_distributed_dpo(
        self, fsdp, precompute_ref_log_probs, create_datadreamer, mocker
    ):
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
                device=[0, 1],
                fsdp=fsdp,
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
                precompute_ref_log_probs=precompute_ref_log_probs,  # We test precompute_ref_log_probs here
            )
            assert data_collator_spy.call_count == 0
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == (
                    "c2d9ce32647299ba" if fsdp else "8560d9ebb0c234f7"
                )
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
            assert os.path.isfile(
                os.path.join(
                    trainer.model_path,
                    "pytorch_model.bin" if fsdp else "model.safetensors",
                )
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2LMHeadModel"
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id=f"test_distributed_dpo_{fsdp}_{precompute_ref_log_probs}",
            #     private=True,
            # )

    @pytest.mark.parametrize("fsdp", [False])
    def test_distributed_ppo(self, fsdp, create_datadreamer, mocker):
        with create_datadreamer():
            dataset = DataSource(
                "Training Data", data={"prompts": ["Hi hi"] + ["Hi"] * 11}
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
                "GPT-2 Trainer",
                model_name="gpt2",
                chat_prompt_template="{{prompt}}, ",
                device=[0, 1],
                fsdp=fsdp,
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
            assert data_collator_spy.call_count == 0
            trainer_path = cast(str, trainer._output_folder_path)
            with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
                assert json.load(f) == "ca7ef3a2f7fc5946"
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
            assert os.path.isfile(
                os.path.join(
                    trainer.model_path,
                    "pytorch_model.bin" if fsdp is AUTO else "model.safetensors",
                )
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "vocab.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(path=export_path)
            assert type(export_result).__name__ == "GPT2LMHeadModel"
            assert os.path.isfile(os.path.join(export_path, "model.safetensors"))
            assert os.path.isfile(os.path.join(export_path, "vocab.json"))
            # trainer.publish_to_hf_hub(
            #     repo_id="test_distributed_ppo_{fsdp}",
            #     private=True,
            # )


class TestTrainDistributedSlow:
    """
    These tests run slower, but also check if training / saving weights worked.
    If these tests pass, it's good sign the DDP / FSDP is working as expected.
    """

    __test__ = (
        torch.cuda.is_available() and torch.cuda.device_count() > 1
    )  # Runs on multi-GPU only

    @pytest.mark.parametrize(
        "fsdp,peft", [(AUTO, False), (AUTO, True), (False, False), (False, True)]
    )
    def test_distributed_sentence_transformer(
        self, fsdp, peft, create_datadreamer, mocker
    ):
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

            if peft:
                peft_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    bias="none",
                    target_modules=["lin1", "lin2"],
                )
            else:
                peft_config = None

            trainer = TrainSentenceTransformer(
                "Distilbert NLI Trainer",
                model_name="sentence-transformers/distilbert-base-nli-mean-tokens",
                peft_config=peft_config,
                device=[0, 1],
                fsdp=fsdp,
            )
            data_collator_spy = mocker.spy(CustomDataCollatorWithPadding, "__call__")
            train_result = trainer.train_with_triplets(
                train_anchors=dataset.output["anchors"],
                train_positives=dataset.output["positives"],
                train_negatives=dataset.output["negatives"],
                validation_anchors=val_dataset.output["anchors"],
                validation_positives=val_dataset.output["positives"],
                validation_negatives=val_dataset.output["negatives"],
                epochs=5,
                batch_size=8,
            )
            assert data_collator_spy.call_count == 0
            trainer_path = cast(str, trainer._output_folder_path)
            # with open(os.path.join(trainer_path, "fingerprint.json"), "r") as f:
            #     assert json.load(f) in [
            #         "08d8424bb253f64e",
            #         "ec0de2f430120ba2",
            #         "4fc7804c20996e59",
            #         "128feaf040658cfe",
            #     ]
            assert train_result is trainer
            assert type(get_orig_model(trainer.model)).__name__ == (
                "PeftModelForFeatureExtraction" if peft else "SentenceTransformer"
            )
            assert trainer.model_path == os.path.join(trainer_path, "_model")
            assert os.path.isfile(
                os.path.join(trainer.model_path, "training_args.json")
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "seed.json"))
            if not peft:
                assert os.path.isfile(
                    os.path.join(trainer.model_path, "1_Pooling", "config.json")
                )
            assert os.path.isfile(
                os.path.join(
                    trainer.model_path,
                    "adapter_model.safetensors"
                    if peft
                    else ("pytorch_model.bin" if fsdp is AUTO else "model.safetensors"),
                )
            )
            assert os.path.isfile(os.path.join(trainer.model_path, "tokenizer.json"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(
                path=export_path, adapter_only=True if peft else False
            )
            assert (
                type(export_result).__name__ == "PeftModelForFeatureExtraction"
                if peft
                else "SentenceTransformer"
            )
            if not peft:
                assert os.path.isfile(
                    os.path.join(export_path, "1_Pooling", "config.json")
                )
            assert os.path.isfile(
                os.path.join(
                    export_path,
                    "adapter_model.safetensors" if peft else "model.safetensors",
                )
            )
            assert os.path.isfile(os.path.join(export_path, "tokenizer.json"))

            # Test the model performance
            # (ensures weights were saved properly)
            from sentence_transformers.util import cos_sim

            inpt = export_result.encode("Base")
            others = export_result.encode(["Wonderful", "Bad"])
            sims = cos_sim(inpt, others)[0]
            assert (sims[0] - sims[1]) > 0.5

            # trainer.publish_to_hf_hub(
            #     repo_id=f"test_distributed_sentence_transformer_{fsdp}_{peft}",
            #     private=True,
            # )

    @pytest.mark.parametrize(
        "fsdp,peft", [(AUTO, False), (AUTO, True), (False, False), (False, True)]
    )
    def test_distributed_resume(self, fsdp, peft, create_datadreamer, mocker):
        exit_on_epoch: None | int = None
        data = {
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
        }

        class ExitCallback(TrainerCallback):
            def on_epoch_begin(self, args, state, control, **kwargs):
                if state.epoch == exit_on_epoch:
                    raise ReferenceError("CustomError")

        # A warning is thrown if not run on GPU by bitsandbytes imported by PEFT
        with ignore_transformers_warnings():
            from peft import LoraConfig

        if peft:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["c_proj"],
                fan_in_fan_out=True,
            )
        else:
            peft_config = None

        with create_datadreamer():
            dataset = DataSource("Training Data", data=data)
            trainer = TrainHFFineTune(
                "GPT-2 Trainer",
                model_name="gpt2",
                peft_config=peft_config,
                device=[0, 1],
                fsdp=fsdp,
            )
            exit_on_epoch = 9
            with pytest.raises(Exception, match=r".*ReferenceError.*CustomError.*"):
                trainer.train(
                    train_input=dataset.output["inputs"],
                    train_output=dataset.output["outputs"],
                    validation_input=dataset.output["inputs"],
                    validation_output=dataset.output["outputs"],
                    epochs=10,
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
            trainer = TrainHFFineTune(
                "GPT-2 Trainer",
                model_name="gpt2",
                peft_config=peft_config,
                device=[0, 1],
                fsdp=fsdp,
            )
            exit_on_epoch = None
            trainer.train(
                train_input=dataset.output["inputs"],
                train_output=dataset.output["outputs"],
                validation_input=dataset.output["inputs"],
                validation_output=dataset.output["outputs"],
                epochs=10,
                batch_size=8,
                callbacks=[ExitCallback],
            )
            assert trainer._resumed
            assert trainer.seed == 42
            trainer_path = cast(str, trainer._output_folder_path)
            assert os.path.isfile(os.path.join(trainer_path, "fingerprint.json"))
            assert os.path.isdir(os.path.join(trainer_path, "_model"))
            export_path = os.path.join(trainer_path, "export")
            export_result = trainer.export_to_disk(
                path=export_path, adapter_only=True if peft else False
            )
            assert (
                type(export_result).__name__ == "PeftModelForCausalLM"
                if peft
                else "GPT2LMHeadModel"
            )

            # Test the model performance
            # (ensures weights were saved properly)
            inputs = trainer.tokenizer(
                ["A founder of Microsft is"], padding=True, return_tensors="pt"
            )
            outputs = trainer.tokenizer.batch_decode(
                export_result.generate(**inputs, max_new_tokens=4, do_sample=False)
            )
            assert ("Bill Gates" in outputs[0] and "<|endoftext|>" in outputs[0]) or (
                "William Henry Gates" in outputs[0]
            )

            # trainer.publish_to_hf_hub(
            #     repo_id="test_fsdp_peft",
            #     private=True,
            # )

            # trainer.publish_to_hf_hub(
            #     repo_id=f"test_distributed_resume_{fsdp}_{peft}",
            #     private=True,
            # )
