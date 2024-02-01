import os
import pickle
from copy import deepcopy
from functools import cached_property
from logging import Logger
from time import time
from typing import TYPE_CHECKING, Any, Callable, Union

import torch
from datasets import IterableDataset
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import logging as transformers_logging

from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..llms.llm import _check_temperature_and_top_p
from ..utils.arg_utils import AUTO, Default, default_to
from ..utils.fs_utils import mkdir
from ..utils.hf_model_utils import is_peft_model
from ..utils.import_utils import ignore_transformers_warnings, ignore_trl_warnings
from ._train_hf_base import (
    CustomDataCollatorWithPadding,
    TrainingArguments,
    _prepare_inputs_and_outputs,
    _start_hf_trainer,
    get_logging_callback,
)
from .train_hf_finetune import TrainHFFineTune
from .train_hf_reward_model import TrainHFRewardModel

with ignore_transformers_warnings():
    from transformers import (
        EarlyStoppingCallback,
        PreTrainedModel,
        PreTrainedTokenizer,
        pipeline,
    )
    from transformers.trainer_callback import PrinterCallback
    from transformers.utils.quantization_config import QuantizationConfigMixin

if TYPE_CHECKING:  # pragma: no cover
    # Two warnings we can't silence are thrown by peft at import-time so
    # we import this library only when needed
    with ignore_transformers_warnings():
        from peft import PeftModel


def get_ppo_trainer(  # noqa: C901
    logger: Logger,
    batch_size: int,
    learning_rate: float,
    ppo_trainer: Any,
    generation_kwargs: dict[str, Any],
    length_sampler: None | Callable,
    reward_model: Union[
        TrainHFRewardModel,
        PreTrainedModel,
        "PeftModel",
        Callable[[list[str]], list[float]],
    ],
    reward_model_tokenizer: None | PreTrainedTokenizer,
    reward_model_is_peft: bool,
    max_kl: None | float,
):
    with ignore_transformers_warnings():
        from transformers import Trainer

    pipe: Any = None

    @torch.no_grad()
    def compute_rewards(prompts: list[str]) -> list[float]:
        nonlocal reward_model, reward_model_tokenizer, reward_model_is_peft, pipe
        if isinstance(reward_model, TrainHFRewardModel):
            reward_model_is_peft = reward_model.peft_config is not None
            reward_model_tokenizer = deepcopy(reward_model.tokenizer)
            reward_model = reward_model.model
        if reward_model_is_peft or isinstance(reward_model, PreTrainedModel):
            if pipe is None:
                transformers_logging_verbosity = transformers_logging.get_verbosity()
                transformers_logging.set_verbosity(transformers_logging.CRITICAL)
                pipeline_optional_kwargs = (
                    {}
                    if getattr(reward_model, "hf_device_map", None) is not None
                    else {"device": reward_model.device}
                )
                pipe = pipeline(
                    "text-classification",
                    model=reward_model,
                    tokenizer=reward_model_tokenizer,
                    function_to_apply="none",
                    **pipeline_optional_kwargs,
                )
                transformers_logging.set_verbosity(transformers_logging_verbosity)
            return [
                reward_result["score"]
                for reward_result in pipe(prompts, batch_size=len(prompts))
            ]
        else:
            return reward_model(prompts)

    class PPOTrainerWrapper(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            def compute_metrics(eval_pred):
                preds, _ = eval_pred
                mean_preds = preds.mean(axis=0)
                return {"rewards": mean_preds}

            self.compute_metrics = compute_metrics

            # Setup for FSDP
            self.accelerator.prepare_model = lambda model, *args, **kwargs: model
            self.accelerator.prepare_optimizer = (
                lambda optimizer, *args, **kwargs: optimizer
            )

        def get_train_dataloader(self) -> DataLoader:
            # PPOTrainer's .step() method does not allow smaller than batch size inputs
            orig_dataloader_drop_last = self.args.dataloader_drop_last
            self.args.dataloader_drop_last = True
            dataloader = super().get_train_dataloader()
            self.args.dataloader_drop_last = orig_dataloader_drop_last
            return dataloader

        def training_step(
            self, model: torch.nn.Module, inputs: dict[str, torch.Tensor | Any]
        ) -> torch.Tensor:
            # Generate from model
            query_tensors = list(inputs["input_ids"])

            response_tensors = ppo_trainer.generate(
                query_tensors,
                **generation_kwargs,
                batch_size=len(query_tensors),
                return_prompt=False,
                length_sampler=length_sampler,
            )
            inputs["prompts"] = self.tokenizer.batch_decode(
                query_tensors,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            inputs["response"] = self.tokenizer.batch_decode(
                response_tensors,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Compute rewards
            texts = [q + r for q, r in zip(inputs["prompts"], inputs["response"])]
            rewards = [torch.tensor(score) for score in compute_rewards(texts)]

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            self.last_ppo_logs = {
                "train_rewards": torch.tensor(rewards).mean().item(),
                "train_value_estimation_loss": stats["ppo/loss/value"],
                "train_policy_alignment_loss": stats["ppo/loss/policy"],
                "train_total_scaled_rewards": stats["ppo/mean_scores"]
                + stats["ppo/mean_non_score_reward"],
                "train_scaled_rewards": stats["ppo/mean_scores"],
                "train_kl_scaled_rewards_penalty": stats["ppo/mean_non_score_reward"],
                "train_kl": stats["objective/kl"],
                "train_avg_ratio": stats["ppo/policy/ratio"].mean().item(),
                "train_clipfrac": stats["ppo/policy/clipfrac"],
                "train_entropy": stats["objective/entropy"],
                "train_kl_coef": stats["objective/kl_coef"],
            }
            if (
                max_kl is not None and self.last_ppo_logs["train_kl"] > max_kl
            ):  # pragma: no cover
                logger.info(
                    f"Stopping training due to `kl` > `max_kl`"
                    f" ({self.last_ppo_logs['train_kl']} > {max_kl}). Set `max_kl`"
                    " = None if you want to continue training."
                )
                self.control.should_training_stop = True
            return torch.tensor(stats["ppo/loss/total"])  # Return dummy loss (not used)

        def log(self, logs: dict[str, float]) -> None:
            is_training_log = (
                not any([k.startswith("eval") for k in logs.keys()])
                and "train_runtime" not in logs
            )
            if is_training_log:
                orig_logs = logs
                logs = getattr(self, "last_ppo_logs", {})
                logs.update(orig_logs)
            super().log(logs)

        def create_optimizer(self):
            class DummyOptimizer(Optimizer):
                def __init__(self):
                    self.param_groups = []
                    self.state = {}  # type:ignore[assignment]

                def __getstate__(self):  # pragma: no cover
                    pass

                def __setstate__(self, state):  # pragma: no cover
                    pass

                def state_dict(self) -> dict[Any, Any]:  # pragma: no cover
                    return ppo_trainer.optimizer.state_dict()

                def load_state_dict(
                    self, state_dict: dict[Any, Any]
                ):  # pragma: no cover
                    return ppo_trainer.optimizer.load_state_dict(state_dict)

                def step(self, *args, **kwargs):
                    pass

            self.optimizer = DummyOptimizer()

        def create_scheduler(
            self,
            num_training_steps: int,
            optimizer: None | torch.optim.Optimizer = None,
        ):
            class DummyLRScheduler(LRScheduler):
                def __init__(self):
                    pass

                def __getstate__(self):  # pragma: no cover
                    pass

                def __setstate__(self, state):  # pragma: no cover
                    pass

                def state_dict(self) -> dict[Any, Any]:  # pragma: no cover
                    if ppo_trainer.lr_scheduler is not None:
                        return ppo_trainer.lr_scheduler.state_dict()
                    else:
                        return {}

                def load_state_dict(
                    self, state_dict: dict[Any, Any]
                ):  # pragma: no cover
                    if ppo_trainer.lr_scheduler is not None:
                        return ppo_trainer.lr_scheduler.load_state_dict(state_dict)

                def step(self, *args, **kwargs):  # pragma: no cover
                    pass

                def get_last_lr(self):  # pragma: no cover
                    if ppo_trainer.lr_scheduler is not None:
                        return ppo_trainer.lr_scheduler.get_last_lr()
                    else:
                        return [ppo_trainer.optimizer.param_groups[0]["lr"]]

            self._created_lr_scheduler = True
            self.lr_scheduler = DummyLRScheduler()

        @torch.no_grad()
        def prediction_step(
            self,
            model: torch.nn.Module,
            inputs: dict[str, torch.Tensor | Any],
            prediction_loss_only: bool,
            ignore_keys: None | list[str] = None,
        ) -> tuple[None | torch.Tensor, None | torch.Tensor, None | torch.Tensor]:
            # Generate from model
            deterministic_generation_kwargs = generation_kwargs.copy()
            deterministic_generation_kwargs.update({"do_sample": False})
            for sample_kwarg in [
                "temperature",
                "top_p",
                "typical_p",
                "top_k",
                "epsilon_cutoff",
                "eta_cutoff",
            ]:
                deterministic_generation_kwargs.pop(sample_kwarg, None)
            query_tensors = list(inputs["input_ids"])
            response_tensors = ppo_trainer.generate(
                query_tensors,
                **deterministic_generation_kwargs,
                batch_size=inputs["input_ids"].shape[0],
                return_prompt=False,
                length_sampler=length_sampler,
            )
            inputs["prompts"] = self.tokenizer.batch_decode(
                query_tensors,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            inputs["response"] = self.tokenizer.batch_decode(
                response_tensors,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Compute rewards
            texts = [q + r for q, r in zip(inputs["prompts"], inputs["response"])]
            rewards = compute_rewards(texts)

            # Update progress
            self.eval_dataloader_size = getattr(
                self, "eval_dataloader_size", len(self.get_eval_dataloader())
            )
            self.prediction_progress_epoch = getattr(
                self, "prediction_progress_epoch", -1.0
            )
            current_epoch = round(self.state.epoch, 2) if self.state.epoch else 0.0
            if self.prediction_progress_epoch != current_epoch:
                self.prediction_progress_epoch = current_epoch
                self.prediction_progress_batch_idx = 0
                self.prediction_progress_last = 0.0
            else:  # pragma: no cover
                self.prediction_progress_batch_idx += 1
            progress = (
                self.prediction_progress_batch_idx + 1
            ) / self.eval_dataloader_size
            progress_int = int(progress * 100)
            if (time() - self.prediction_progress_last) > 120:
                self.prediction_progress_last = time()
                self.log(
                    {
                        "eval_epoch": current_epoch,
                        "eval_loss": 0.0,
                        "eval_progress": f"{progress_int}%",  # type:ignore[dict-item]
                    }
                )

            return (
                torch.tensor(0.0).to(model.device),
                torch.tensor(rewards).to(model.device),
                torch.tensor(0.0).to(model.device),
            )

        def _save_optimizer_and_scheduler(self, output_dir):
            super()._save_optimizer_and_scheduler(output_dir)
            if output_dir is None:  # pragma: no cover
                return
            mkdir(output_dir)
            with open(os.path.join(output_dir, "ppo_kl_ctl.pkl"), "wb+") as f:
                pickle.dump(ppo_trainer.kl_ctl.__dict__, f)
            with open(os.path.join(output_dir, "ppo_running.pkl"), "wb+") as f:
                running_dict = ppo_trainer.running.__dict__.copy()
                del running_dict["accelerator"]
                pickle.dump(running_dict, f)

        def _load_optimizer_and_scheduler(self, output_dir):
            super()._load_optimizer_and_scheduler(output_dir)
            if output_dir is None:  # pragma: no cover
                return
            with open(os.path.join(output_dir, "ppo_kl_ctl.pkl"), "rb") as f:
                ppo_trainer.kl_ctl.__dict__.update(pickle.load(f))
            with open(os.path.join(output_dir, "ppo_running.pkl"), "rb") as f:
                ppo_trainer.running.__dict__.update(pickle.load(f))

    return PPOTrainerWrapper


class TrainHFPPO(TrainHFFineTune):
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
        if isinstance(device, list) and fsdp is not False:  # pragma: no cover
            # TODO (fix later if TRL/transformers/PyTorch updates):
            # This is a larger side-effect of how FSDP is implemented in PyTorch.
            #
            # https://github.com/pytorch/pytorch/issues/100069
            # https://github.com/pytorch/pytorch/issues/103682
            raise RuntimeError(
                "TrainHFPPO does not support multiple devices with FSDP. Please"
                " explicitly pass TrainHFPPO(..., fsdp=False) to train on multiple"
                " devices."
            )

    def _train(  # type:ignore[override]  # noqa: C901
        self,
        train_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        reward_model: Union[
            TrainHFRewardModel,
            PreTrainedModel,
            "PeftModel",
            Callable[[list[str]], list[float]],
        ],
        reward_model_tokenizer: None | PreTrainedTokenizer = None,
        max_new_tokens: None | int = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generation_kwargs: None | dict[str, Any] = None,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        optimizer: Optimizer | Default = AUTO,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        lr_scheduler: None | LRScheduler = None,
        seed: int = 42,
        length_sampler: None | Callable = None,
        init_kl_coef: float = 0.2,
        adap_kl_ctrl: bool = True,
        adap_kl_target: float = 6.0,
        max_kl: None | float | Default = AUTO,
        **kwargs,
    ):
        with ignore_trl_warnings():
            from trl import (
                AutoModelForCausalLMWithValueHead,
                AutoModelForSeq2SeqLMWithValueHead,
                PPOConfig,
                PPOTrainer,
            )

        # Validate arguments
        reward_model_is_peft = is_peft_model(reward_model)
        assert (
            not isinstance(reward_model, PreTrainedModel) and not reward_model_is_peft
        ) or reward_model_tokenizer, (
            "You must specify `reward_model_tokenizer` if passing a"
            " PreTrainedModel as a `reward_model`."
        )
        assert not reward_model_tokenizer or (
            isinstance(reward_model, PreTrainedModel) or reward_model_is_peft
        ), (
            "You should only specify `reward_model_tokenizer` if passing a"
            " PreTrainedModel as a `reward_model`."
        )
        for kwarg in [
            "num_shared_layers",
            "eval_accumulation_steps",
            "lr_scheduler_type",
            "use_cpu",
            "jit_mode_eval",
            "use_ipex",
            "bf16",
            "fp16",
            "fsdp",
            "fsdp_config",
            "deepspeed",
            "optim",
            "gradient_checkpointing",
            "use_mps_device",
            "torch_compile",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "max_grad_norm",
            "warmup_ratio",
            "warmup_steps",
            "jit_mode_eval",
            "include_inputs_for_metrics",
            "include_tokens_per_second",
            "neftune_noise_alpha",
        ]:
            assert kwarg not in kwargs, f"`{kwarg}` is not supported."
        for kwarg in [
            "steps",
            "early_stopping",
            "target_kl",
            "compare_steps",
            "push_to_hub_if_best_kwargs",
        ]:
            assert kwarg not in kwargs, (
                f"`{kwarg}` is not supported. Use the equivalent from"
                " `TrainingArguments` instead."
            )

        # Prepare datasets
        assert (
            self._is_encoder_decoder or truncate
        ), "`truncate=False` is not supported for this model."
        train_dataset, validation_dataset, _, _ = _prepare_inputs_and_outputs(
            self,
            train_columns={("input_ids", "Train Input"): train_prompts},
            validation_columns={("input_ids", "Validation Input"): validation_prompts},
            truncate=truncate,
        )

        # Prepare data collator
        left_tokenizer = self.__class__.tokenizer.func(self)  # type: ignore[attr-defined]
        left_tokenizer.padding_side = "left"
        data_collator = kwargs.pop(
            "data_collator", None
        ) or CustomDataCollatorWithPadding(
            tokenizer=self.tokenizer,
            fields_to_pad=[
                {
                    "name": "input_ids",
                    "output_name": "input_ids",
                    "output_attention_mask_name": "attention_mask",
                    "tokenizer": left_tokenizer
                    if not self._is_encoder_decoder
                    else self.tokenizer,
                }
            ],
        )

        # Prepare compute metrics
        compute_metrics = kwargs.pop("compute_metrics", None)

        # Prepare callbacks
        callbacks = [get_logging_callback(self, log_loss=False)]
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

        # Prepare preprocess_logits_for_metrics
        preprocess_logits_for_metrics = None

        # Prepare model and reference model
        self.seed = seed
        model = self._create_model()
        if self.peft_config:
            # PPOTrainer will automatically use the model with the adapters disabled
            # as the reference model
            ref_model = None
        else:
            ref_model = self._create_model(is_ref_model=True)

        # Prepare training arguments
        other_training_args = {
            kwarg: kwargs.pop(kwarg)
            for kwarg in [
                "eval_delay",
                "max_steps",
                "log_level",
                "log_level_replica",
                "log_on_each_node",
                "logging_dir",
                "logging_first_step",
                "logging_nan_inf_filter",
                "save_steps",
                "save_total_limit",
                "data_seed",
                "dataloader_drop_last",
                "eval_steps",
                "dataloader_num_workers",
                "past_index",
                "run_name",
                "deepspeed",
                "label_smoothing_factor",
                "debug",
                "group_by_length",
                "length_column_name",
                "dataloader_pin_memory",
                "skip_memory_metrics",
                "push_to_hub",
                "resume_from_checkpoint",
                "hub_model_id",
                "hub_strategy",
                "hub_token",
                "hub_private_repo",
                "hub_always_push",
                "auto_find_batch_size",
                "full_determinism",
                "split_batches",
                "dispatch_batches",
            ]
            if kwarg in kwargs
        }
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
            optim="adamw_torch",
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            logging_strategy=kwargs.pop("logging_strategy", None) or "steps",
            logging_steps=kwargs.pop("logging_steps", 1),
            evaluation_strategy=kwargs.pop("evaluation_strategy", None) or "epoch",
            save_strategy=kwargs.pop("save_strategy", None) or "epoch",
            save_total_limit=kwargs.pop("save_total_limit", 1),
            save_safetensors=True,
            metric_for_best_model=kwargs.pop("metric_for_best_model", None)
            or "eval_rewards",
            greater_is_better=kwargs.pop("greater_is_better", True),
            load_best_model_at_end=kwargs.pop("load_best_model_at_end", True),
            seed=seed,
            **other_training_args,
        )
        if adap_kl_ctrl:
            if "horizon" in kwargs:  # pragma: no cover
                horizon = kwargs.pop("horizon")
            else:
                assert not isinstance(train_dataset, IterableDataset), (
                    "The train input columns must be of known length. Use `total_num_rows` if"
                    " using iterable datasets."
                )
                horizon = len(train_dataset) * epochs
        else:  # pragma: no cover
            horizon = 10000
        ppo_config = PPOConfig(
            tracker_project_name=f"DataDreamer - {self.name}",
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_encoder_decoder=self._is_encoder_decoder,
            is_peft_model=(self.peft_config is not None),
            seed=seed,
            optimize_device_cache=kwargs.pop(
                "optimize_device_cache", kwargs.pop("optimize_cuda_cache", False)
            ),
            ppo_epochs=kwargs.pop("ppo_epochs", 4),
            mini_batch_size=kwargs.pop("mini_batch_size", batch_size),
            use_score_scaling=kwargs.pop("use_score_scaling", True),
            init_kl_coef=init_kl_coef,
            adap_kl_ctrl=adap_kl_ctrl,
            target=adap_kl_target,
            horizon=horizon,
            **kwargs,
        )
        max_kl = default_to(max_kl, adap_kl_target * 1.5 if adap_kl_ctrl else None)
        assert not isinstance(max_kl, Default)
        temperature, top_p = _check_temperature_and_top_p(
            temperature=temperature, top_p=top_p, supports_zero_temperature=False
        )
        final_generation_kwargs: dict[str, Any] = {
            "top_k": 50,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
        }
        if length_sampler is None and (
            generation_kwargs is None
            or (
                "max_length" not in generation_kwargs
                and "max_new_tokens" not in generation_kwargs
            )
        ):
            if max_new_tokens is None:  # pragma: no cover
                final_generation_kwargs["max_length"] = self.tokenizer.model_max_length
            else:  # pragma: no cover
                final_generation_kwargs["max_new_tokens"] = max_new_tokens
        else:  # pragma: no cover
            assert max_new_tokens is None, (
                "You cannot specify `max_new_tokens` if also specifying a max"
                " generation length through `generation_kwargs` or `length_sampler`."
            )

        final_generation_kwargs.update(generation_kwargs or {})

        # Add value head to model
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # pragma: no cover
                torch.cuda.manual_seed_all(seed)
        if self._is_encoder_decoder:
            model_with_value_head = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
                model
            )
        else:
            model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(
                model
            )

        # Add value head to reference model
        if ref_model is not None:
            if seed:
                torch.manual_seed(seed)
                if torch.cuda.is_available():  # pragma: no cover
                    torch.cuda.manual_seed_all(seed)
            if self._is_encoder_decoder:
                ref_model_with_value_head = (
                    AutoModelForSeq2SeqLMWithValueHead.from_pretrained(ref_model)
                )
            else:
                ref_model_with_value_head = (
                    AutoModelForCausalLMWithValueHead.from_pretrained(ref_model)
                )
        else:
            ref_model_with_value_head = None

        # Prepare optimizer
        optimizer = default_to(
            optimizer,
            AdamW(
                filter(lambda p: p.requires_grad, model_with_value_head.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay,
            ),
        )

        # Setup trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model_with_value_head,
            ref_model=ref_model_with_value_head,
            tokenizer=self.__class__.tokenizer.func(self),  # type: ignore[attr-defined]
            dataset=train_dataset,
            data_collator=data_collator,
            num_shared_layers=None,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        trainer = get_ppo_trainer(
            logger=self.logger,
            batch_size=batch_size,
            learning_rate=learning_rate,
            ppo_trainer=ppo_trainer,
            generation_kwargs=final_generation_kwargs,
            length_sampler=length_sampler,
            reward_model=reward_model,
            reward_model_tokenizer=reward_model_tokenizer,
            reward_model_is_peft=reward_model_is_peft,
            max_kl=max_kl,
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
            training_args=ppo_config,
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
        validation_prompts: OutputDatasetColumn | OutputIterableDatasetColumn,
        reward_model: Union[
            TrainHFRewardModel,
            PreTrainedModel,
            "PeftModel",
            Callable[[list[str]], list[float]],
        ],
        reward_model_tokenizer: None | PreTrainedTokenizer = None,
        max_new_tokens: None | int = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generation_kwargs: None | dict[str, Any] = None,
        truncate: bool = True,
        epochs: float = 3.0,
        batch_size: int = 8,
        optimizer: Optimizer | Default = AUTO,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        lr_scheduler: None | LRScheduler = None,
        seed: int = 42,
        length_sampler: None | Callable = None,
        init_kl_coef: float = 0.2,
        adap_kl_ctrl: bool = True,
        adap_kl_target: float = 6.0,
        max_kl: None | float | Default = AUTO,
        **kwargs,
    ) -> "TrainHFPPO":
        self._setup_folder_and_resume(
            train_prompts=train_prompts,
            validation_prompts=validation_prompts,
            reward_model=reward_model,
            reward_model_tokenizer=reward_model_tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generation_kwargs=generation_kwargs,
            truncate=truncate,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_scheduler=lr_scheduler,
            seed=seed,
            length_sampler=length_sampler,
            init_kl_coef=init_kl_coef,
            adap_kl_ctrl=adap_kl_ctrl,
            adap_kl_target=adap_kl_target,
            max_kl=max_kl,
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
@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford,"""
            """ Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
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


__all__ = ["TrainHFPPO"]
