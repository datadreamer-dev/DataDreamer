import json
import os
from copy import copy
from functools import cached_property, partial
from io import BytesIO
from shutil import copy2
from typing import TYPE_CHECKING, Any, Type, cast

import torch
from datasets.fingerprint import Hasher

from .. import DataDreamer
from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..logging import logger
from ..utils.arg_utils import AUTO, DEFAULT, Default, default_to
from ..utils.background_utils import RunIfTimeout
from ..utils.device_utils import model_to_device, validate_device
from ..utils.distributed_utils import (
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
    filter_model_warnings,
    get_attn_implementation,
    get_config,
    get_model_optional_kwargs,
    get_model_prompt_template,
    get_tokenizer,
    is_encoder_decoder,
    peft_module_casting_to_dtype,
    validate_peft_config,
    validate_quantization_config,
)
from ..utils.import_utils import ignore_transformers_warnings
from .trainer import Trainer as DataDreamerTrainer

with ignore_transformers_warnings():
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    from transformers.utils.quantization_config import QuantizationConfigMixin


if TYPE_CHECKING:  # pragma: no cover
    from ..utils.hf_training_utils import TrainingArguments


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
        self.quantization_config = validate_quantization_config(
            quantization_config=quantization_config, dtype=self.dtype
        )
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

        # Get device and device_map
        model_device = default_to(device, self.device)
        to_device, to_device_map, to_device_map_max_memory = model_to_device(
            device=model_device,
            device_map=None,
            is_train=True,
            model_name=self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            quantization_config=self.quantization_config,
        )

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
            torch_dtype=self.dtype,
            attn_implementation=get_attn_implementation(
                model_name=self.model_name,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                model_kwargs=self.kwargs,
                optimize=True,
            ),
            device_map=to_device_map,
            max_memory=to_device_map_max_memory,
            **self.kwargs,
            **get_model_optional_kwargs(quantization_config=self.quantization_config),
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
        if to_device is not None:
            model = model.to(to_device)

        # Create PeftModel if peft_config
        if self.peft_config:
            # Two warnings we can't silence are thrown by peft at import-time so
            # we import this library only when needed
            with ignore_transformers_warnings():
                from peft import get_peft_model, prepare_model_for_kbit_training

            if self.quantization_config:  # pragma: no cover
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=True
                )
            model = get_peft_model(
                model, validate_peft_config(model=model, peft_config=self.peft_config)
            )
            peft_module_casting_to_dtype(model=model, dtype=self.dtype)

        # Switch model to train mode
        if is_ref_model:
            model.eval()
        else:
            model.train()

        # Filter any warnings from the model
        filter_model_warnings()

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
        training_args: "None | TrainingArguments",
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

        # Get device and device_map
        to_device, to_device_map, to_device_map_max_memory = model_to_device(
            device=self.device,
            device_map=None,
            is_train=False,
            model_name=self.model_name,
            revision=self.revision,
            trust_remote_code=self.trust_remote_code,
            quantization_config=self.quantization_config,
        )

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
                torch_dtype=self.dtype,
                attn_implementation=get_attn_implementation(
                    model_name=self.model_name,
                    revision=self.revision,
                    trust_remote_code=self.trust_remote_code,
                    model_kwargs=self.kwargs,
                    optimize=with_optimizations,
                ),
                device_map=to_device_map,
                max_memory=to_device_map_max_memory,
                **self.kwargs,
                **get_model_optional_kwargs(
                    quantization_config=self.quantization_config
                ),
                **classification_kwargs,
            )
            model = PeftModel.from_pretrained(
                model,
                model_id=MODEL_DIR,
                torch_dtype=self.dtype,
                **self.kwargs,
                **get_model_optional_kwargs(
                    quantization_config=self.quantization_config
                ),
            )
        else:
            model = self.auto_cls.from_pretrained(
                MODEL_DIR,
                torch_dtype=self.dtype,
                attn_implementation=get_attn_implementation(
                    model_name=self.model_name,
                    revision=self.revision,
                    trust_remote_code=self.trust_remote_code,
                    model_kwargs=self.kwargs,
                    optimize=with_optimizations,
                ),
                device_map=to_device_map,
                max_memory=to_device_map_max_memory,
                **self.kwargs,
                **get_model_optional_kwargs(
                    quantization_config=self.quantization_config
                ),
            )

        # Send model to accelerator device
        if to_device is not None:
            model = model.to(to_device)

        # Switch model to eval mode
        model.eval()

        if with_optimizations:
            # Torch compile
            #
            # Note: Disabling due to a bug in PyTorch where encoder-decoder (T5)
            # models get compiled over an over again, making it slow. If enabled in the
            # future, it would be better to use TrainingArguments()'s torch_compile
            # arg.
            #
            # torch._dynamo.config.suppress_errors = True
            # model = torch.compile(model)
            pass

        # Filter any warnings from the model
        filter_model_warnings()

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
{pipeline_tag}
{widget}
---
# Model Card

[Add more information here](https://huggingface.co/templates/model-card-example)

{body}

---
This model was trained with"""
            f""" a {"synthetic " if is_synthetic else ""}dataset with"""
            f""" [DataDreamer 🤖💤](https://datadreamer.dev)."""
            f""" The {"synthetic " if is_synthetic else ""}dataset card and model"""
            f""" card can be found [here](datadreamer.json)."""
            f""" The training arguments can be found [here](training_args.json)."""
        )
        if os.path.exists(self.model_name) and os.path.isdir(self.model_name):
            readme_contents = readme_contents.replace("\nbase_model: {base_model}", "")
        else:
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
