import gc
import json
import logging
import os
from functools import cached_property
from logging import Logger
from time import sleep
from typing import cast
from uuid import uuid4

import jsonlines
import openai
from datasets.fingerprint import Hasher
from jsonlines import Writer
from openai import BadRequestError, NotFoundError
from tiktoken import Encoding

from ..datasets import OutputDatasetColumn, OutputIterableDatasetColumn
from ..llms.openai import OpenAI, _is_chat_model
from ..steps.step import Step
from ..utils import ring_utils as ring
from ..utils.arg_utils import AUTO, Default
from ..utils.fingerprint_utils import stable_fingerprint
from ..utils.fs_utils import safe_fn
from .trainer import ModelNoLongerExistsError, Trainer


class TrainOpenAIFineTune(Trainer):
    def __init__(
        self,
        name: str,
        model_name: str,
        system_prompt: None | str = None,
        organization: None | str = None,
        api_key: None | str = None,
        base_url: None | str = None,
        api_version: None | str = None,
        retry_on_fail: bool = True,
        force: bool = False,
        verbose: bool | None = None,
        log_level: int | None = None,
        **kwargs,
    ):
        self.model_name = model_name
        super().__init__(name, force, verbose, log_level)
        self.organization = organization
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.kwargs = kwargs
        self.system_prompt = system_prompt
        if self.system_prompt is None and _is_chat_model(self.model_name):
            self.system_prompt = "You are a helpful assistant."

        # Setup API calling helpers
        self.retry_on_fail = retry_on_fail

    @property
    def resumable(self) -> bool:
        return True

    def get_logger(
        self, key: str, verbose: None | bool = None, log_level: None | int = None
    ) -> Logger:
        return self.logger

    @cached_property
    def retry_wrapper(self):
        return OpenAI.retry_wrapper.func(self)  # type: ignore[attr-defined]

    @cached_property
    def client(self) -> openai.OpenAI | openai.AzureOpenAI:
        return OpenAI.client.func(self)  # type: ignore[attr-defined]

    @cached_property
    def tokenizer(self) -> Encoding:
        return OpenAI.tokenizer.func(self)  # type: ignore[attr-defined]

    @ring.lru(maxsize=128)
    def get_max_context_length(self, max_new_tokens: int) -> int:
        """Gets the maximum context length for the model. When ``max_new_tokens`` is
        greater than 0, the maximum number of tokens that can be used for the prompt
        context is returned.

        Args:
            max_new_tokens: The maximum number of tokens that can be generated.

        Returns:
            The maximum context length.
        """  # pragma: no cover
        return OpenAI.get_max_context_length._callable.wrapped_callable(
            self, max_new_tokens
        )

    @ring.lru(maxsize=5000)
    def count_tokens(self, value: str) -> int:
        """Counts the number of tokens in a string.

        Args:
            value: The string to count tokens for.

        Returns:
            The number of tokens in the string.
        """
        pass
        return OpenAI.count_tokens._callable.wrapped_callable(self, value)

    def _train(  # type:ignore[override] # noqa: C901
        self,
        train_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_output: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_output: OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        epochs: float | Default = AUTO,
        batch_size: int | Default = AUTO,
        learning_rate_multiplier: float | Default = AUTO,
        **kwargs,
    ):
        # Save information for publishing
        train_step = train_input.step
        self._step_metadata = train_step._get_metadata(train_step.output)

        # Define Create JSONL Files Step
        class CreateOpenAIFineTuneJSONL(Step):
            def setup(self):
                self.register_input("input")
                self.register_input("output")
                self.register_output("none")

            def run(self_step):
                input, output = self_step.inputs["input"], self_step.inputs["output"]
                out_path = os.path.join(
                    self_step.get_run_output_folder_path(), "out.jsonl"
                )
                with cast(Writer, jsonlines.open(out_path, mode="w")) as writer:
                    for i, (prompt, completion) in enumerate(zip(input, output)):
                        if input.num_rows is not None or output.num_rows is not None:
                            self_step.progress = i / cast(
                                int, (input.num_rows or output.num_rows)
                            )
                        if not truncate:
                            if (
                                self.count_tokens(prompt)
                                + self.count_tokens(completion)
                            ) > self.get_max_context_length(max_new_tokens=0):
                                raise ValueError(
                                    "The length of your input and output exceeds the"
                                    " context length of the model."
                                )
                        if _is_chat_model(self.model_name):
                            writer.write(
                                {
                                    "messages": [
                                        {
                                            "role": "system",
                                            "content": self.system_prompt,
                                        },
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": completion},
                                    ]
                                }
                            )
                        else:
                            writer.write({"prompt": prompt, "completion": completion})
                return None

        # Create JSONL Files
        for split in ["train", "validation"]:
            split_jsonl_path = os.path.join(self._output_folder_path, f"{split}.jsonl")
            if not os.path.exists(split_jsonl_path):
                create_jsonl_step = CreateOpenAIFineTuneJSONL(
                    f"Create {split}.jsonl",
                    inputs={
                        "input": train_input if split == "train" else validation_input,
                        "output": (
                            train_output if split == "train" else validation_output
                        ),
                    },
                    progress_interval=120,
                )
                os.rename(
                    os.path.join(
                        create_jsonl_step.get_run_output_folder_path(), "out.jsonl"
                    ),
                    split_jsonl_path,
                )

        # Upload JSONL Files
        file_ids: dict[str, str] = {}
        for split in ["train", "validation"]:
            split_jsonl_path = os.path.join(self._output_folder_path, f"{split}.jsonl")
            split_file_id_path = os.path.join(
                self._output_folder_path, f"{split}_file_id.json"
            )

            for first_try in [True, False]:
                # Read file_id
                file_id = None
                if os.path.exists(split_file_id_path):
                    with open(split_file_id_path, "r") as split_file_id_fp:
                        file_id = json.load(split_file_id_fp)
                try:
                    file = self.client.files.retrieve(file_id=str((file_id)))
                    break
                except (NotFoundError,) if first_try else ():  # noqa: B030
                    pass

                # Upload file
                with open(split_jsonl_path, "rb") as split_jsonl_fp:
                    file = self.client.files.create(
                        file=(
                            "DataDreamer_"
                            + safe_fn(self.name)
                            + "_"
                            + os.path.basename(split_jsonl_path),
                            split_jsonl_fp,
                        ),
                        purpose="fine-tune",
                    )

                    # Save file_id
                    with open(split_file_id_path, "w+") as split_file_id_fp:
                        json.dump(file.id, split_file_id_fp)

            # Update file_id
            assert file_id is not None
            file_ids[split] = file_id

        # Create fine-tune job
        ft_job_id = None
        ft_job_id_path = os.path.join(self._output_folder_path, "ft_job_id.json")
        for first_try in [True, False]:
            # Read ft_job_id
            if os.path.exists(ft_job_id_path):
                with open(ft_job_id_path, "r") as ft_job_id_fp:
                    ft_job_id = json.load(ft_job_id_fp)
            try:
                self.client.fine_tuning.jobs.retrieve(fine_tuning_job_id=str(ft_job_id))
                break
            except (NotFoundError,) if first_try else ():  # noqa: B030
                pass

            # Create fine-tune job
            hyperparameters = {}
            if not isinstance(epochs, Default):
                hyperparameters["n_epochs"] = epochs
            if not isinstance(batch_size, Default):
                hyperparameters["batch_size"] = batch_size
            if not isinstance(learning_rate_multiplier, Default):
                hyperparameters["learning_rate_multiplier"] = learning_rate_multiplier
            with open(
                os.path.join(self._output_folder_path, "training_args.json"), "w+"
            ) as f:
                json.dump(hyperparameters, f, indent=4)
            ft_job = self.client.fine_tuning.jobs.create(
                training_file=file_ids["train"],
                validation_file=file_ids["validation"],
                model=self.model_name,
                hyperparameters=hyperparameters,  # type: ignore[arg-type]
                suffix="datadreamer",
            )

            # Save ft_job_id
            with open(ft_job_id_path, "w+") as ft_job_id_fp:
                json.dump(ft_job.id, ft_job_id_fp)
        assert ft_job_id is not None

        # Wait for fine-tune job to complete and log events and metrics
        try:
            trained_model_name = None
            trained_model_name_path = os.path.join(
                self._output_folder_path, "trained_model_name.json"
            )
            if not os.path.exists(trained_model_name_path):
                finished = False
                seen_event_ids = set()
                while True:
                    # Get events while there are more events
                    after = None
                    all_events = []
                    while True:
                        list_of_events = self.retry_wrapper(
                            func=self.client.fine_tuning.jobs.list_events,
                            fine_tuning_job_id=ft_job_id,
                            limit=1000,
                            after=after,
                        )
                        batch_of_events = list_of_events.data
                        if len(list_of_events.data) > 0:
                            after = list_of_events.data[-1].id
                        all_events.extend(batch_of_events)
                        if not list_of_events.has_more or any(
                            e.id in seen_event_ids for e in batch_of_events
                        ):
                            break

                    # Log events
                    for event in sorted(all_events, key=lambda e: e.created_at):
                        if (
                            getattr(logging, event.level.upper()) >= self.logger.level
                            and event.id not in seen_event_ids
                        ):
                            self.logger.info(event.message)
                            seen_event_ids.add(event.id)

                    # If finished, exit
                    if finished:
                        break

                    # Get the fine-tune job
                    ft_job = self.retry_wrapper(
                        func=self.client.fine_tuning.jobs.retrieve,
                        fine_tuning_job_id=ft_job_id,
                    )

                    # Check if finished
                    if ft_job.status == "succeeded":
                        finished = True
                        trained_model_name = ft_job.fine_tuned_model
                        sleep(1)
                    elif ft_job.status in ["failed", "cancelled"]:  # pragma: no cover
                        raise RuntimeError(
                            f"Fine-tune job failed with status {ft_job}."
                        )
                    else:
                        sleep(10)

                # Save trained_model_name
                with open(trained_model_name_path, "w+") as trained_model_name_fp:
                    json.dump(trained_model_name, trained_model_name_fp)

                    # Clean up uploaded files
                    self.retry_wrapper(
                        func=self.client.files.delete, file_id=file_ids["train"]
                    )
                    self.retry_wrapper(
                        func=self.client.files.delete, file_id=file_ids["validation"]
                    )
        finally:
            try:
                self.client.fine_tuning.jobs.cancel(ft_job_id)
            except (NotFoundError, BadRequestError):
                pass

            # Clean up resources after training
            self.unload_model()

    def train(  # type:ignore[override]
        self,
        train_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        train_output: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_input: OutputDatasetColumn | OutputIterableDatasetColumn,
        validation_output: OutputDatasetColumn | OutputIterableDatasetColumn,
        truncate: bool = True,
        epochs: float | Default = AUTO,
        batch_size: int | Default = AUTO,
        learning_rate_multiplier: float | Default = AUTO,
        **kwargs,
    ) -> "TrainOpenAIFineTune":
        self._setup_folder_and_resume(
            train_input=train_input,
            train_output=train_output,
            validation_input=validation_input,
            validation_output=validation_output,
            truncate=truncate,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
            **kwargs,
        )
        return self

    def _load(self, with_optimizations: bool = True):
        trained_model_name_path = os.path.join(
            self._output_folder_path, "trained_model_name.json"
        )
        with open(trained_model_name_path, "r") as trained_model_name_fp:
            trained_model_name = json.load(trained_model_name_fp)
        try:
            self.client.models.retrieve(model=trained_model_name)
        except NotFoundError:  # pragma: no cover
            raise ModelNoLongerExistsError(
                "The model no longer exists. It was possibly deleted from your account."
            ) from None
        return trained_model_name

    @property
    def model(self):
        """The name of the trained model after training."""
        assert (
            self._done
        ), "This trainer has not been run yet. Use `.train()` to start training."
        if self._model is None:  # pragma: no cover
            self._model = self._load()
        return self._model

    @property
    def model_path(self) -> str:
        """The name of the trained model after training."""
        return self.model

    @property
    def base_model_card(self) -> None | str:
        return OpenAI.model_card.func(self)  # type: ignore[attr-defined]

    @property
    def license(self) -> None | str:
        return OpenAI.license.func(self)  # type: ignore[attr-defined]

    @property
    def citation(self) -> None | list[str]:
        return OpenAI.citation.func(self)  # type: ignore[attr-defined]

    @cached_property
    def display_name(self) -> str:
        return f"{self.name} ({self.model_name})"

    def compute_fingerprint(self, **kwargs) -> str:
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
            self.system_prompt,
            column_fingerprints,
            stable_fingerprint(kwargs),
        ]
        fingerprint = Hasher.hash(to_hash)
        self.fingerprint = fingerprint
        return fingerprint

    def __ring_key__(self) -> int:
        return uuid4().int

    def unload_model(self):
        super().unload_model()

        # Delete cached client and tokenizer
        if "client" in self.__dict__:
            del self.__dict__["client"]
        if "tokenizer" in self.__dict__:
            del self.__dict__["tokenizer"]

        # Garbage collect
        gc.collect()


__all__ = ["TrainOpenAIFineTune"]
