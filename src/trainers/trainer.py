import gc
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from functools import cache, cached_property, partial, total_ordering
from logging import Logger
from typing import Any

import torch

from .. import DataDreamer, logging as datadreamer_logging
from ..logging import DATEFMT, logger
from ..project.environment import RUNNING_IN_PYTEST
from ..steps.data_card import DataCardType, sort_data_card
from ..utils.distributed_utils import run_distributed
from ..utils.fs_utils import clear_dir, mkdir, move_dir, safe_fn
from ..utils.import_utils import ignore_training_warnings, ignore_transformers_warnings

with ignore_transformers_warnings():
    from transformers import TrainerState


class ModelNoLongerExistsError(Exception):
    pass


@total_ordering
@dataclass(eq=False)
class JointMetric:
    is_joint_metric: bool
    primary: float
    primary_name: str
    secondary: float
    secondary_name: str
    secondary_inversed: bool

    def __eq__(self, other):
        return (self.primary, self.secondary) == (other.primary, other.secondary)

    def __lt__(self, other):
        return (self.primary, self.secondary) < (other.primary, other.secondary)

    def __sub__(self, other):
        return self.secondary - other.secondary

    def __repr__(self) -> str:  # pragma: no cover
        secondary = (-1 * self.secondary) if self.secondary_inversed else self.secondary
        return (
            f"JointMetric({self.primary_name}={self.primary},"
            f" {self.secondary_name}={secondary})"
        )

    def __str_(self) -> str:  # pragma: no cover
        return self.__repr__()


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


class Trainer(ABC):
    _trainer_tags = ["datadreamer"]

    def __init__(  # noqa: C901
        self,
        name: str,
        force: bool = False,
        verbose: None | bool = None,
        log_level: None | int = None,
    ):
        """Base class for all trainers.

        Args:
            name: The name of the trainer.
            force: Whether to force run the trainer (ignore saved results).
            verbose: Whether or not to print verbose logs.
            log_level: The logging level to use (:py:data:`~logging.DEBUG`, :py:data:`~logging.INFO`, etc.).
        """
        if not DataDreamer.initialized() or DataDreamer.is_running_in_memory():
            raise RuntimeError("Trainers only be run within DataDreamer() context.")

        # Check pid
        if DataDreamer.is_background_process():  # pragma: no cover
            raise RuntimeError(
                f"Trainers must be initialized in the same process"
                f" ({os.getpid()}) as the DataDreamer() context manager"
                f" ({DataDreamer.ctx.pid})."
            )

        self.name = DataDreamer._new_step_name(name)
        if len(self.name) == 0:
            raise ValueError("You must provide a name for the Trainer.")
        self.force = force

        # Create an output folder for the trainer
        self._output_folder_path = os.path.join(
            DataDreamer.get_output_folder_path(),
            safe_fn(self.name, allow_slashes=True, to_lower=True),
        )
        os.makedirs(self._output_folder_path, exist_ok=True)

        # Initialize the logger
        self.verbose = verbose
        self.log_level = log_level
        self.logger: Logger
        stderr_handler = logging.StreamHandler()
        stderr_handler.setLevel(logging.NOTSET)
        self.logger = logging.getLogger(
            f"datadreamer.trainer.{safe_fn(self.name, allow_slashes=True, to_lower=True)}"
        )
        if RUNNING_IN_PYTEST:
            self.logger.propagate = True
        else:
            self.logger.propagate = False  # pragma: no cover
        log_format: str = (
            logger.handlers[0].formatter and logger.handlers[0].formatter._fmt
        ) or datadreamer_logging.STANDARD_FORMAT
        log_format = log_format.replace(
            "%(message)s", f"[{self.display_name}] %(message)s"
        )
        formatter = logging.Formatter(log_format, datefmt=DATEFMT, validate=False)
        stderr_handler.setFormatter(formatter)
        self.logger.handlers.clear()
        self.logger.addHandler(stderr_handler)
        effective_level = logger.level if self.log_level is None else self.log_level
        if self.verbose:
            self.logger.setLevel(min(logging.DEBUG, effective_level))
        elif self.verbose is False:
            self.logger.setLevel(logging.CRITICAL + 1)
        else:
            self.logger.setLevel(effective_level)

        # Initialize resume, done variables
        self._resumed = False
        self._done = False

        # Create placeholders that will be set after trained model
        self._step_metadata: None | dict[str, Any] = None
        self._model: Any = None
        self.fingerprint: None | str = None

        # Unused variables
        self._orig_background = False
        self.background = False

    @property
    def resumable(self) -> bool:
        return True

    def _setup_folder(self, fingerprint: str, should_clear: bool = False):
        mkdir(self._output_folder_path)
        if should_clear:
            clear_dir(self._output_folder_path)
        mkdir(os.path.join(self._output_folder_path, "_checkpoints"))
        fingerprint_path = os.path.join(self._output_folder_path, "fingerprint.json")
        with open(fingerprint_path, "w+") as f:
            json.dump(fingerprint, f, indent=4)

    def __setup_folder_and_resume(self, **kwargs):  # noqa: C901
        # Compute fingerprint
        fingerprint = self.compute_fingerprint(**kwargs)

        # Paths
        done_path = os.path.join(self._output_folder_path, "training_complete.json")
        fingerprint_path = os.path.join(self._output_folder_path, "fingerprint.json")

        # Check if fingerprint.json exists
        prev_fingerprint: None | str = None
        if os.path.exists(fingerprint_path):
            try:
                with open(fingerprint_path, "r") as f:
                    prev_fingerprint = json.load(f)
            except FileNotFoundError:  # pragma: no cover
                pass

        # Check if training is complete
        if os.path.exists(done_path):
            try:
                with open(done_path, "r") as f:
                    self._done = json.load(f)
            except FileNotFoundError:  # pragma: no cover
                pass

        # We have already run this trainer
        if prev_fingerprint == fingerprint and not self.force:
            self._resumed = True
            if self._done:
                try:
                    self._model = self._load()
                except ModelNoLongerExistsError:  # pragma: no cover
                    if os.path.exists(done_path):
                        os.remove(done_path)
                    if os.path.exists(fingerprint_path):
                        os.remove(fingerprint_path)
                    self.__setup_folder_and_resume(**kwargs)  # Retry loading again
                logger.info(
                    f"Trainer '{self.name}' result loaded from disk. 🙌 It was previously run"
                    " and saved."
                )
                return

        # We have already run this trainer, but it is outdated, back up the results
        if prev_fingerprint is not None and (
            prev_fingerprint != fingerprint or self.force
        ):
            # ...but it was a different version, backup the results and we'll need
            # to re-run this trainer
            logger.info(
                f"Trainer '{self.name}' was previously run and saved, but was outdated. 😞"
            )
            backup_path = os.path.join(
                DataDreamer.get_output_folder_path(),
                "_backups",
                safe_fn(self.name, allow_slashes=True, to_lower=True),
                prev_fingerprint,
            )
            logger.debug(
                f"Trainer '{self.name}' outdated results are being backed up: {backup_path}"
            )
            move_dir(self._output_folder_path, backup_path)
            logger.debug(
                f"Trainer '{self.name}' outdated results are backed up: {backup_path}"
            )

        # Check if we have old results for this trainer that can be restored
        restore_path = os.path.join(
            DataDreamer.get_output_folder_path(),
            "_backups",
            safe_fn(self.name, allow_slashes=True, to_lower=True),
            fingerprint,
        )
        if (
            (prev_fingerprint is None or prev_fingerprint != fingerprint)
            and os.path.isfile(os.path.join(restore_path, "fingerprint.json"))
            and not self.force
        ):
            logger.info(
                f"Trainer '{self.name}' was previously run and the results were backed up. 💾"
            )
            logger.info(
                f"Trainer '{self.name}' backed up results are being restored: {restore_path}"
            )
            move_dir(restore_path, self._output_folder_path)
            logger.info(f"Trainer '{self.name}' backed up results were restored.")
            self.__setup_folder_and_resume(**kwargs)  # Retry loading
            return

        # Train
        self._setup_folder(fingerprint=fingerprint, should_clear=not self.resumable)
        if self._resumed and self.resumable:
            logger.info(f"Trainer '{self.name}' is running (resumed). ⏳")
        else:
            self._resumed = False
            logger.info(f"Trainer '{self.name}' is running. ⏳")
        try:
            orig_cuda = (
                torch.cuda.current_device() if torch.cuda.is_available() else None
            )
            orig_environ = os.environ.copy()
            if hasattr(self, "device") and isinstance(
                self.device, list
            ):  # pragma: no cover
                run_distributed(
                    distributed_config=self.distributed_config,  # type:ignore[attr-defined]
                    devices=self.device,
                    func=partial(self._train, **kwargs),
                    args=(),
                    logger=self.logger,
                )
            else:
                with ignore_training_warnings():
                    self._train(**kwargs)
        finally:
            # TrainingArguments() postinit modifies os.environ, so we restore it
            # after running any training procedure
            os.environ.clear()
            for k, v in orig_environ.items():
                os.environ[k] = v
            if torch.cuda.is_available():  # pragma: no cover
                torch.cuda.set_device(orig_cuda)
        gc.collect()
        with open(done_path, "w+") as f:
            json.dump(True, f, indent=4)
        self._done = True
        self._model = self._load()
        logger.info(f"Trainer '{self.name}' finished and is saved to disk. 🎉")

    def _setup_folder_and_resume(self, **kwargs):
        assert not self._done, "This trainer has already been run."
        try:
            DataDreamer._start_step(self)
            self.__setup_folder_and_resume(**kwargs)
        finally:
            DataDreamer._stop_step()

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def train(self) -> "Trainer":
        """Train the model."""
        pass

    @abstractmethod
    def _load(self, with_optimizations: bool = True):
        pass

    @property
    def model(self):
        """An instance of the trained model after training."""
        assert (
            self._done
        ), "This trainer has not been run yet. Use `.train()` to start training."
        if self._model is None:  # pragma: no cover
            self._model = self._load()
        return self._model

    @property
    @abstractmethod
    def model_path(self) -> str:
        """The path to the trained model after training."""
        pass

    @property
    @abstractmethod
    def base_model_card(self) -> None | str:
        return None

    @property
    @abstractmethod
    def license(self) -> None | str:
        return None

    @property
    @abstractmethod
    def citation(self) -> None | list[str]:
        return None

    @property
    def _model_card(self):
        assert (
            self._done
        ), "This trainer has not been run yet. Use `.train()` to start training."
        assert self._step_metadata is not None
        orig_step_metadata = self._step_metadata.copy()
        model_card: dict[str, list[Any]] = {
            DataCardType.DATETIME: datetime.now().isoformat()  # type:ignore[dict-item]
        }
        if self.base_model_card is not None:
            model_card[DataCardType.MODEL_CARD] = [self.base_model_card]
        if self.license is not None:
            model_card[DataCardType.LICENSE] = [self.license]
        if self.citation is not None:
            model_card[DataCardType.CITATION] = self.citation
        data_card = orig_step_metadata.pop("data_card")
        orig_step_metadata["type"] = type(self).__name__
        orig_step_metadata["name"] = self.name
        orig_step_metadata["version"] = self.version
        orig_step_metadata["fingerprint"] = self.fingerprint
        del orig_step_metadata["pickled"]
        return {
            "model_card": sort_data_card(model_card),
            "data_card": data_card,
            **orig_step_metadata,
        }

    def model_card(self):
        """Print the data card for the step."""
        print(json.dumps(self._model_card, indent=4))

    @property
    def version(self) -> float:  # pragma: no cover
        return 1.0

    @cached_property
    def display_name(self) -> str:  # pragma: no cover
        return f"{self.name}"

    @abstractmethod
    def compute_fingerprint(self) -> str:
        pass

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.display_name}>"

    def unload_model(self):
        """Unloads the trained model from memory."""
        self._model = None

        # Garbage collect
        gc.collect()


__all__ = ["Trainer"]
