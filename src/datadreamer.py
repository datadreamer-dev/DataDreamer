import logging
import os
import threading
from typing import TYPE_CHECKING

from transformers import logging as transformers_logging

from datasets.utils import logging as datasets_logging

from .logging import DATEFMT, DATETIME_FORMAT, STANDARD_FORMAT, logger
from .utils.fs_utils import safe_fn

if TYPE_CHECKING:  # pragma: no cover
    from .steps import Step


class DataDreamer:
    ctx = threading.local()

    def __init__(
        self,
        output_folder_path: str,
        verbose: bool = True,
        log_level: None | int = None,
        log_date: bool = False,
        hf_log=False,
    ):
        if os.path.isfile(output_folder_path):
            raise ValueError(
                f"Expected path to folder, but file exists at path {output_folder_path}."
            )
        self.output_folder_path = output_folder_path

        # Setup logger
        if log_date:
            formatter = logging.Formatter(
                DATETIME_FORMAT, datefmt=DATEFMT, validate=False
            )
            logger.handlers[0].setFormatter(formatter)
        else:
            formatter = logging.Formatter(
                STANDARD_FORMAT, datefmt=DATEFMT, validate=False
            )
            logger.handlers[0].setFormatter(formatter)
        if verbose:
            logger.setLevel(log_level or logging.INFO)
        else:
            logger.setLevel(logging.CRITICAL + 1)

        logger.info(f"Intialized. 🚀 Dreaming to folder: {self.output_folder_path}")

        # Take over HF loggers to prevent them from spewing logs
        DataDreamer.ctx.hf_log = hf_log
        DataDreamer.ctx._hf_datasets_prog_bar = (
            datasets_logging.is_progress_bar_enabled()
        )
        DataDreamer.ctx._hf_datasets_verbosity = datasets_logging.get_verbosity()
        DataDreamer.ctx._hf_transformers_verbosity = (
            transformers_logging.get_verbosity()
        )
        DataDreamer.ctx._hf_transformers_prog_bar = (
            transformers_logging.is_progress_bar_enabled()
        )
        if not DataDreamer.ctx.hf_log:
            DataDreamer._disable_hf_datasets_logging()
            DataDreamer._disable_hf_transformers_logging()

    @staticmethod
    def initialized() -> bool:
        return hasattr(DataDreamer.ctx, "initialized")

    @staticmethod
    def _has_step_name(name: str) -> bool:
        return (
            name in DataDreamer.ctx.step_names
            or safe_fn(name) in DataDreamer.ctx.step_names
        )

    @staticmethod
    def _add_step(step: "Step"):
        if DataDreamer._has_step_name(step.name):
            raise ValueError(f"A step already exists with the name: {step.name}")
        DataDreamer.ctx.steps.append(step)
        DataDreamer.ctx.step_names.add(step.name)
        DataDreamer.ctx.step_names.add(safe_fn(step.name))

    @staticmethod
    def _new_step_name(old_name: str, transform: str):
        i = 1
        new_name = f"{old_name} ({transform})"
        while DataDreamer._has_step_name(new_name):
            i += 1
            new_name = f"{old_name} ({transform} #{i})"
        return new_name

    @staticmethod
    def _enable_hf_datasets_logging(logs=False, progress_bars=True):
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            if logs:
                datasets_logging.set_verbosity(DataDreamer.ctx._hf_datasets_verbosity)
            if progress_bars:
                if DataDreamer.ctx._hf_datasets_prog_bar:
                    datasets_logging.enable_progress_bar()
                else:
                    datasets_logging.disable_progress_bar()

    @staticmethod
    def _disable_hf_datasets_logging():
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            datasets_logging.set_verbosity_error()
            datasets_logging.disable_progress_bar()

    @staticmethod
    def _enable_hf_transformers_logging(logs=False, progress_bars=True):
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            if logs:
                transformers_logging.set_verbosity(
                    DataDreamer.ctx._hf_transformers_verbosity
                )
            if progress_bars:
                if DataDreamer.ctx._hf_transformers_prog_bar:
                    transformers_logging.enable_progress_bar()
                else:
                    transformers_logging.disable_progress_bar()

    @staticmethod
    def _disable_hf_transformers_logging():
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()

    def __enter__(self):
        if hasattr(DataDreamer.ctx, "steps"):
            raise RuntimeError("Cannot nest DataDreamer() context managers.")
        os.makedirs(self.output_folder_path, exist_ok=True)
        DataDreamer.ctx.output_folder_path = self.output_folder_path
        DataDreamer.ctx.steps = []
        DataDreamer.ctx.step_names = set()
        DataDreamer.ctx.initialized = True

    def __exit__(self, exc_type, exc_value, exc_tb):
        if not DataDreamer.ctx.hf_log:
            DataDreamer._enable_hf_datasets_logging(logs=True, progress_bars=True)
            DataDreamer._enable_hf_transformers_logging(logs=True, progress_bars=True)
        DataDreamer.ctx = threading.local()
        logger.info(f"Done. ✨ Results in folder: {self.output_folder_path}")
