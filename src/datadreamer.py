import logging
import os
import threading

from .logging import DATEFMT, DATETIME_FORMAT, STANDARD_FORMAT, logger
from .utils.fs_utils import safe_fn


class DataDreamer:
    ctx = threading.local()

    def __init__(
        self,
        output_folder_path: str,
        verbose: bool = True,
        log_level: None | int = None,
        log_date: bool = False,
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
    def _add_step(step):
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

    def __enter__(self):
        if hasattr(DataDreamer.ctx, "steps"):
            raise RuntimeError("Cannot nest DataDreamer() context managers.")
        os.makedirs(self.output_folder_path, exist_ok=True)
        DataDreamer.ctx.output_folder_path = self.output_folder_path
        DataDreamer.ctx.steps = []
        DataDreamer.ctx.step_names = set()
        DataDreamer.ctx.initialized = True

    def __exit__(self, exc_type, exc_value, exc_tb):
        DataDreamer.ctx = threading.local()
        logger.info(f"Done. ✨ Results in folder: {self.output_folder_path}")
