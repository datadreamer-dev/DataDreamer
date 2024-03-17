import inspect
import locale
import logging
import os
import sys
from collections import UserDict, defaultdict
from multiprocessing.context import SpawnProcess
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, cast

import tqdm
from datasets.fingerprint import disable_caching, enable_caching, is_caching_enabled
from datasets.utils import logging as datasets_logging
from filelock import FileLock
from sqlitedict import SqliteDict

from . import logging as datadreamer_logging
from .logging import DATEFMT, logger
from .utils.background_utils import get_thread_id
from .utils.fs_utils import safe_fn
from .utils.import_utils import ignore_pydantic_warnings, ignore_transformers_warnings

with ignore_transformers_warnings():
    from optimum.utils import logging as optimum_logging
    from setfit import logging as setfit_logging
    from transformers import logging as transformers_logging

with ignore_pydantic_warnings():
    from huggingface_hub.utils import logging as huggingface_hub_logging

if TYPE_CHECKING:  # pragma: no cover
    from .steps import Step
    from .trainers import Trainer

_DATADREAMER_CTX_LOCK = Lock()
_ADD_STEP_LOCK = Lock()

_old_tqdm__init__ = tqdm.std.tqdm.__init__


class DataDreamer:
    ctx: Any = UserDict()

    def __init__(
        self,
        output_folder_path: str,
        verbose: None | bool = None,
        log_level: None | int = None,
        log_date: bool = False,
        hf_log=False,
    ):
        """Constructs a DataDreamer session.

        Args:
            output_folder_path: The output folder path to organize, cache, and save
                results of each step or trainer run within a session.
            verbose: Whether or not to print verbose logs.
            log_level: The logging level to use (:py:data:`~logging.DEBUG`, :py:data:`~logging.INFO`, etc.).
            log_date: Whether or not to include the date and time in the logs.
            hf_log: Whether to override and silence overly verbose Hugging Face logs
                within the session. Defaults to ``True``. Set to ``False`` to debug
                issues related to Hugging Face libraries.
        """
        self.output_folder_path: None | str
        if output_folder_path.strip().lower() in ":memory:":
            self.output_folder_path = None
        else:
            if os.path.isfile(output_folder_path):
                raise ValueError(
                    "Expected path to folder, but file exists at path"
                    " {output_folder_path}."
                )
            self.output_folder_path = output_folder_path
        self.verbose = verbose
        self.log_level = log_level
        self.log_date = log_date
        self.hf_log = hf_log
        self.patched_loggers: dict[str, tuple] = {}

    @staticmethod
    def initialized() -> bool:
        """Queries whether or not a DataDreamer session is currently active.

        Returns:
            Whether or not a DataDreamer session is currently active.
        """
        return hasattr(DataDreamer.ctx, "initialized")

    @staticmethod
    def is_running_in_memory() -> bool:
        return DataDreamer.initialized() and DataDreamer.ctx.in_memory

    @staticmethod
    def is_background_process() -> bool:
        return int(os.environ.get("DATADREAMER_BACKGROUND_PROCESS", -1)) == 1 or (
            DataDreamer.initialized() and DataDreamer.ctx.pid != os.getpid()
        )

    @staticmethod
    def is_registered_thread() -> bool:
        return (
            not DataDreamer.initialized()
            or get_thread_id() in DataDreamer.ctx.thread_ids
        )

    @staticmethod
    def get_output_folder_path() -> str:
        """Gets the output folder path of the current DataDreamer session.

        Returns:
            The output folder path of the current DataDreamer session.
        """
        if hasattr(DataDreamer.ctx, "output_folder_path"):
            return DataDreamer.ctx.output_folder_path
        else:
            raise RuntimeError("DataDreamer is running in-memory.")

    @staticmethod
    def _has_step_name(name: str) -> bool:
        return (
            name in DataDreamer.ctx.step_names
            or safe_fn(name, allow_slashes=True, to_lower=True)
            in DataDreamer.ctx.step_names
            or name.split(" / ")[-1] == "_dataset"
        )

    @staticmethod
    def _start_step(step: "Step | Trainer"):
        from .steps import SuperStep
        from .trainers import Trainer

        with _ADD_STEP_LOCK:
            if len(DataDreamer.ctx.step_stack[get_thread_id()]) > 0:
                parent_step = DataDreamer.ctx.step_stack[get_thread_id()][-1]
                parent_is_superstep = isinstance(parent_step, SuperStep) or isinstance(
                    parent_step, Trainer
                )
                child_is_superstep = isinstance(step, SuperStep)
                if not parent_is_superstep:
                    raise RuntimeError(
                        f"Step '{step.name}' was instantiated within step"
                        f" '{parent_step.name}'. The '{parent_step.name}' step must be"
                        " of the `SuperStep` type to run nested steps within it."
                    )
                if parent_step._orig_background:
                    step._orig_background = True
                    if not child_is_superstep:
                        step.background = True
            DataDreamer.ctx.step_stack[get_thread_id()].append(step)
            DataDreamer.ctx.steps.append(step)

    @staticmethod
    def _get_parent_steps() -> "list[Step | Trainer]":
        return DataDreamer.ctx.step_stack[get_thread_id()][:-1]

    @staticmethod
    def _get_parent_step() -> "None | Step | Trainer":
        if len(DataDreamer._get_parent_steps()) > 0:
            return DataDreamer._get_parent_steps()[-1]
        else:
            return None

    @staticmethod
    def _stop_step():
        DataDreamer.ctx.step_stack[get_thread_id()].pop()

    @staticmethod
    def _register_child_thread(parent_thread_id: tuple[int, int]):
        DataDreamer.ctx.thread_ids.add(get_thread_id())
        if parent_thread_id in DataDreamer.ctx.step_stack:
            DataDreamer.ctx.step_stack[get_thread_id()] = DataDreamer.ctx.step_stack[
                parent_thread_id
            ]
        else:
            DataDreamer.ctx.step_stack[get_thread_id()] = []

    @staticmethod
    def _add_process(process: SpawnProcess):
        DataDreamer.ctx.background_processes.append(process)

    @staticmethod
    def _new_step_name(
        old_name: str, transform: None | str = None, record: bool = True
    ):
        # Check the name
        assert (
            "/" not in old_name
        ), f"The step name '{old_name}' cannot contain '/' characters."
        assert (
            old_name not in ["_backups", ".datadreamer_save_cache"]
        ), f"The step name '{old_name}' is invalid, please choose a different step name."

        # Get final name
        if DataDreamer.initialized():
            old_name = old_name.split(" / ")[-1]
            prefix = " / ".join(
                [
                    step.name.split(" / ")[-1]
                    for step in DataDreamer.ctx.step_stack[get_thread_id()]
                ]
            )
            if len(DataDreamer.ctx.step_stack[get_thread_id()]) > 0:
                prefix += " / "
            i = 1
            if transform:
                new_name = f"{prefix}{old_name} ({transform})"
            else:
                new_name = f"{prefix}{old_name}"
            while DataDreamer._has_step_name(new_name):
                i += 1
                if transform:
                    new_name = f"{prefix}{old_name} ({transform} #{i})"
                else:
                    new_name = f"{prefix}{old_name} #{i}"
            if record:
                DataDreamer.ctx.step_names.add(new_name)
                DataDreamer.ctx.step_names.add(
                    safe_fn(new_name, allow_slashes=True, to_lower=True)
                )
            return new_name
        else:
            return old_name

    @staticmethod
    def _add_cleanup_func(cleanup_func: Callable):
        DataDreamer.ctx.cleanup_funcs.append(cleanup_func)

    def _patch_logger(self, module: str, display_name: str):
        if module not in self.patched_loggers:
            # Get the logger and save its information
            module_logger = logging.getLogger(module)
            self.patched_loggers[module] = (
                module_logger.level,
                module_logger.propagate,
                module_logger.handlers,
            )

            # Set up the logger
            module_logger.propagate = False
            module_logger.handlers.clear()
            stderr_handler = logging.StreamHandler()
            stderr_handler.setLevel(logging.NOTSET)
            module_logger.propagate = False
            log_format: str = (
                logger.handlers[0].formatter and logger.handlers[0].formatter._fmt
            ) or datadreamer_logging.STANDARD_FORMAT
            log_format = log_format.replace(
                "%(message)s", f"[{display_name}] %(message)s"
            )
            formatter = logging.Formatter(log_format, datefmt=DATEFMT, validate=False)
            stderr_handler.setFormatter(formatter)
            module_logger.addHandler(stderr_handler)

    def _unpatch_loggers(self):
        for module, (level, propagate, handlers) in self.patched_loggers.items():
            # Get the logger and its saved information
            module_logger = logging.getLogger(module)
            module_logger.handlers.clear()
            module_logger.setLevel(level)
            module_logger.propagate = propagate
            module_logger.handlers = handlers

    def _patch_tqdm(self):
        def tqdm__init__patch(_self, *args, **kwargs):
            _old_tqdm__init__(_self, *args, **kwargs)
            outer_module = inspect.getmodule(inspect.stack()[2][0])
            if hasattr(outer_module, "__name__"):
                module_name = outer_module.__name__.split(  # type:ignore[union-attr]
                    "."
                )[0]
            else:  # pragma: no cover
                module_name = None
            if module_name in self.patched_loggers:
                module_logger = logging.getLogger(module_name)
                if (
                    len(module_logger.handlers) > 0
                    and module_logger.handlers[0].formatter
                ):
                    fmt = cast(str, module_logger.handlers[0].formatter._fmt)
                    fmt_prefix = (
                        fmt[: fmt.index("%(message)s")]
                        .replace("%(asctime)s", "")
                        .replace("%(name)s", "")
                        .replace("%(levelname)s", "")
                        .strip()
                    )
                    _self.__dict__["bar_format"] = kwargs.get(
                        "bar_format", fmt_prefix + " {l_bar}{bar}{r_bar}"
                    )

        tqdm.std.tqdm.__init__ = tqdm__init__patch

    def _unpatch_tqdm(self):
        tqdm.std.tqdm.__init__ = _old_tqdm__init__

    @staticmethod
    def _enable_hf_datasets_logging(logs=False, progress_bars=True):
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            if logs:
                datasets_logging.set_verbosity(DataDreamer.ctx._hf_datasets_verbosity)
            if progress_bars:
                if DataDreamer.ctx._hf_datasets_prog_bar:
                    datasets_logging.enable_progress_bar()
                else:
                    datasets_logging.disable_progress_bar()  # pragma: no cover

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
                    transformers_logging.disable_progress_bar()  # pragma: no cover

    @staticmethod
    def _disable_hf_transformers_logging():
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()

    @staticmethod
    def _enable_hf_optimum_logging(logs=False):
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            if logs:
                optimum_logging.set_verbosity(DataDreamer.ctx._hf_optimum_verbosity)

    @staticmethod
    def _disable_hf_optimum_logging():
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            optimum_logging.set_verbosity_error()

    @staticmethod
    def _enable_hf_huggingface_hub_logging(logs=False):
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            if logs:
                huggingface_hub_logging.set_verbosity(
                    DataDreamer.ctx._hf_huggingface_hub_verbosity
                )

    @staticmethod
    def _disable_hf_huggingface_hub_logging():
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            huggingface_hub_logging.set_verbosity_error()

    @staticmethod
    def _enable_setfit_logging(logs=False, progress_bars=True):
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            if logs:
                setfit_logging.set_verbosity(DataDreamer.ctx._setfit_verbosity)
            if progress_bars:
                if DataDreamer.ctx._setfit_prog_bar:
                    setfit_logging.enable_progress_bar()
                else:
                    setfit_logging.disable_progress_bar()  # pragma: no cover

    @staticmethod
    def _disable_setfit_logging():
        if hasattr(DataDreamer.ctx, "hf_log") and not DataDreamer.ctx.hf_log:
            setfit_logging.set_verbosity_error()
            setfit_logging.disable_progress_bar()

    def __enter__(self):  # noqa: C901
        from .utils.distributed_utils import is_distributed

        if hasattr(DataDreamer.ctx, "steps"):
            raise RuntimeError("Only one DataDreamer context may be active at a time.")

        # Set encoding to UTF-8
        is_utf_8_encoding = lambda: any(  # noqa: E731
            utf8_locale in (locale.getlocale(locale.LC_CTYPE)[1] or "").lower()
            for utf8_locale in ["utf8", "utf-8"]
        )
        if not is_utf_8_encoding():  # pragma: no cover
            # Detect if the default encoding is not UTF-8 and try to see if it is available
            # and can be changed. This is to fix a bug on some improperly configured older
            # Linux systems.
            # See: https://github.com/datadreamer-dev/DataDreamer/issues/13
            for locale_string in ["C.UTF8", "C.UTF-8", "en_US.UTF-8"]:
                try:
                    locale.setlocale(locale.LC_CTYPE, locale_string)
                    if is_utf_8_encoding():
                        # Worked we were able to reset the encoding back to UTF-8
                        # Now, we apply hacks to now set the encodings to utf-8 across some of
                        # the standard places where Python may use the wrong encoding.
                        sys.stdin.reconfigure(encoding="utf-8")  # type:ignore[attr-defined]
                        sys.stdout.reconfigure(encoding="utf-8")  # type:ignore[attr-defined]
                        sys.stderr.reconfigure(encoding="utf-8")  # type:ignore[attr-defined]
                        locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"
                        break
                except locale.Error:
                    pass

        # Initialize
        _DATADREAMER_CTX_LOCK.acquire()
        if self.output_folder_path:
            DataDreamer.ctx.in_memory = False
            os.makedirs(self.output_folder_path, exist_ok=True)
            DataDreamer.ctx.output_folder_path = self.output_folder_path
        else:
            DataDreamer.ctx.in_memory = True
            self._hf_datasets_caching = is_caching_enabled()
            disable_caching()
        DataDreamer.ctx.step_stack = defaultdict(list)
        DataDreamer.ctx.steps = []
        DataDreamer.ctx.step_names = set()
        DataDreamer.ctx.thread_ids = set([get_thread_id()])
        DataDreamer.ctx.background_processes = []
        DataDreamer.ctx.pid = os.getpid()
        DataDreamer.ctx.caches = cast(dict[str, tuple[SqliteDict, FileLock]], {})
        DataDreamer.ctx.cleanup_funcs = []

        # Setup logger
        if self.log_date:
            formatter = logging.Formatter(
                datadreamer_logging.DATETIME_FORMAT, datefmt=DATEFMT, validate=False
            )
            logger.handlers[0].setFormatter(formatter)
        else:
            formatter = logging.Formatter(
                datadreamer_logging.STANDARD_FORMAT, datefmt=DATEFMT, validate=False
            )
            logger.handlers[0].setFormatter(formatter)
        effective_level = logging.INFO if self.log_level is None else self.log_level
        if self.verbose:
            logger.setLevel((min(logging.DEBUG, effective_level)))
        elif self.verbose is False:
            logger.setLevel(logging.CRITICAL + 1)
        else:
            logger.setLevel(effective_level)

        if not DataDreamer.is_background_process() and not is_distributed():
            if self.output_folder_path:
                logger.info(
                    f"Initialized. 🚀 Dreaming to folder: {self.output_folder_path}"
                )
            else:
                logger.info("Initialized. 🚀 Dreaming in-memory: 🧠")

        # Take over HF loggers to prevent them from spewing logs
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        DataDreamer.ctx.hf_log = self.hf_log
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
        DataDreamer.ctx._hf_optimum_verbosity = optimum_logging.get_verbosity()
        DataDreamer.ctx._hf_huggingface_hub_verbosity = (
            huggingface_hub_logging.get_verbosity()
        )
        DataDreamer.ctx._setfit_verbosity = setfit_logging.get_verbosity()
        DataDreamer.ctx._setfit_prog_bar = setfit_logging.is_progress_bar_enabled()
        DataDreamer.ctx._sentence_transformers_verbosity = logging.getLogger(
            "sentence_transformers"
        ).level
        DataDreamer.ctx._transformers_trainer_verbosity = logging.getLogger(
            "transformers.trainer"
        ).level
        DataDreamer.ctx._torch_distributed_verbosity = logging.getLogger(
            "torch.distributed"
        ).level
        DataDreamer.ctx._torch_distributed_multiprocessing_verbosity = (
            logging.getLogger("torch.distributed.elastic.multiprocessing.api").level
        )
        if not DataDreamer.ctx.hf_log:
            try:  # pragma: no cover
                from huggingface_hub.utils import _token

                _token._CHECK_GOOGLE_COLAB_SECRET = False
            except (ModuleNotFoundError, ImportError):  # pragma: no cover
                pass
            DataDreamer._disable_hf_datasets_logging()
            DataDreamer._disable_hf_transformers_logging()
            DataDreamer._disable_hf_optimum_logging()
            DataDreamer._disable_hf_huggingface_hub_logging()
            DataDreamer._disable_setfit_logging()
            logging.getLogger("sentence_transformers").level = logging.ERROR
            logging.getLogger("torch.distributed").level = logging.ERROR
            logging.getLogger(
                "torch.distributed.elastic.multiprocessing.api"
            ).level = logging.ERROR

        self._patch_logger("datasets", " 🤗 Datasets")
        self._patch_logger("optimum", " 🤗 Optimum")
        self._patch_logger("accelerate", " 🤗 Accelerate")
        self._patch_logger("evaluate", " 🤗 Evaluate")
        self._patch_logger("transformers", " 🤗 Transformers")
        self._patch_logger("peft", " 🤗 PEFT")
        self._patch_logger("petals", " 🌸 Petals")
        self._patch_logger("huggingface_hub", " 🤗 HF Hub")
        self._patch_logger("vllm", "vLLM")
        self._patch_logger("litellm", "🚅 LiteLLM")
        self._patch_logger("sentence_transformers", "Sentence Transformers")
        self._patch_logger("InstructorEmbedding", "Instructor")
        self._patch_logger("trl", " 🤗 TRL")
        self._patch_logger("setfit", "SetFit")
        self._patch_logger("torch", " 🔥 PyTorch")
        self._patch_logger("torch.distributed", " 🔥 PyTorch Distributed")
        self._patch_logger(
            "torch.distributed.elastic.multiprocessing.api", " 🔥 PyTorch Distributed"
        )
        self._patch_tqdm()

        # Set initialized to True
        DataDreamer.ctx.instance = self
        DataDreamer.ctx.initialized = True

    def __exit__(self, exc_type, exc_value, exc_tb):
        for cleanup_func in DataDreamer.ctx.cleanup_funcs:
            cleanup_func()
        if not DataDreamer.ctx.hf_log:
            DataDreamer._enable_hf_datasets_logging(logs=True, progress_bars=True)
            DataDreamer._enable_hf_transformers_logging(logs=True, progress_bars=True)
            DataDreamer._enable_hf_optimum_logging(logs=True)
            DataDreamer._enable_hf_huggingface_hub_logging(logs=True)
            logging.getLogger(
                "sentence_transformers"
            ).level = DataDreamer.ctx._sentence_transformers_verbosity
            logging.getLogger(
                "torch.distributed"
            ).level = DataDreamer.ctx._torch_distributed_verbosity
            logging.getLogger(
                "torch.distributed.elastic.multiprocessing.api"
            ).level = DataDreamer.ctx._torch_distributed_multiprocessing_verbosity
            DataDreamer._enable_hf_huggingface_hub_logging(logs=True)
            DataDreamer._enable_setfit_logging(logs=True)
            logging.getLogger(
                "transformers.trainer"
            ).level = DataDreamer.ctx._transformers_trainer_verbosity

        self._unpatch_loggers()
        self._unpatch_tqdm()
        processes_to_terminate = DataDreamer.ctx.background_processes
        DataDreamer.ctx = UserDict()
        if self.output_folder_path:
            logger.info(f"Done. ✨ Results in folder: {self.output_folder_path}")
        else:
            if self._hf_datasets_caching:
                enable_caching()
            logger.info("Done. ✨")
        _DATADREAMER_CTX_LOCK.release()
        for process in processes_to_terminate:
            if process.is_alive():
                process.terminate()  # pragma: no cover

    def start(self):  # pragma: no cover
        """
        Starts a DataDreamer session. This is an alternative to using a Python context
        manager. Using the context manager, however, is recommended and preferred.
        This method might be useful if you want to run DataDreamer in an interactive
        environment where a ``with`` block is not possible or cumbersome.
        """
        self.__enter__()

    def stop(self):  # pragma: no cover
        """
        Stops a DataDreamer session. This is an alternative to using a Python context
        manager. Using the context manager, however, is recommended and preferred.
        This method might be useful if you want to run DataDreamer in an interactive
        environment where a ``with`` block is not possible or cumbersome.
        """
        self.__exit__(None, None, None)

    def __getstate__(self):
        state = self.__dict__.copy()

        # Clear any patched_loggers (not serializable, these are set again in the
        # background process)
        state["patched_loggers"] = {}

        return state

    def __setstate__(self, state):  # pragma: no cover
        self.__dict__.update(state)
