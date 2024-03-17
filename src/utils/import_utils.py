import contextlib
import importlib
import warnings
from types import ModuleType

# These Pydantic warnings happen way too much and needs to be done
# as a module-level ignore
warnings.filterwarnings("ignore", message="Pydantic V1 style .*")
warnings.filterwarnings("ignore", message="`pydantic.*")
warnings.filterwarnings("ignore", message="Support for class-based .*")


@contextlib.contextmanager
def ignore_transformers_warnings():
    # Filter these globally
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*byte fallback option.*",
        module="transformers.convert_slow_tokenizer",
    )

    # Filter these within the context manager
    try:
        from torch.storage import _warn_typed_storage_removal

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="TypedStorage.*"
            )
            _warn_typed_storage_removal()
    except ImportError:  # pragma: no cover
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="distutils Version classes are deprecated.*",
            module="torch.utils.tensorboard",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The installed version of bitsandbytes was compiled without GPU.*",
            module="bitsandbytes.cextension",
        )
        yield None


@contextlib.contextmanager
def ignore_training_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="Passing the following arguments to.*",
            module="accelerate.accelerator",
        )
        yield None


@contextlib.contextmanager
def ignore_pydantic_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Pydantic V1 style .*")
        warnings.filterwarnings("ignore", message="`pydantic.*")
        warnings.filterwarnings("ignore", message="Support for class-based .*")
        yield None


@contextlib.contextmanager
def ignore_litellm_warnings():
    with ignore_pydantic_warnings():
        warnings.filterwarnings("ignore", message="Deprecated call to.*pkg_resources.*")
        warnings.filterwarnings("ignore", message="pkg_resources.*")
        warnings.filterwarnings("ignore", message="open_text is deprecated.*")
        yield None


@contextlib.contextmanager
def ignore_trl_warnings():
    with ignore_transformers_warnings():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The `optimize_cuda_cache`.*",
                module="trl.trainer.ppo_config",
            )
            yield None


@contextlib.contextmanager
def ignore_faiss_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="distutils Version classes are deprecated.*",
            module="faiss.loader",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="distutils Version classes are deprecated.*",
        )
        yield None


@contextlib.contextmanager
def ignore_hivemind_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="There is no current event loop.*",
            module="hivemind.utils.mpfuture",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The given NumPy array is not writable.*",
            module="hivemind.compression.base",
        )
        yield None


@contextlib.contextmanager
def ignore_torch_trainer_distributed_warnings():  # pragma: no cover
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*OMP_NUM_THREADS.*",
            module="accelerate.state",
        )
        yield None


def import_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        root_module_name = module_name.split(".")[0]
        raise ModuleNotFoundError(
            f"Please install `{root_module_name}` with"
            f" `pip3 install {root_module_name}`."
        ) from None
