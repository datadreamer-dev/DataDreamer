import contextlib
import importlib
import warnings
from types import ModuleType
from unittest import mock

from tqdm.auto import tqdm

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
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="You are using `torch.load` with `weights_only=False`.*",
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
        warnings.filterwarnings(
            "ignore", category=UserWarning, message=".*use_reentrant.*"
        )
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*torch.cpu.amp.autocast.*"
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*FSDP.state_dict_type.*deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Merge.*may get different generations due to rounding error.*",
        )
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="To copy construct from a tensor.*"
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="You are using `torch.load` with `weights_only=False`.*",
        )
        yield None


@contextlib.contextmanager
def ignore_inference_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="`do_sample` is set to `False`.*"
        )
        yield None


@contextlib.contextmanager
def ignore_pydantic_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Valid config keys have changed .*")
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
        warnings.filterwarnings(
            "ignore", message="Use 'content=<...>' to upload raw bytes/text content."
        )
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
            warnings.filterwarnings(
                "ignore", category=FutureWarning, message="`tokenizer` is deprecated .*"
            )
            yield None


@contextlib.contextmanager
def ignore_setfit_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="You are using `torch.load` with `weights_only=False`.*",
        )
        yield None


@contextlib.contextmanager
def ignore_hf_token_warnings():  # pragma: no cover
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
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
def ignore_hivemind_warnings():  # pragma: no cover
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
def ignore_tqdm():  # pragma: no cover
    original_tqdm_init = tqdm.__init__

    def mock_tqdm_init(self, *args, **kwargs):
        kwargs["disable"] = True
        original_tqdm_init(self, *args, **kwargs)

    with mock.patch("tqdm.auto.tqdm.__init__", new=mock_tqdm_init):
        yield None


@contextlib.contextmanager
def ignore_torch_trainer_distributed_warnings():  # pragma: no cover
    with ignore_training_warnings():
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
