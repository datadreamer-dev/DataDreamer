import logging
import os
import sys
from logging import Logger
from multiprocessing import get_context
from typing import Any, Callable

import dill
import torch
import torch.cuda

from .. import DataDreamer
from ..logging import logger as datadreamer_logger
from .arg_utils import Default, default_to
from .background_utils import (
    get_parent_process_context,
    get_thread_id,
    restore_parent_process_context,
)
from .device_utils import get_device_env_variables
from .hf_model_utils import get_base_model_from_peft_model, is_peft_model
from .import_utils import (
    ignore_torch_trainer_distributed_warnings,
    ignore_transformers_warnings,
)

with ignore_transformers_warnings():
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedModel


def default_fsdp_config(model: PreTrainedModel) -> dict[str, Any]:  # pragma: no cover
    if is_peft_model(model):
        model = get_base_model_from_peft_model(model)
    if isinstance(model, SentenceTransformer):
        model = model[0].auto_model
    no_split_modules = (
        model._no_split_modules
        if getattr(model, "_no_split_modules", None) is not None
        else []
    )
    if len(no_split_modules) == 0:  # The model doesn't define any no_split_modules
        no_split_modules = [model.__class__.__name__]
    peft_modules = ["PrefixEncoder", "PromptEncoder", "PromptEmbedding"]
    transformer_layer_cls_to_wrap = no_split_modules + peft_modules
    named_modules = str([p[1].__class__.__name__ for p in model.named_modules()])
    transformer_layer_cls_to_wrap = list(
        filter(lambda x: x in named_modules, transformer_layer_cls_to_wrap)
    )
    return {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "backward_prefetch": "backward_pre",
            "transformer_layer_cls_to_wrap": transformer_layer_cls_to_wrap,
        },
    }


def apply_distributed_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = kwargs.copy()
    _device = kwargs.pop("_device")
    self._selected_device = _device
    _model = kwargs.pop("_model")
    default_fsdp_kwargs = (
        default_fsdp_config(_model)
        if isinstance(_device, list)
        else {"fsdp": "", "fsdp_config": None}
    )
    fsdp = default_to(kwargs.pop("fsdp"), default_fsdp_kwargs["fsdp"])
    fsdp_is_enabled = (
        (isinstance(fsdp, str) and len(fsdp.strip()) > 0)
        or (isinstance(fsdp, list) and len(fsdp) > 0)
        or (isinstance(fsdp, bool) and fsdp)
    )
    distributed_kwargs = {
        "fsdp": fsdp,
        "fsdp_config": default_to(
            kwargs.pop("fsdp_config"),
            default_fsdp_kwargs["fsdp_config"] if fsdp_is_enabled else None,
        ),
    }
    if not fsdp_is_enabled and isinstance(_device, list):  # pragma: no cover
        distributed_kwargs["ddp_find_unused_parameters"] = False
    kwargs.update(distributed_kwargs)
    return kwargs


def is_distributed():
    return int(os.environ.get("DATADREAMER_DISTRIBUTED", -1)) == 1


def get_global_rank() -> int:  # pragma: no cover
    rank = os.environ.get("RANK", -1)
    return int(rank)


def not_distributed_or_main_process():
    return not is_distributed() or get_global_rank() == 0


def not_main_process():
    return is_distributed() and get_global_rank() > 0


def get_local_rank() -> int:  # pragma: no cover
    rank = os.environ.get("LOCAL_RANK", -1)
    return int(rank)


def get_local_world_size() -> int:  # pragma: no cover
    world_size = os.environ.get("LOCAL_WORLD_SIZE", -1)
    return int(world_size)


def _init_worker(
    pipe, logger: Logger, parent_thread_id: tuple[int, int]
):  # pragma: no cover
    DataDreamer.ctx.distributed_pipe = pipe
    DataDreamer.ctx.pid = os.getpid()
    DataDreamer._register_child_thread(parent_thread_id)
    if logger.level > logging.DEBUG and get_global_rank() != 0:
        hf_transformers_trainer_logger = logging.getLogger("transformers.trainer")

        class NoCheckpointModelMissingKeysWarningFilter(logging.Filter):
            def filter(self, record):
                return not record.getMessage().startswith("There were missing keys")

        hf_transformers_trainer_logger.addFilter(
            NoCheckpointModelMissingKeysWarningFilter()
        )

    logger.info(
        f"Initialized worker #{get_global_rank()} in the distributed environment."
    )


def _exit_worker(logger: Logger):  # pragma: no cover
    logger.info(f"Exiting worker #{get_global_rank()} in the distributed environment.")


def _global_func_wrapper(func, *args, **kwargs):  # pragma: no cover
    logging.getLogger("torch.distributed").level = logging.ERROR
    logging.getLogger(
        "torch.distributed.elastic.multiprocessing.api"
    ).level = logging.ERROR
    return dill.loads(func)(*args, **kwargs)


def validate_distributed_config(
    distributed_config: dict[str, Any] | Default,
) -> dict[str, Any]:
    # Setup the distributed config
    return default_to(distributed_config, {}) or {}


def get_node_rank_from_distributed_config(
    distributed_config: dict[str, Any],
) -> int:  # pragma: no cover
    return int(distributed_config.get("node_rank", 0))


def get_num_nodes_from_distributed_config(
    distributed_config: dict[str, Any],
) -> int:  # pragma: no cover
    return int(distributed_config.get("nnodes", 1))


def run_distributed(
    distributed_config: dict[str, Any],
    devices: list[int | str | torch.device],
    func: Callable,
    args: tuple[Any, ...],
    logger: None | Logger = None,
):  # pragma: no cover
    # Get Logger
    final_logger = logger or datadreamer_logger

    # Get true device IDs
    device_env = get_device_env_variables(devices)

    # Wrap the function that will be run in a spawned process
    def func_wrapper(pipe, parent_context, env, parent_thread_id, args):
        # Restore parent process context
        env.update(
            {"DATADREAMER_BACKGROUND_PROCESS": "0", "DATADREAMER_DISTRIBUTED": "1"}
        )
        env.update(device_env)
        restore_parent_process_context(parent_context=parent_context, env=env)

        # Initialized
        _init_worker(pipe=pipe, logger=final_logger, parent_thread_id=parent_thread_id)

        # Run the wrapped function
        with ignore_torch_trainer_distributed_warnings():
            func(*dill.loads(args))
        _exit_worker(logger=final_logger)

    # Configure options for distributed process spawns
    def configure_and_launch():
        from torch.distributed.launcher.api import elastic_launch
        from torch.distributed.run import config_from_args, parse_args

        # Create launch configuration
        config, _, _ = config_from_args(parse_args(["no_op.py"]))

        # Setup multi-node launch configuration parameters
        config.min_nodes = distributed_config.pop("nnodes", 1)
        config.max_nodes = distributed_config.pop("nnodes", 1)
        if "node_rank" in distributed_config:
            config.rdzv_configs["rank"] = distributed_config.pop("node_rank")
        if "master_addr" in distributed_config:
            master_addr = distributed_config.pop("master_addr")
            master_port = distributed_config.pop("master_port")
            config.rdzv_endpoint = f"{master_addr}:{master_port}"

        # Setup normal configuration parameters
        nproc_per_node = distributed_config.pop("nproc_per_node", len(devices))
        for k, v in distributed_config.items():
            assert hasattr(
                config, k
            ), f"{k} is not a valid configuration option for PyTorch distributed launching."
            setattr(config, k, v)
        config.start_method = "spawn"
        config.nproc_per_node = nproc_per_node

        # Create a communication pipe
        spawn_context = get_context(method="spawn")
        pipe: Any = spawn_context.Queue(2)

        # Launch the spawned child processes (share the parent context with them)
        if final_logger.level > logging.DEBUG:
            torch_distributed_multiprocessing_logger = logging.getLogger(
                "torch.distributed.elastic.multiprocessing.api"
            )

            class NoProcessFailedErrorFilter(logging.Filter):
                def filter(self, record):
                    return record.levelno != logging.ERROR

            torch_distributed_multiprocessing_logger.addFilter(
                NoProcessFailedErrorFilter()
            )
        elastic_launch(config=config, entrypoint=_global_func_wrapper)(
            dill.dumps(func_wrapper),
            pipe,
            get_parent_process_context(),
            os.environ.copy(),
            get_thread_id(),
            dill.dumps(args),
        )

    # Launch
    configure_and_launch()

    # Exit if any node other than 0
    if (
        get_node_rank_from_distributed_config(distributed_config) > 0
    ):  # pragma: no cover
        datadreamer_logger.info(
            f"Exiting node #{int(distributed_config['node_rank'])}."
            " See the main node for results."
        )
        sys.exit(0)


def save_distributed_model(
    trainer, accelerator, fsdp, peft, model, output_dir
):  # pragma: no cover
    MODEL_DIR = output_dir
    from ..trainers.train_sentence_transformer import SentenceTransformerLossWrapper

    # Check if sentence transformers
    #
    # There are special cases for Sentence Transformers models because of the wrapper.
    is_sentence_transformers = fsdp and isinstance(
        model._fsdp_wrapped_module, SentenceTransformerLossWrapper
    )

    if fsdp and not peft:
        # Save an FSDP model
        from accelerate.utils import save_fsdp_model
        from accelerate.utils.constants import FSDP_MODEL_NAME

        # Save the model weights
        if is_sentence_transformers:
            weights_module = model._fsdp_wrapped_module.wrapped_model.model[
                0
            ].auto_model
        else:
            weights_module = model
        save_fsdp_model(
            accelerator.state.fsdp_plugin, accelerator, weights_module, MODEL_DIR
        )

        # Only on the main process
        if get_global_rank() == 0:
            # Rename the weights to a more standard name that can be loaded
            os.rename(
                os.path.join(MODEL_DIR, f"{FSDP_MODEL_NAME}.bin"),
                os.path.join(MODEL_DIR, "pytorch_model.bin"),
            )

            # Finally save the model's configuration files (JSON files)
            model = model._fsdp_wrapped_module
            if is_sentence_transformers:
                # SentenceTransformer doesn't have a way to just save the config
                # so we temporarily remove the main transformer module's ability to
                # save
                main_sentence_transformer_module = model.wrapped_model.model[
                    0
                ].auto_model
                orig_save_pretrained = main_sentence_transformer_module.save_pretrained
                main_sentence_transformer_module.save_pretrained = (
                    lambda *args, **kwargs: None
                )
                trainer._save_resource(
                    main_sentence_transformer_module.config, MODEL_DIR
                )
                trainer._save_resource(model, MODEL_DIR)
                main_sentence_transformer_module.save_pretrained = orig_save_pretrained
            else:
                trainer._save_resource(model.config, MODEL_DIR)
    else:
        if is_sentence_transformers:
            if fsdp and peft:
                model = model._fsdp_wrapped_module
        trainer._save_resource(model, MODEL_DIR)
