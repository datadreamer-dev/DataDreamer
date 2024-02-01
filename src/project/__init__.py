"""``project`` provides project-wide helpers and utilities useful in machine learning projects.

Attributes:
    INITIAL_CWD (None | str): The initial current working directory path.
    context (dict): A dictionary to use to store global context.
    RUNNING_IN_PYTEST (bool): Whether or not the project is running in ``pytest``.
    RUNNING_IN_CLUSTER (bool): Whether or not the project is running on a cluster.
"""

import json
import os
import sys

from loguru import logger

from .debug import bash, context, debugger
from .devices import (
    get_jax_cpu_device,
    get_jax_device,
    get_jax_devices,
    get_tf_cpu_device,
    get_tf_device,
    get_tf_devices,
    get_torch_cpu_device,
    get_torch_device,
    get_torch_devices,
)
from .environment import RUNNING_IN_CLUSTER, RUNNING_IN_PYTEST
from .persistent_storage import get_persistent_dir
from .report import reporter  # type:ignore[attr-defined]
from .serve import run_ngrok

# Initial cwd (defined in __main__.py)
INITIAL_CWD: None | str = None

# Make sure CUDA/NVIDIA_VISIBLE_DEVICES is set if it is needed
if os.environ.get("PROJECT_ACCELERATOR_TYPE", None) == "cuda":
    if "PROJECT_VISIBLE_ACCELERATOR_DEVICES" in os.environ:
        os.environ["NVIDIA_VISIBLE_DEVICES"] = os.environ[
            "PROJECT_VISIBLE_ACCELERATOR_DEVICES"
        ]
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ[
            "PROJECT_VISIBLE_ACCELERATOR_DEVICES"
        ]

# Make sure if CUDA/NVIDIA_VISIBLE_DEVICES is set, PROJECT_*_ACCELERATOR_* is set
if (
    "CUDA_VISIBLE_DEVICES" in os.environ
    and os.environ.get("PROJECT_ACCELERATOR_TYPE", None) is None
):
    os.environ["PROJECT_ACCELERATOR_TYPE"] = "cuda"
    os.environ["PROJECT_VISIBLE_ACCELERATOR_DEVICES"] = os.environ[
        "CUDA_VISIBLE_DEVICES"
    ]
elif (
    "NVIDIA_VISIBLE_DEVICES" in os.environ
    and os.environ.get("PROJECT_ACCELERATOR_TYPE", None) is None
):
    os.environ["PROJECT_ACCELERATOR_TYPE"] = "cuda"
    os.environ["PROJECT_VISIBLE_ACCELERATOR_DEVICES"] = os.environ[
        "NVIDIA_VISIBLE_DEVICES"
    ]


def init():
    """Initializes the project. Adds logging and does any other project setup."""
    # Setup logger
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<bold>[{process}]</bold> | <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",  # noqa: B950
    )

    # Write args
    if RUNNING_IN_CLUSTER:
        with open(os.environ["PROJECT_ARGS_FILE"], "w+") as f:
            f.write(json.dumps(sys.argv, indent=2))


__all__ = [
    "RUNNING_IN_CLUSTER",
    "RUNNING_IN_PYTEST",
    "bash",
    "context",
    "debugger",
    "get_jax_cpu_device",
    "get_jax_device",
    "get_jax_devices",
    "get_tf_cpu_device",
    "get_tf_device",
    "get_tf_devices",
    "get_torch_cpu_device",
    "get_torch_device",
    "get_torch_devices",
    "get_persistent_dir",
    "reporter",
    "run_ngrok",
    "init",
]
