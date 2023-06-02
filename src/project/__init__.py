import json
import os
import sys

from loguru import logger

from project.debug import bash, context, debugger  # noqa: F401
from project.devices import get_jax_cpu_device  # noqa: F401
from project.devices import get_jax_device  # noqa: F401
from project.devices import get_jax_devices  # noqa: F401
from project.devices import get_tf_cpu_device  # noqa: F401
from project.devices import get_tf_device  # noqa: F401
from project.devices import get_tf_devices  # noqa: F401
from project.devices import get_torch_cpu_device  # noqa: F401
from project.devices import get_torch_device  # noqa: F401
from project.devices import get_torch_devices  # noqa: F401
from project.persistent_storage import get_persistent_dir  # noqa: F401
from project.report import reporter  # noqa: F401
from project.serve import run_ngrok  # noqa: F401

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
    if "PROJECT_CLUSTER" in os.environ:
        with open(os.environ["PROJECT_ARGS_FILE"], "w+") as f:
            f.write(json.dumps(sys.argv, indent=2))
