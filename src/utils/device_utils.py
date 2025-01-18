import base64
import os
import re
from functools import cached_property
from typing import TYPE_CHECKING, Any, Sequence, Type, cast

import dill
import torch

from .. import DataDreamer
from .hf_model_utils import is_bnb_quantized
from .import_utils import ignore_transformers_warnings

with ignore_transformers_warnings():
    from transformers import TrainerCallback
    from transformers.utils.quantization_config import QuantizationConfigMixin

if TYPE_CHECKING:  # pragma: no cover
    from ..trainers._train_hf_base import _TrainHFBase


def validate_device(
    device: None | int | str | torch.device | list[int | str | torch.device],
) -> None | int | str | torch.device | list[int | str | torch.device]:
    if isinstance(device, list):  # pragma: no cover
        use_cpu_as_backup, true_device_ids = get_true_device_ids(device)
        if len(true_device_ids) == 0:
            if use_cpu_as_backup:
                device = "cpu"
            else:
                raise RuntimeError(
                    f"The device list you specified ({device}) could not be found on this system."
                )
    if isinstance(device, list) and len(device) == 1:  # pragma: no cover
        device = device[0]
    return device


def is_cpu_device(device: None | int | str | torch.device) -> bool:
    return (
        device is None
        or device == -1
        or (isinstance(device, str) and device.lower().strip().startswith("cpu"))
        or (
            isinstance(device, torch.device)
            and isinstance(device.type, str)
            and device.type.lower().strip() == "cpu"
        )
    )


def device_to_device_id(device: None | int | str | torch.device) -> None | int:
    if is_cpu_device(device):
        return None
    if isinstance(device, str):
        search_result = re.search(r"\d+", device)  # Get integer from str
        return 0 if search_result is None else int(search_result.group())
    elif isinstance(device, torch.device):
        return device.index or 0
    else:
        return device


def device_id_to_true_device_id(device_id: int) -> None | int | str:
    visible_device_ids = torch.cuda._parse_visible_devices()
    if device_id >= 0 and device_id < len(visible_device_ids):
        return visible_device_ids[device_id]
    else:
        return None


def get_true_device_ids(
    devices: list[int | str | torch.device],
) -> tuple[bool, list[int | str]]:  # pragma: no cover
    device_ids = list(map(device_to_device_id, devices))
    use_cpu_as_backup = any(device_id is None for device_id in device_ids)
    device_ids = [device_id for device_id in device_ids if device_id is not None]
    true_device_ids = [
        true_device_id
        for true_device_id in map(
            device_id_to_true_device_id, cast(list[int], device_ids)
        )
        if true_device_id is not None
    ]
    true_device_ids = sorted(set(true_device_ids), key=true_device_ids.index)
    return use_cpu_as_backup, true_device_ids


def get_device_env_variables(devices: list[int | str | torch.device]) -> dict[str, Any]:
    _, true_device_ids = get_true_device_ids(devices)
    assert (
        len(true_device_ids) == len(devices)
    ), f"The device list you specified ({devices}) is invalid (or devices could not be found)."
    device_env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, true_device_ids))}
    device_env["NCCL_P2P_DISABLE"] = "1"
    return device_env


def memory_usage_format(num, suffix="B"):  # pragma: no cover
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


class __TrainingArgumentDeviceOverrideMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def place_model_on_device(self):
        from .distributed_utils import is_distributed

        if is_distributed():  # pragma: no cover
            return True
        else:
            return False

    @property
    def n_gpu(self):
        return min(
            super().n_gpu,  # type:ignore[misc]
            len(self._selected_device)  # type:ignore[attr-defined]
            if isinstance(self._selected_device, list)  # type:ignore[attr-defined]
            else 1,
        )

    @property
    def device(self) -> torch.device:
        # super().device has a side-effect (it runs code due to @property)
        # (don't move or delete unless you are sure)
        super_device = super().device  # type:ignore[misc]
        if isinstance(self._selected_device, list):  # type:ignore[attr-defined]
            return super_device  # pragma: no cover
        else:
            from accelerate import PartialState

            if is_cpu_device(self._selected_device):  # type:ignore[attr-defined]
                os.environ["ACCELERATE_USE_CPU"] = "true"
                PartialState().device = torch.device("cpu")
            else:  # pragma: no cover
                os.environ["ACCELERATE_TORCH_DEVICE"] = str(
                    torch.device(self._selected_device)  # type:ignore[attr-defined]
                )
                PartialState().device = torch.device(self._selected_device)  # type:ignore[attr-defined]
            return (
                torch.device("cpu")
                if is_cpu_device(self._selected_device)  # type:ignore[attr-defined]
                else torch.device(self._selected_device)  # type:ignore[attr-defined]
            )


class _SentenceTransformerTrainingArgumentDeviceOverrideMixin(
    __TrainingArgumentDeviceOverrideMixin
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def _selected_device(self):
        dill.loads(
            base64.b64decode(os.environ["_DATADREAMER_SETFIT_DEVICE"].encode("utf8"))
        )


class _TrainingArgumentDeviceOverrideMixin(__TrainingArgumentDeviceOverrideMixin):
    def __init__(self, *args, **kwargs):
        from .distributed_utils import apply_distributed_config

        kwargs = apply_distributed_config(self, kwargs)
        super().__init__(*args, **kwargs)


def get_device_memory_monitoring_callback(trainer: "_TrainHFBase") -> Type:
    from .distributed_utils import (
        get_current_accelerator,
        get_global_rank,
        get_local_rank,
        get_local_world_size,
        is_distributed,
    )

    class DeviceMemoryMonitoringCallback(TrainerCallback):
        def _log_device_memory_usage(self):
            if is_distributed():  # pragma: no cover
                DataDreamer.ctx.distributed_pipe.put(
                    (get_global_rank(), torch.cuda.memory_allocated(get_global_rank()))
                )
                if get_local_rank() == 0:
                    get_current_accelerator().wait_for_everyone()
                    device_memory_usage = dict(
                        map(
                            lambda x: (f"Worker #{x[0]}", memory_usage_format(x[1])),
                            sorted(
                                [
                                    DataDreamer.ctx.distributed_pipe.get()
                                    for _ in range(get_local_world_size())
                                ]
                            ),
                        )
                    )
                    trainer.logger.debug(
                        f"Device Memory Usage -- {device_memory_usage}"
                    )
                else:  # pragma: no cover
                    get_current_accelerator().wait_for_everyone()
            elif not isinstance(trainer.device, list):
                if (
                    not is_cpu_device(trainer.device) and torch.cuda.is_available()
                ):  # pragma: no cover
                    device_memory_usage = {
                        f"Device #{device_idx}": memory_usage_format(
                            torch.cuda.memory_allocated(device_idx)
                        )
                        for device_idx in range(torch.cuda.device_count())
                        if device_to_device_id(trainer.device) == device_idx
                    }
                    trainer.logger.debug(
                        f"Device Memory Usage -- {device_memory_usage}"
                    )

        def on_train_begin(self, args, state, control, **kwargs):
            self._log_device_memory_usage()

        def on_epoch_end(self, args, state, control, **kwargs):
            self._log_device_memory_usage()

        def on_train_end(self, args, state, control, **kwargs):
            self._log_device_memory_usage()

    return DeviceMemoryMonitoringCallback


def model_to_device(
    device: None | int | str | torch.device | list[int | str | torch.device],
    device_map: None | dict | str,
    is_train: bool,
    model_name: str,
    revision: None | str,
    trust_remote_code: bool,
    quantization_config: None | QuantizationConfigMixin | dict,
) -> tuple[None | int | str | torch.device, None | dict | str, None | dict[int, int]]:
    from .distributed_utils import get_global_rank

    to_device: None | int | str | torch.device = None
    to_device_map: None | dict | str = None
    to_device_map_max_memory: None | dict[int, int] = None

    if device_map is None:
        if (not is_train and isinstance(device, list)) or is_bnb_quantized(
            model_name=model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
        ):  # pragma: no cover
            if is_train:
                to_device_map = {
                    "": get_global_rank() if isinstance(device, list) else device
                }
                to_device_map_max_memory = None
            else:
                list_of_devices: Sequence[None | int | str | torch.device] = (
                    cast(list[None | int | str | torch.device], device)
                    if isinstance(device, list)
                    else [device]
                )
                list_of_device_ids = [device_to_device_id(d) for d in list_of_devices]
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        _ = torch.tensor([0], device=i)
                    max_memory = {
                        device_id: (
                            0
                            if device_id not in list_of_device_ids
                            else torch.cuda.mem_get_info(device_id)[0]
                        )
                        for device_id in range(torch.cuda.device_count())
                    }
                    to_device_map = "auto"
                    to_device_map_max_memory = max_memory
                else:
                    assert all(
                        is_cpu_device(d) for d in list_of_devices
                    ), f"The device you specified ({list_of_devices}) is invalid (or devices could not be found)."
                    to_device_map = {"": "cpu"}
                    to_device_map_max_memory = None
        else:
            to_device = "cpu" if isinstance(device, list) else device
    else:  # pragma: no cover
        to_device_map = device_map
        to_device_map_max_memory = None
    if to_device is None and to_device_map is None and to_device_map_max_memory is None:
        to_device = "cpu"
    return to_device, to_device_map, to_device_map_max_memory
