import os

import pytest
import torch

from ...utils.device_utils import (
    device_id_to_true_device_id,
    device_to_device_id,
    get_device_env_variables,
    get_true_device_ids,
    is_cpu_device,
)


class TestDeviceUtils:
    def test_is_cpu_device(self):
        assert is_cpu_device(None)
        assert is_cpu_device(-1)
        assert is_cpu_device("cpu")
        assert is_cpu_device("cpu:0")
        assert is_cpu_device(torch.device("cpu"))
        assert is_cpu_device(torch.device("cpu:0"))
        assert not is_cpu_device(0)
        assert not is_cpu_device(1)
        assert not is_cpu_device(torch.device("cuda:0"))
        assert not is_cpu_device(torch.device("cuda:1"))
        assert not is_cpu_device("cuda")
        assert not is_cpu_device("mps")
        assert not is_cpu_device("cuda:1")

    def test_device_to_device_id(self):
        assert device_to_device_id(None) is None  # type:ignore[arg-type]
        assert device_to_device_id(-1) is None
        assert device_to_device_id("cpu") is None
        assert device_to_device_id("cpu:0") is None
        assert device_to_device_id(torch.device("cpu")) is None
        assert device_to_device_id(0) == 0
        assert device_to_device_id(1) == 1
        assert device_to_device_id(torch.device("cuda")) == 0
        assert device_to_device_id(torch.device("cuda:0")) == 0
        assert device_to_device_id(torch.device("cuda:1")) == 1
        assert device_to_device_id(torch.device("mps")) == 0
        assert device_to_device_id(torch.device("mps:0")) == 0
        assert device_to_device_id(torch.device("mps:1")) == 1
        assert device_to_device_id("cuda") == 0
        assert device_to_device_id("cuda:0") == 0
        assert device_to_device_id("cuda:1") == 1
        assert device_to_device_id("mps") == 0
        assert device_to_device_id("mps:0") == 0
        assert device_to_device_id("mps:1") == 1

    def test_device_id_to_true_device_id(self):
        assert device_id_to_true_device_id(-1) is None
        assert device_id_to_true_device_id(0) == 0
        assert device_id_to_true_device_id(1) == 1
        assert device_id_to_true_device_id(2) == 2
        assert device_id_to_true_device_id(999999) is None
        os.environ["CUDA_VISIBLE_DEVICES"] = "6,4,3"
        assert device_id_to_true_device_id(-1) is None
        assert device_id_to_true_device_id(0) == 6
        assert device_id_to_true_device_id(1) == 4
        assert device_id_to_true_device_id(2) == 3
        assert device_id_to_true_device_id(3) is None
        assert device_id_to_true_device_id(999999) is None

    def test_get_true_device_ids(self):
        assert get_true_device_ids([0, 1, 2]) == (False, [0, 1, 2])
        assert get_true_device_ids([0, 2, 999999, 0, 1, -1, -1]) == (True, [0, 2, 1])
        assert get_true_device_ids([2, 1, 999999, 2, 2, -1, -1]) == (True, [2, 1])
        assert get_true_device_ids([2, 1, 999999, 2, 2]) == (False, [2, 1])
        os.environ["CUDA_VISIBLE_DEVICES"] = "6,4,3"
        assert get_true_device_ids([0, 1, 2]) == (False, [6, 4, 3])
        assert get_true_device_ids([0, 2, 999999, 0, 1, -1, -1]) == (True, [6, 3, 4])
        assert get_true_device_ids([2, 1, 999999, 2, 2, -1, -1]) == (True, [3, 4])
        assert get_true_device_ids([2, 1, 999999, 2, 2]) == (False, [3, 4])
        assert get_true_device_ids(
            [
                torch.device(0),
                torch.device(2),
                torch.device(999999),
                torch.device(0),
                torch.device(1),
            ]
        ) == (False, [6, 3, 4])

    def test_get_device_env_variables(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "6,4,3"
        with pytest.raises(AssertionError):
            get_device_env_variables([0, 2, 999999, 0, 1, -1, -1])
        with pytest.raises(AssertionError):
            get_device_env_variables([0, 2, 0, 1])
        assert get_device_env_variables([0, 2, 1]) == {"CUDA_VISIBLE_DEVICES": "6,3,4"}
