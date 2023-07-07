import os


def get_tf_devices():
    """Returns a list of TensorFlow accelerator devices.

    Returns:
        list[Any]: A list of TensorFlow accelerator devices.
    """
    import tensorflow as tf

    if os.environ.get("PROJECT_ACCELERATOR_TYPE", None) == "cuda":
        gpus = tf.config.list_physical_devices("GPU")
        return [gpu.name for gpu in gpus]
    elif os.environ.get("PROJECT_ACCELERATOR_TYPE", None) == "tpu":
        tpus = tf.config.list_physical_devices("TPU")
        return [tpu.name for tpu in tpus]
    else:
        return []


def get_tf_cpu_device():
    """Returns the TensorFlow CPU device.

    Returns:
        Any: The TensorFlow CPU device.
    """
    import tensorflow as tf

    return tf.device("/device:cpu:0")


def get_tf_device(rank=0, fallback_to_cpu=True):
    """Gets the `rank` TensorFlow accelerator device, potentially falling back to the
    TensorFlow CPU device if that accelerator device cannot be found.

    Args:
        rank (int, optional): Which TensorFlow accelerator device to get. Defaults to 0.
        fallback_to_cpu (bool, optional): Default to using the TensorFlow CPU device
            if the specified TensorFlow accelerator device cannot be found. Defaults
            to True.

    Raises:
        RuntimeError: Thrown if the specified TensorFlow accelerator device cannot be
            found.

    Returns:
        Any: The TensorFlow device.
    """
    devices = get_tf_devices()
    if rank < len(devices):
        return devices[rank]
    else:
        if fallback_to_cpu:
            return get_tf_cpu_device()
        else:
            raise RuntimeError(f"No TensorFlow device found for rank {rank}")


def get_torch_devices():
    """Returns a list of PyTorch accelerator devices.

    Returns:
        list[device]: A list of PyTorch accelerator devices.
    """
    import torch

    if os.environ.get("PROJECT_ACCELERATOR_TYPE", None) == "cuda":
        return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    elif os.environ.get("PROJECT_ACCELERATOR_TYPE", None) == "tpu":
        import torch_xla

        return torch_xla.core.xla_model.get_xla_supported_devices(devkind="TPU")
    else:
        return []


def get_torch_cpu_device():
    """Returns the PyTorch CPU device.

    Returns:
        device: The PyTorch CPU device.
    """
    import torch

    return torch.device("cpu")


def get_torch_device(rank=0, fallback_to_cpu=True):
    """Gets the `rank` PyTorch accelerator device, potentially falling back to the
    PyTorch CPU device if that accelerator device cannot be found.

    Args:
        rank (int, optional): Which PyTorch accelerator device to get. Defaults to 0.
        fallback_to_cpu (bool, optional): Default to using the PyTorch CPU device
            if the specified PyTorch accelerator device cannot be found. Defaults
            to True.

    Raises:
        RuntimeError: Thrown if the specified PyTorch accelerator device cannot be
            found.

    Returns:
        Any: The PyTorch device.
    """
    devices = get_torch_devices()
    if rank < len(devices):
        return devices[rank]
    else:
        if fallback_to_cpu:
            return get_torch_cpu_device()
        else:
            raise RuntimeError(f"No PyTorch device found for rank {rank}")


def get_jax_devices():
    """Returns a list of JAX accelerator devices.

    Returns:
        list[Device]: A list of JAX accelerator devices.
    """
    import jax

    if os.environ.get("PROJECT_ACCELERATOR_TYPE", None) == "cuda":
        return jax.devices("gpu")
    elif os.environ.get("PROJECT_ACCELERATOR_TYPE", None) == "tpu":
        return jax.devices("tpu")
    else:
        return []


def get_jax_cpu_device():
    """Returns the JAX CPU device.

    Returns:
        Device: The JAX CPU device.
    """
    import jax

    return jax.devices("cpu")[0]


def get_jax_device(rank=0, fallback_to_cpu=True):
    """Gets the `rank` JAX accelerator device, potentially falling back to the
    JAX CPU device if that accelerator device cannot be found.

    Args:
        rank (int, optional): Which JAX accelerator device to get. Defaults to 0.
        fallback_to_cpu (bool, optional): Default to using the JAX CPU device
            if the specified JAX accelerator device cannot be found. Defaults
            to True.

    Raises:
        RuntimeError: Thrown if the specified JAX accelerator device cannot be
            found.

    Returns:
        Any: The JAX device.
    """
    devices = get_jax_devices()
    if rank < len(devices):
        return devices[rank]
    else:
        if fallback_to_cpu:
            return get_jax_cpu_device()
        else:
            raise RuntimeError(f"No JAX device found for rank {rank}")
