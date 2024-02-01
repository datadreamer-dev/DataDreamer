import hashlib
import json
import os

from .report import reporter  # type:ignore[attr-defined]


def _dict_hash(dictionary):
    """Returns the MD5 hash of a dictionary.

    Args:
        dictionary (dict[str, Any]): The dictionary to hash.

    Returns:
        str: The MD5 hash.
    """
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_persistent_dir(name, config_path):
    """Returns the path to a persistent directory that will be usable across jobs. Any
    future jobs

    Args:
        name (str): [description]
        config_path (str): [description]
    """
    config_hash = _dict_hash(reporter.get(config_path))
    persistent_dir = os.path.join(
        os.environ["PROJECT_DATA_OUTPUT_PERSISTENT_DATA"], config_hash
    )
    local_persistent_dir = os.path.join(os.environ["PROJECT_WRITE_DIR"], name)
    os.makedirs(persistent_dir, exist_ok=True)
    try:
        os.symlink(persistent_dir, local_persistent_dir, target_is_directory=True)
    except FileExistsError:
        pass
    return local_persistent_dir
