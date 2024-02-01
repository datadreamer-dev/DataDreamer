import os
import shutil
from typing import Any


def mkdir(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except FileExistsError:  # pragma: no cover
        pass


def safe_fn(value: str, allow_slashes=False, to_lower=False) -> str:
    if allow_slashes:
        value = value.replace(" / ", "/")
    value = value.replace(" ", "-")
    if not allow_slashes:
        value = value.replace("/", "-")
    safe_chars: Any = ("-", "_")
    if allow_slashes:
        safe_chars = ("-", "_", "/")
    strip_chars = "".join(c for c in value if c.isalnum() or c in safe_chars).strip()
    if to_lower:
        return strip_chars.lower()
    else:
        return strip_chars


def rm_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def clear_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    mkdir(path)


def move_dir(src_path: str, dst_path: str):
    mkdir(src_path)
    clear_dir(dst_path)
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    clear_dir(src_path)


def dir_size(path: str) -> int:  # pragma: no cover
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size
