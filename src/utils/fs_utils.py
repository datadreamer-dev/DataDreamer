import os
import shutil


def safe_fn(value: str, allow_slashes=False) -> str:
    value = value.replace(" ", "-")
    if not allow_slashes:
        value = value.replace("/", "-")
    safe_chars = ("-", "_")
    if not allow_slashes:
        safe_chars = ("-", "_", "/")
    return "".join(c for c in value if c.isalnum() or c in safe_chars).strip()


def clear_dir(path: str):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def move_dir(src_path: str, dst_path: str):
    os.makedirs(src_path, exist_ok=True)
    clear_dir(dst_path)
    shutil.rmtree(dst_path, ignore_errors=True)
    shutil.copytree(src_path, dst_path)
    clear_dir(src_path)
