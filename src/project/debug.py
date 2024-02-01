import code
import inspect
import os
import pty
import tempfile
from time import sleep

from .report import _deep_defaultdict  # type:ignore[attr-defined]


def _get_callers_locals_and_globals():
    """Gets the local and global variables from the caller's frame.

    Returns:
        tuple[dict, dict]: A tuple of a dictionary of local variables and global
            variables.
    """
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        try:
            return frame.f_back.f_back.f_locals, frame.f_back.f_back.f_globals
        finally:
            del frame


def debugger(rank=None, launch_on_rank=0):
    """Pauses execution and opens an interactive REPL with access to local and global
    variables.

    Args:
        rank (Any, optional): The current rank. Defaults to None (will always
            launch the debugger).
        launch_on_rank (Any, optional): What rank the debugger should be launched on.
            Defaults to 0.
    """
    if "PROJECT_INTERACTIVE" in os.environ:
        from filelock import FileLock

        lock = FileLock(
            os.path.join(
                os.path.dirname(tempfile.mkdtemp()),
                f"{os.environ['PROJECT_NAME']}-debugger.lock",
            )
        )
        if rank is None or rank == launch_on_rank:
            with lock.acquire():
                ls, gs = _get_callers_locals_and_globals()
                all_items = list(ls.items()) + list(gs.items())
                code.interact(
                    banner="Opening Python REPL (press 'Ctrl-D' to exit the shell)...",
                    local=dict(all_items),
                )
        else:
            sleep(5)
            while True:
                with lock.acquire():
                    break


def bash(rank=None, launch_on_rank=0):
    """Pauses execution and opens a bash shell.

    Args:
        rank (Any, optional): The current rank. Defaults to None (will always
            launch bash).
        launch_on_rank (Any, optional): What rank bash should be launched on.
            Defaults to 0.
    """
    if "PROJECT_INTERACTIVE" in os.environ:
        from filelock import FileLock

        lock = FileLock(
            os.path.join(
                os.path.dirname(tempfile.mkdtemp()),
                f"{os.environ['PROJECT_NAME']}-bash.lock",
            )
        )
        if rank is None or rank == launch_on_rank:
            with lock.acquire():
                print("Opening bash shell (type 'exit' to exit the shell)...")
                pty.spawn("/bin/bash")
        else:
            sleep(5)
            while True:
                with lock.acquire():
                    break


# Create a context to help store global context when debugging
context = _deep_defaultdict()
