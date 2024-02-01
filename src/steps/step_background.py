from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from time import sleep
from typing import TYPE_CHECKING, Callable

from .. import DataDreamer
from ..errors import StepOutputError
from ..utils.background_utils import get_thread_id, run_in_background_thread

if TYPE_CHECKING:  # pragma: no cover
    from .step import Step


def _check_step_output(step: "Step") -> bool:
    try:
        step.output  # noqa: B018
        return True
    except StepOutputError:
        return False


def _waiter(steps, poll_interval=1.0):
    while len(steps) > 0:
        step = steps[-1]
        if _check_step_output(step):
            steps.pop()
        else:
            sleep(poll_interval)


def wait(*steps: "Step", poll_interval=1.0):
    """Wait for all steps to complete if they are running in the background.

    Args:
        poll_interval: How often to poll in seconds.
    """
    from ..steps import Step

    if not all([isinstance(s, Step) for s in steps]):
        raise TypeError("All arguments to wait() must be of type Step.")
    if all([_check_step_output(step) for step in steps]):
        return
    steps_list = list(steps)
    wait_thread = run_in_background_thread(
        _waiter, steps_list, poll_interval=poll_interval
    )
    wait_thread.join()


def concurrent(*funcs: Callable):
    """Run a set of functions (which run steps) concurrently.

    Args:
        *funcs: The functions to run concurrently.
    """
    parent_thread_id = get_thread_id()

    def wrapper_func(parent_thread_id, func):
        if DataDreamer.initialized():
            DataDreamer._register_child_thread(parent_thread_id)
        return func()

    if not all([callable(f) for f in funcs]):
        raise TypeError("All arguments to concurrent() must be functions.")
    thread_pool = ThreadPool(len(funcs))
    results = thread_pool.map(partial(wrapper_func, parent_thread_id), funcs)
    thread_pool.close()
    thread_pool.join()
    return results
