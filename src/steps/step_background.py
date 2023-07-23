from time import sleep

from ..errors import StepOutputError
from ..utils.background_utils import run_in_background_thread
from .step import Step


def _waiter(steps, poll_interval=1.0):
    while len(steps) > 0:
        try:
            step = steps[-1]
            step.output
            steps.pop()
        except StepOutputError:
            sleep(poll_interval)


def wait(*steps: Step, poll_interval=1.0):
    steps_list = list(steps)
    wait_thread = run_in_background_thread(
        _waiter, steps_list, poll_interval=poll_interval
    )
    wait_thread.join()
