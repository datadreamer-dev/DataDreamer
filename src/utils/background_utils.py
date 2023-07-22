from collections import namedtuple
from functools import partial
from multiprocessing import Process, Queue
from typing import Any, Callable, Generator

import dill

EOD = namedtuple("EOD", [])  # End-of-Data indicator


def _func_wrapper(func: Callable, output_queue: Any, kwargs, *args):
    kwargs = dill.loads(kwargs)
    return func(output_queue, *args, **kwargs)


def run_in_background(func: Callable, *args, **kwargs) -> tuple[Process, Any]:
    output_queue: Any = Queue(1)
    p = Process(
        target=partial(_func_wrapper, func),
        args=(output_queue, dill.dumps(kwargs), *args),
    )
    p.daemon = True
    p.start()
    return p, output_queue


def _generator_in_background(generator_output_queue, generator: Callable):
    for _v in dill.loads(generator)():
        generator_output_queue.put(_v)
    generator_output_queue.put(EOD())


def get_generator_in_background(generator: Callable) -> Generator:
    p, generator_output_queue = run_in_background(
        _generator_in_background, generator=dill.dumps(generator)
    )
    while True:
        data = generator_output_queue.get()
        if isinstance(data, EOD):
            break
        yield data
    p.terminate()


__all__ = ["run_in_background", "EOD", "get_generator_in_background"]
