from collections import namedtuple
from functools import partial
from multiprocessing import Process, Queue
from threading import Thread
from time import sleep
from typing import Any, Callable, Generator

import dill

EOD = namedtuple("EOD", [])  # End-of-Data indicator


def _thread_func_wrapper(func: Callable, kwargs, *args):
    kwargs = kwargs
    return func(*args, **kwargs)


def _process_func_wrapper(func: Callable, pipe: Any, kwargs, *args):  # pragma: no cover
    kwargs = dill.loads(kwargs)
    return func(pipe, *args, **kwargs)


def run_in_background_thread(func: Callable, *args, **kwargs) -> Thread:
    t = Thread(
        target=partial(_thread_func_wrapper, func),
        args=(kwargs, *args),
    )
    t.start()
    return t


def run_in_background_process(func: Callable, *args, **kwargs) -> tuple[Process, Any]:
    pipe: Any = Queue(1)
    p = Process(
        target=partial(_process_func_wrapper, func),
        args=(pipe, dill.dumps(kwargs), *args),
    )
    p.daemon = True
    p.start()
    return p, pipe


def run_in_background_process_no_block(
    func: Callable,
    result_process_func: Callable,
    result_func: Callable,
    *args,
    **kwargs
):
    def _run_in_background_process_no_block(
        func: Callable,
        result_process_func: Callable,
        result_func: Callable,
        *args,
        **kwargs
    ):
        p, pipe = run_in_background_process(func, *args, **kwargs)
        result_process_func(p)
        result_func(pipe.get())
        p.terminate()

    run_in_background_thread(
        partial(
            _run_in_background_process_no_block,
            func,
            result_process_func,
            result_func,
            *args,
            **kwargs
        )
    )


def _generator_in_background(generator_pipe, generator: Callable):  # pragma: no cover
    for _v in dill.loads(generator)():
        while generator_pipe.qsize() > 10:
            sleep(0.3)
        generator_pipe.put(_v)
    generator_pipe.put(EOD())


def get_generator_in_background(generator: Callable) -> Generator:
    p, generator_pipe = run_in_background_process(
        _generator_in_background, generator=dill.dumps(generator)
    )
    while True:
        data = generator_pipe.get()
        if isinstance(data, EOD):
            break
        yield data
    p.terminate()


__all__ = ["run_in_background_process", "EOD", "get_generator_in_background"]
