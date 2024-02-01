import importlib
import os
import socket
import sys
from collections import UserDict, namedtuple
from contextlib import closing
from functools import cache, partial
from logging import Logger, StreamHandler
from multiprocessing import get_context, spawn
from multiprocessing.context import SpawnProcess
from threading import Lock, Thread, get_ident
from time import sleep, time
from typing import Any, Callable, Generator, Type

import dill
import Pyro5
import Pyro5.api
import Pyro5.server
import torch

from .. import logging as datadreamer_logging

_PROCESS_SPAWN_LOCK = Lock()
EOD = namedtuple("EOD", [])  # End-of-Data indicator


def get_thread_id() -> tuple[int, int]:
    return (os.getpid(), get_ident())


def _thread_func_wrapper(func: Callable, kwargs, *args):
    kwargs = kwargs
    return func(*args, **kwargs)


_old_spawn_get_preparation_data = spawn.get_preparation_data


def _get_preparation_data__dont_run_main_again(name):
    prep_data = _old_spawn_get_preparation_data(name)
    del prep_data["init_main_from_path"]
    try:  # pragma: no cover
        import datadreamer  # noqa: F401

        prep_data["init_main_from_name"] = "datadreamer"
    except (ModuleNotFoundError, ImportError):  # pragma: no cover
        # In the test environment we can't import datadreamer, so we use os instead
        prep_data["init_main_from_name"] = "os"
    return prep_data


@cache
def _monkey_patch_get_preparation_data():
    # IMPORTANT: This prevents multiprocessing spawn from running
    # everything the main.py again if the user did not protect
    # the entrypoint with if __name__ == 'main':
    # See: https://superfastpython.com/multiprocessing-spawn-runtimeerror/
    # See CPython for why this works:
    #   https://github.com/python/cpython/blob/f7c05d7ad3075a1dbeed86b6b12903032e4afba6/
    #   Lib/multiprocessing/spawn.py#L160
    spawn.get_preparation_data = _get_preparation_data__dont_run_main_again


def get_parent_process_context() -> dict[str, Any]:
    from .. import DataDreamer

    # Monkey patch get_preparation_data for spawn
    _monkey_patch_get_preparation_data()

    # Remove non-picklable (and un-needed attrs from ctx)
    ctx: Any = UserDict()
    ctx.__dict__.update(DataDreamer.ctx.__dict__.copy())
    if hasattr(ctx, "background_processes"):
        del ctx.background_processes
    if hasattr(ctx, "caches"):
        ctx.caches = {}

    return {
        "manager": dill.dumps(Logger.manager),
        "log_dicts": dill.dumps(
            [
                {
                    k: v
                    for k, v in lgr.__dict__.items()
                    if k in ["level", "propagate", "disabled"]
                }
                for lgr in Logger.manager.loggerDict.values()
            ]
        ),
        "log_handlers": dill.dumps(
            [
                [
                    (h.level, h.formatter)
                    for h in lgr.handlers
                    if isinstance(h, StreamHandler)
                ]
                if isinstance(lgr, Logger)
                else []
                for lgr in Logger.manager.loggerDict.values()
            ]
        ),
        "ctx": dill.dumps(ctx),
    }


def restore_parent_process_context(
    parent_context: dict[str, Any], env: dict[str, Any]
):  # pragma: no cover
    # Setup environment variables
    os.environ["DATADREAMER_BACKGROUND_PROCESS"] = "1"
    os.environ.update(env)
    importlib.reload(torch.cuda)
    importlib.reload(torch.cuda.random)

    # Helper to replace the logger format with a worker-specific
    # logger format if running in distributed mode
    from .distributed_utils import get_global_rank

    def format_for_worker_logging(_fmt: str) -> str:
        return (
            _fmt.replace(" 💤 ]", f" 💤 (Worker #{get_global_rank()}) ]")
            if "DATADREAMER_DISTRIBUTED" in env
            else _fmt
        )

    # Setup loggers to be exactly like the parent process
    datadreamer_logging.logger.handlers.clear()
    datadreamer_logging.STANDARD_FORMAT = format_for_worker_logging(
        datadreamer_logging.STANDARD_FORMAT
    )
    datadreamer_logging.DATETIME_FORMAT = format_for_worker_logging(
        datadreamer_logging.DATETIME_FORMAT
    )
    Logger.manager = dill.loads(parent_context["manager"])
    for lgr, log_dict, log_handler in zip(
        Logger.manager.loggerDict.values(),
        dill.loads(parent_context["log_dicts"]),
        dill.loads(parent_context["log_handlers"]),
    ):
        lgr.__dict__.update(log_dict)
        for level, formatter in log_handler:
            if "DATADREAMER_DISTRIBUTED" in env:
                formatter._style._fmt = format_for_worker_logging(formatter._style._fmt)
                formatter._fmt = format_for_worker_logging(formatter._fmt)
            stderr_handler = StreamHandler()
            stderr_handler.setLevel(level)
            stderr_handler.setFormatter(formatter)
            if isinstance(lgr, Logger):
                lgr.addHandler(stderr_handler)

    # Setup DataDreamer context from parent process
    from .. import DataDreamer

    ctx = dill.loads(parent_context["ctx"])
    ctx.instance.__enter__()
    DataDreamer.ctx = ctx

    # Run any monkey patches that were applied in the parent process
    from ..llms.petals import _monkey_patch_ServerInferenceSession_step
    from ..steps.step_output import _monkey_patch_iterable_dataset_apply_feature_types

    if getattr(ctx, "_monkey_patched_iterable_dataset_apply_feature_types", False):
        _monkey_patch_iterable_dataset_apply_feature_types()
    if getattr(ctx, "_monkey_patched_ServerInferenceSession_step", False):
        _monkey_patch_ServerInferenceSession_step()


def _process_func_wrapper(
    parent_context: dict[str, Any],
    env: dict[str, Any],
    func: Callable,
    pipe: Any,
    args,
    kwargs,
):  # pragma: no cover
    # Restore parent process context
    restore_parent_process_context(parent_context=parent_context, env=env)

    # Unpickle and run the function
    args = dill.loads(args)
    kwargs = dill.loads(kwargs)
    return dill.loads(func)(pipe, *args, **kwargs)


def run_in_background_thread(func: Callable, *args, **kwargs) -> Thread:
    t = Thread(target=partial(_thread_func_wrapper, func), args=(kwargs, *args))
    t.start()
    return t


def run_in_background_process(
    func: Callable, *args, env=None, **kwargs
) -> tuple[SpawnProcess, Any]:
    spawn_context = get_context(method="spawn")
    pipe: Any = spawn_context.Queue(1)
    env = env if env is not None else os.environ.copy()
    assert env is not None
    p = spawn_context.Process(
        target=partial(
            _process_func_wrapper, get_parent_process_context(), env, dill.dumps(func)
        ),
        args=(pipe, dill.dumps(args), dill.dumps(kwargs)),
    )
    p.daemon = False
    with _PROCESS_SPAWN_LOCK:
        p.start()
    return p, pipe


def run_in_background_process_no_block(
    func: Callable,
    result_process_func: Callable,
    result_func: Callable,
    *args,
    **kwargs,
):
    def _run_in_background_process_no_block(
        func: Callable,
        result_process_func: Callable,
        result_func: Callable,
        *args,
        **kwargs,
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
            **kwargs,
        )
    )


def _generator_in_background(generator_pipe, generator: Callable):  # pragma: no cover
    for _v in dill.loads(generator)():
        if sys.platform != "darwin":
            # https://stackoverflow.com/questions/65609529/
            # python-multiprocessing-queue-notimplementederror-macos
            while generator_pipe.qsize() > 10:
                sleep(0.3)
        generator_pipe.put(_v)
    generator_pipe.put(EOD())


def get_generator_in_background(generator: Callable) -> Generator:
    from .. import DataDreamer

    p, generator_pipe = run_in_background_process(
        _generator_in_background, generator=dill.dumps(generator)
    )
    if DataDreamer.initialized() and not DataDreamer.is_background_process():
        DataDreamer._add_process(p)
    while True:
        data = generator_pipe.get()
        if isinstance(data, EOD):
            break
        yield data
    p.terminate()


class RunIfTimeout:
    def __init__(
        self, func: Callable[[], None], timeout: float, poll_interval: float = 1.0
    ):
        self.func = func
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.start_time = time()
        self.disabled = False
        self.timed_out = False
        self.thread = run_in_background_thread(self._worker)

    def _worker(self):
        while True:
            if self.disabled:
                break
            if (time() - self.start_time) >= self.timeout:  # pragma: no cover
                self.timed_out = True
                self.func()
                break
            sleep(self.poll_interval)

    def stop(self, func: Callable[[], None]):
        self.disabled = True
        if self.timed_out:  # pragma: no cover
            func()
        self.thread.join()


def find_free_port():  # pragma: no cover
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def wait_for_port(port: int):  # pragma: no cover
    while True:
        try:
            with socket.create_connection(("localhost", port), timeout=5):
                break
        except OSError:
            pass


def proxy_resource_in_background(resource: Type, env=None):  # pragma: no cover
    from .. import DataDreamer

    # Configuration options for the connection between a client/server
    Pyro5.config.SERIALIZER = "marshal"
    Pyro5.config.DETAILED_TRACEBACK = True
    free_port = find_free_port()
    uri = f"PYRO:resource@localhost:{free_port}"
    resource = Pyro5.server.expose(resource)

    # Define a server
    def _server_function(pipe, resource, free_port):  # pragma: no cover
        Pyro5.server.serve(
            {resource: "resource"}, port=free_port, use_ns=False, verbose=False
        )

    # Run server in a background process
    process, _ = run_in_background_process(
        _server_function, resource, free_port, env=env
    )
    if DataDreamer.initialized() and not DataDreamer.is_background_process():
        DataDreamer._add_process(process)

    # Create client object that proxys to the server
    class BackgroundResource(object):
        def __init__(self):
            self.proxy = Pyro5.api.Proxy(uri)
            self.process = process
            wait_for_port(free_port)

        def __del__(self):
            self.proxy._pyroRelease()
            if process.is_alive():
                process.terminate()

    return BackgroundResource()


__all__ = [
    "get_thread_id",
    "run_in_background_thread",
    "run_in_background_process",
    "EOD",
    "get_generator_in_background",
    "RunIfTimeout",
    "proxy_resource_in_background",
]
