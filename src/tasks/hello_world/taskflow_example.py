"""``hello_world`` is an example command-line task."""
from tkinter import E
import click
import contextlib
import functools
from loguru import logger

from time import sleep
from taskflow.exceptions import WrappedFailure, NotFound
from taskflow import engines
from taskflow.patterns import graph_flow
from taskflow import task
from taskflow.types import notifier
from taskflow.persistence import backends
from taskflow.persistence import models
from uuid import uuid4

engine = None

EVENT_UPDATE_METADATA = "update_metadata"


class TaskA(task.Task):
    TASK_EVENTS = set(list(task.Task.TASK_EVENTS) + [EVENT_UPDATE_METADATA])

    def execute(self):
        logger.info(f"Executing '{self.name}'")
        start_i = engine.storage._atomdetail_by_name(self.name)[0].meta.get("i", -1)
        for i in range(start_i + 1, 30):
            sleep(1)
            self._notifier.notify(EVENT_UPDATE_METADATA, {"i": i})
            self.update_progress(i / 29)
        return "a"


class TaskB(task.Task):
    def execute(self, taska_output):
        logger.info(f"Executing '{self.name}'")
        sleep(30)
        logger.info(f"Got input '{taska_output}'")


class TaskC(task.Task):
    def execute(self):
        logger.info(f"Executing '{self.name}'")


def flow_watch(state, details):
    logger.debug("Flow => %s (%s)" % (state, details))


def task_watch(state, details):
    logger.debug("Task %s => %s (%s)" % (details.get("task_name"), state, details))


def progress_printer(task, event_type, details):
    # This callback, attached to each task will be called in the local
    # process (not the child processes)...
    progress = details.pop("progress")
    progress = int(progress * 100.0)
    print("Task '%s' reached %d%% completion" % (task.name, progress))


def update_metadata(task, event_type, details):
    update_with = details
    engine.storage.update_atom_metadata(task.name, update_with)


def hello_world(ctx):
    """This command says Hello World!"""
    global engine

    logger.info("Constructing...")
    wf = graph_flow.Flow("my-flow")
    a_task = TaskA("A Task", provides=["taska_output"])
    wf.add(a_task)
    wf.add(TaskB("B Task", requires=["taska_output"]))
    wf.add(TaskC("C1 Task"))
    wf.add(TaskC("C2 Task"))

    logger.info("Graph Representation...")
    logger.info(wf._graph.pformat())

    logger.info("Loading...")
    backend = backends.fetch({"connection": "sqlite:///test.sqlite"})
    with contextlib.closing(backend.get_connection()) as conn:
        conn.upgrade()
        book = models.LogBook("My Log Book", uuid="dummy")
        flow_detail = models.FlowDetail("My Flow Detail", uuid="dummy")
        book.add(flow_detail)
        try:
            resumed_flow_detail = conn.get_logbook("dummy").find("dummy")
        except NotFound:
            resumed_flow_detail = None
        if resumed_flow_detail is None:
            logger.info("Initial run.")
            conn.save_logbook(book)
        else:
            logger.info("Resuming run!")
            flow_detail = resumed_flow_detail
    engine = engines.load(
        wf,
        executor="processes",
        engine="parallel",
        flow_detail=flow_detail,
        book=book,
        backend=backend,
        max_workers=4,
    )

    # Register notification hooks
    engine.notifier.register(notifier.Notifier.ANY, flow_watch)
    engine.atom_notifier.register(notifier.Notifier.ANY, task_watch)
    a_task.notifier.register(
        task.EVENT_UPDATE_PROGRESS, functools.partial(progress_printer, a_task)
    )
    a_task.notifier.register(
        EVENT_UPDATE_METADATA, functools.partial(update_metadata, a_task)
    )

    logger.info("Compiling...")
    engine.compile()

    logger.info("Preparing...")
    engine.prepare()

    logger.info("Running...")
    try:
        engine.run()
    except WrappedFailure as wrapped_failure:
        for e in wrapped_failure:
            print(e)

    logger.info("Done...")


__all__ = ["hello_world"]
