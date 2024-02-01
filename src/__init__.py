"""

DataDreamer Sessions
====================

You can run prompting, synthetic data generation, and training workflows within
a DataDreamer session using a context manager like so:

.. code-block:: python

        from datadreamer import DataDreamer

        with DataDreamer('./output/'):
            # ... run steps or trainers here ...

Inside the ``with`` block, you can run any :py:class:`~datadreamer.steps.Step` or
:py:class:`~datadreamer.trainers.Trainer` you want. DataDreamer will automatically
organize, cache, and save the results of each step run within a session to the output
folder.

In-Memory Sessions
------------------------------------------------

Optionally, you can run DataDreamer fully in-memory, without it saving anything to disk,
by passing ``':memory:'`` as the ``output_folder_path`` argument like
``with DataDreamer(':memory:'):``.

Sessions in Interactive Environments
------------------------------------------------

As an alternative to using a Python context manager (``with`` block), you can also
structure your code with :py:meth:`~DataDreamer.start` and :py:meth:`~DataDreamer.stop`
to achieve the same result. Using the context manager, however, is recommended and
preferred. Using :py:meth:`~DataDreamer.start` and :py:meth:`~DataDreamer.stop` may be
useful if you want to run DataDreamer in a Jupyter or Google Colab notebook or
other interactive environments.

.. code-block:: python
    
            from datadreamer import DataDreamer
    
            dd = DataDreamer('./output/')
            dd.start()
            # ... run steps or trainers here ...
            dd.stop()

Caching
=======

DataDreamer caches the results of each step or trainer run within a session to the
output folder. If a session is interrupted and re-run, DataDreamer will automatically
load the results of previously completed steps from disk and resume where it left off.


Attributes:
    __version__ (str): The version of DataDreamer installed.
"""

from .utils import import_utils  # isort: skip # noqa: F401

import importlib.metadata
import os

from .datadreamer import DataDreamer

try:
    project_root_dir = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(project_root_dir, "./pyproject.toml")) as pyproject_fp:
        version_line = [
            line.strip() for line in pyproject_fp if line.startswith("version")
        ][0]
        __version__ = version_line[version_line.find('"') + 1 : version_line.rfind('"')]
except FileNotFoundError:  # pragma: no cover
    __version__ = importlib.metadata.version(
        os.path.basename(os.path.dirname(__file__)) + "-dev"
    )

__all__ = ["__version__", "DataDreamer"]
