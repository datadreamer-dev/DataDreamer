"""
:py:class:`TaskModel` objects help perform some sort of arbitrary NLP task
(classification, etc.).
All task models derive from the :py:class:`TaskModel` base class.

.. tip::

   Instead of using :py:meth:`~TaskModel.run` directly, use a
   :py:class:`step <datadreamer.steps>` that takes a :py:class:`TaskModel` as an
   ``args`` argument such as :py:class:`~datadreamer.steps.RunTaskModel`.

Caching
=======
Task models internally perform caching to disk, so if you run the same text multiple
times, the task model will only run once and then cache the results for future runs.
"""

from .hf_classification_task_model import HFClassificationTaskModel
from .parallel_task_model import ParallelTaskModel
from .task_model import TaskModel

__all__ = ["TaskModel", "HFClassificationTaskModel", "ParallelTaskModel"]
