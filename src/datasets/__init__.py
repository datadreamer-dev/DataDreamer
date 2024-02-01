"""
:py:class:`OutputDataset` and :py:class:`OutputIterableDataset` dataset objects are
returned as outputs from :py:class:`~datadreamer.steps.Step` objects under the
:py:attr:`~datadreamer.steps.Step.output` attribute.

.. tip::

   You never need to construct a dataset object yourself. They are returned as
   :py:attr:`~datadreamer.steps.Step.output` from
   :py:class:`~datadreamer.steps.Step` objects. If you need to convert in-memory Python
   data or data in files to a DataDreamer dataset object, see the
   `DataSource steps <./datadreamer.steps.html#types-of-steps>`_
   available in :py:mod:`datadreamer.steps`.

Accessing Columns
=================

To access a column on the dataset objects you can use the ``__getitem__`` operator like
so: ``step.output['column_name']``. This will return a :py:class:`OutputDatasetColumn`
or :py:class:`OutputIterableDatasetColumn` column object that can be passed as an input
to the ``inputs`` argument of a :py:class:`~datadreamer.steps.Step`.
"""

from .datasets import (
    OutputDataset,
    OutputDatasetColumn,
    OutputIterableDataset,
    OutputIterableDatasetColumn,
)

__all__ = [
    "OutputDataset",
    "OutputDatasetColumn",
    "OutputIterableDataset",
    "OutputIterableDatasetColumn",
]
