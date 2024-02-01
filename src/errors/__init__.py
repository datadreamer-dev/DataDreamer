"""
Various exceptions that may be raised when using DataDreamer.

The :py:class:`StepOutputError` exception or any exceptions that derive from it can
be raised when a :py:class:`~datadreamer.steps.Step` is constructing its
:py:attr:`~datadreamer.steps.Step.output`.
"""

from .steps.step import StepOutputError, StepOutputTypeError

__all__ = ["StepOutputError", "StepOutputTypeError"]
