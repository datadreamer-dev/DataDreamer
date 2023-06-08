"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    __version__ (str): The version of the currently installed package.

Todo:
    * Update this docstring

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""
import importlib.metadata
import os

try:
    project_root_dir = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(project_root_dir, "./pyproject.toml")) as pyproject_fp:
        version_line = [
            line.strip() for line in pyproject_fp if line.startswith("version")
        ][0]
        __version__ = version_line[version_line.find('"') + 1 : version_line.rfind('"')]
except FileNotFoundError:
    __version__ = importlib.metadata.version(
        os.path.basename(os.path.dirname(__file__))
    )

__all__ = ["__version__"]
