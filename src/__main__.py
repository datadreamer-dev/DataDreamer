import os

from . import project

try:
    from . import __entry__  # type: ignore[attr-defined] # noqa: F401
except ImportError:
    pass
from .__cli__ import _main

if __name__ == "__main__":
    # Set the initial cwd
    project.INITIAL_CWD = os.path.abspath(os.getcwd())

    # Set the working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    _main()
