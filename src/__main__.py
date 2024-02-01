import os

from . import project

try:
    from . import __entry__  # type: ignore[attr-defined] # noqa: F401
except ImportError:
    pass
from .__cli__ import _main

if __name__ == "__main__":  # pragma: no cover
    # Set the initial cwd
    project.INITIAL_CWD = os.path.abspath(os.getcwd())

    _main()
