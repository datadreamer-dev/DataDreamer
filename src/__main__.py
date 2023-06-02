import os
import sys

import project  # noqa (must be imported in this file for proper environment variable reset)

try:
    import __entry__  # noqa: F401
except ImportError:
    pass
from __cli__ import _main

if __name__ == "__main__":
    # Set module import root
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

    # Set the working directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    _main()
