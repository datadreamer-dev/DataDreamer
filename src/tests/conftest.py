import os
from glob import glob

from .. import project


# Register pytest fixtures
def refactor(string: str) -> str:
    return string.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [
    refactor(fixture)
    for fixture in glob("src/tests/test_utils/fixtures/*.py")
    if "__" not in fixture
]

# Set the initial cwd
project.INITIAL_CWD = os.path.abspath(os.getcwd())

# Set the working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))
