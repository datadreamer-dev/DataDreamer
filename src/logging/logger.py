import logging
from logging import Logger

from ..project.environment import RUNNING_IN_PYTEST

DATEFMT: str = "[%Y-%m-%d %H:%M:%S %z]"
STANDARD_FORMAT: str = "[ \N{ESC}[35m🤖 Data\N{ESC}[33mDr\N{ESC}[31mea\N{ESC}[35mmer\u001b[0m 💤 ] %(message)s"  # noqa: B950
DATETIME_FORMAT: str = "%(asctime)s [ \N{ESC}[35m🤖 Data\N{ESC}[33mDr\N{ESC}[31mea\N{ESC}[35mmer\u001b[0m 💤 ] %(message)s"  # noqa: B950

# stderr Handler
STDERR_HANDLER = logging.StreamHandler()
STDERR_HANDLER.setLevel(logging.DEBUG)

# Logger
logger: Logger = logging.getLogger("datadreamer")
if RUNNING_IN_PYTEST:
    logger.propagate = True
else:
    logger.propagate = False  # pragma: no cover
formatter = logging.Formatter(
    STANDARD_FORMAT, datefmt="[%Y-%m-%d %H:%M:%S %z]", validate=False
)
STDERR_HANDLER.setFormatter(formatter)
logger.addHandler(STDERR_HANDLER)
logger.setLevel(logging.CRITICAL + 1)
