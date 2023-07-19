import logging

from ..project.environment import RUNNING_IN_PYTEST

# stderr Handler
stderr_handler = logging.StreamHandler()
stderr_handler.setLevel(logging.DEBUG)

# Logger
logger = logging.getLogger("datadreamer")
if RUNNING_IN_PYTEST:
    logger.propagate = True
else:
    logger.propagate = False  # pragma: no cover
formatter = logging.Formatter(
    "[ \N{ESC}[35m🤖 Data\N{ESC}[33mDr\N{ESC}[31mea\N{ESC}[35mmer\u001b[0m 💤 ] %(message)s"
)
stderr_handler.setFormatter(formatter)
logger.addHandler(stderr_handler)
logger.setLevel(logging.CRITICAL + 1)
