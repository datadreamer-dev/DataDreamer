"""``hello_world`` is an example command-line task."""
import click
from loguru import logger


@click.option(
    "--times",
    "-t",
    type=int,
    required=False,
    default=1,
    help="The number of times to log Hello World!",
)
def hello_world(ctx, times):
    """This command says Hello World!"""
    for _ in range(times):
        logger.success("Hello world!")


__all__ = ["hello_world"]
