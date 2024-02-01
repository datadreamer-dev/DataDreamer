import click

from . import project


# Register main
@click.group()
@click.pass_context
def _main(*args, **kwargs):  # pragma: no cover
    # Run init
    project.init()
