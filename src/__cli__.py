import click

import project
from project.builtin_tasks import register_builtin_tasks
from tasks.hello_world import hello_world


# Register main
@click.group()
@click.pass_context
@click.option(
    "--local_rank", type=int, required=False, hidden=True
)  # DeepSpeed passes this when used
def _main(*args, **kwargs):
    # Run init
    project.init()


# Register built-in tasks
register_builtin_tasks(_main)

# Register tasks
_main.command()(click.pass_context(hello_world))
