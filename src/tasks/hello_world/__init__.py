"""``hello_world`` is an example command-line task."""
from tkinter import E
import click
from loguru import logger
import guidance
from ...project import debugger


def hello_world(ctx):
    """This command says Hello World!"""
    gpt2 = guidance.llms.Transformers("gpt2")
    guidance.llm = gpt2

    program = guidance(
        """My cell phone number is {{gen "completion" pattern="[0-9\\-\\(\\)\\. ]+" stop=" " save_stop_text=True temperature=0.7}}"""
    )
    executed_program = program()
    logger.info(executed_program)
    logger.info(executed_program.variables())

    program = guidance(
        """My cell phone number is {{gen "completion" pattern="[0-9\\-\\(\\)\\. ]+" stop=" " save_stop_text=True temperature=0.65}}"""
    )
    executed_program = program()
    logger.info(executed_program)
    logger.info(executed_program.variables())

    program = guidance(
        """My cell phone number is {{gen "completion" pattern="[0-9\\-\\(\\)\\. ]+" stop=" " save_stop_text=True temperature=0.4}}"""
    )
    executed_program = program()
    logger.info(executed_program)
    logger.info(executed_program.variables())

    program = guidance(
        """My cell phone number is {{gen "completion" pattern="[0-9\\-\\(\\)\\. ]+" stop=" " save_stop_text=True temperature=0.8}}"""
    )
    executed_program = program()
    logger.info(executed_program)
    logger.info(executed_program.variables())


__all__ = ["hello_world"]
