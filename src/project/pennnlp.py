import os
import shutil
import sys

from loguru import logger

"""This file contains PennNLP cluster specific utilities."""

NLPDATA_PATH = os.path.join("/nlp/data/", os.environ["USER"])

SCRATCH_PATH = os.path.join("/scratch/", os.environ["USER"], os.environ["PROJECT_NAME"])


def detect_pennnlp():
    """Detect if running on PennNLP's cluster.

    Returns:
        bool: Whether or not we are running on PennNLP's cluster.
    """
    return os.path.exists(NLPDATA_PATH)


def copy_file(src, dest):
    """Copies a file from the source path to the destination path, but skips the copying
    if the file already exists (determined by last modified time or file size).

    Args:
        src (str): The source path.
        dest (str): The destination path.
    """
    if (
        (not os.path.exists(dest))
        or (os.stat(src).st_mtime - os.stat(dest).st_mtime > 1)
        or (os.stat(src).st_size != os.stat(dest).st_size)
    ):
        shutil.copy2(src, dest)


def copy_files_to_ssd(*paths, subfolder=None):
    if detect_pennnlp():
        # Create scratch dir for SSD disk speed on PennNLP cluster
        scratch_path = SCRATCH_PATH
        if subfolder is True:
            scratch_path = os.path.join(SCRATCH_PATH, sys.argv[1])
        elif subfolder:
            scratch_path = os.path.join(SCRATCH_PATH, subfolder)
        os.makedirs(scratch_path, exist_ok=True)

        # Copy files to scratch dir for SSD disk speed on PennNLP cluster
        new_paths = []
        for path in paths:
            path = os.path.normpath(os.path.abspath(path))
            basename = os.path.basename(path)
            new_path = os.path.join(scratch_path, basename)
            new_paths.append(new_path)
            logger.debug(f"Copying file {path} to PennNLP scratch path: {new_path}...")
            copy_file(path, new_path)
            logger.debug("Done copying file.")
        return new_paths
    else:
        return paths
