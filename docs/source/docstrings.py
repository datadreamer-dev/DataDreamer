def append_new_lines(lines, n=1):
    for _ in range(n):
        lines.append("")


def clear(lines):
    lines.clear()
    append_new_lines(lines, 1)


def split_lines(v):
    return v.strip().split("\n")


def indent_lines(lines):
    lines.copy()
    for line_idx in range(len(lines)):
        lines[line_idx] = f"    {lines[line_idx]}"
    return lines


def append(lines, v, indent_level=0):
    v = split_lines(v)
    for _ in range(indent_level):
        v = indent_lines(v)
    lines.extend(v)


STEP_HELP = """.. code-block:: jsonnet
    :caption: CLS_NAME.help
"""

RETRIEVER_RUN = [
    "Retrieves the closest texts to the input queries.",
    "",
    ":type queries: :py:class:`~typing.Iterable`\\[:py:data:`~typing.Any`]",
    ":param queries: The queries to retrieve the closest texts to.",
    ":type k: :py:class:`int`, default: ``5``",
    ":param k: The number of closest texts to retrieve.",
    ":type batch_size: :py:class:`int`, default: ``10``",
    ":param batch_size: The batch size to use for retrieval.",
    ":type batch_scheduler_buffer_size: :py:data:`~typing.Optional`\\[:py:class:`int`], default: ``None``",
    ":param batch_scheduler_buffer_size: The buffer size to use for the batch scheduler.",
    ":type adaptive_batch_size: :py:class:`bool`, default: ``False``",
    ":param adaptive_batch_size: Whether to use adaptive batch sizing.",
    ":type progress_interval: :py:data:`~typing.Optional`\\[:py:class:`int`], default: ``60``",
    ":param progress_interval: How often to log progress in seconds.",
    ":type force: :py:class:`bool`, default: ``False``",
    ":param force: Whether to force run the step (ignore saved results).",
    ":type cache_only: :py:class:`bool`, default: ``False``",
    ":param cache_only: Whether to only use the cache.",
    ":type verbose: :py:data:`~typing.Optional`\\[:py:class:`bool`], default: ``None``",
    ":param verbose: Whether or not to print verbose logs.",
    ":type log_level: :py:data:`~typing.Optional`\\[:py:class:`int`], default: ``None``",
    ":param log_level: The logging level to use (:py:data:`~logging.DEBUG`, :py:data:`~logging.INFO`, etc.).",
    ":type total_num_queries: :py:data:`~typing.Optional`\\[:py:class:`int`], default: ``None``",
    ":param total_num_queries: The total number of queries being processed (helps with",
    "                          displaying progress).",
    ":type return_generator: :py:class:`bool`, default: ``False``",
    ":param return_generator: Whether to return a generator instead of a list.",
    ":type \\*\\*kwargs: ",
    ":param \\*\\*kwargs: Additional keyword arguments to pass to the embedder.",
    "",
    ":rtype: :py:data:`~typing.Union`\\[:py:class:`~typing.Generator`\\[:py:class:`dict`\\[:py:class:`str`, :py:data:`~typing.Any`], :py:obj:`None`, :py:obj:`None`], :py:class:`list`\\[:py:class:`dict`\\[:py:class:`str`, :py:data:`~typing.Any`]]]",
    ":returns: A set of results.",
    "",
]

TASK_MODEL_RUN = [
    "Runs the model on the texts.",
    "",
    ":type texts: :py:class:`~typing.Iterable`\\[:py:data:`~typing.Any`]",
    ":param texts: The texts to run against the model.",
    ":type truncate: :py:class:`bool`, default: ``False``",
    ":param truncate: Whether to truncate the texts.",
    ":type batch_size: :py:class:`int`, default: ``10``",
    ":param batch_size: The batch size to use.",
    ":type batch_scheduler_buffer_size: :py:data:`~typing.Optional`\\[:py:class:`int`], default: ``None``",
    ":param batch_scheduler_buffer_size: The buffer size to use for the batch scheduler.",
    ":type adaptive_batch_size: :py:class:`bool`, default: ``False``",
    ":param adaptive_batch_size: Whether to use adaptive batch sizing.",
    ":type progress_interval: :py:data:`~typing.Optional`\\[:py:class:`int`], default: ``60``",
    ":param progress_interval: How often to log progress in seconds.",
    ":type force: :py:class:`bool`, default: ``False``",
    ":param force: Whether to force run the step (ignore saved results).",
    ":type cache_only: :py:class:`bool`, default: ``False``",
    ":param cache_only: Whether to only use the cache.",
    ":type verbose: :py:data:`~typing.Optional`\\[:py:class:`bool`], default: ``None``",
    ":param verbose: Whether or not to print verbose logs.",
    ":type log_level: :py:data:`~typing.Optional`\\[:py:class:`int`], default: ``None``",
    ":param log_level: The logging level to use (:py:data:`~logging.DEBUG`, :py:data:`~logging.INFO`, etc.).",
    ":type total_num_texts: :py:data:`~typing.Optional`\\[:py:class:`int`], default: ``None``",
    ":param total_num_texts: The total number of texts being processed (helps with",
    "                          displaying progress).",
    ":type return_generator: :py:class:`bool`, default: ``False``",
    ":param return_generator: Whether to return a generator instead of a list.",
    ":type \\*\\*kwargs: ",
    ":param \\*\\*kwargs: Additional keyword arguments to pass when running the model.",
    "",
    ":rtype: :py:data:`~typing.Union`\\[:py:class:`~typing.Generator`\\[:py:class:`dict`\\[:py:class:`str`, :py:data:`~typing.Any`], :py:obj:`None`, :py:obj:`None`], :py:class:`list`\\[:py:class:`dict`\\[:py:class:`str`, :py:data:`~typing.Any`]]]",
    ":returns: The result of running the model on the texts.",
    "",
]

TASK_MODEL_RUN_WITH_INSTRUCTION = TASK_MODEL_RUN[0:4] + ([
    ":type instruction: :py:class:`str`",
    ":param instruction: An instruction to prepend to the texts before running.",
]) + TASK_MODEL_RUN[4:]