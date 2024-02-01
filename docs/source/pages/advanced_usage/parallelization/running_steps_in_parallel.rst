Running Steps in Parallel
#######################################################

There are two ways to run steps in parallel:

1. **Running steps in different processes:** Steps can be run asynchronously in a background process.
To run multiple steps in parallel, you can run them all in the background and then wait for them to completed.

2. **Running steps in different threads:** You can group steps into Python functions. You can then run these functions
in parallel using :py:func:`~datadreamer.steps.concurrent`.

Running steps in different processes
====================================
You can run steps in the background by passing the ``background=True`` keyword argument to :py:class:`~datadreamer.steps.Step` construction.
This will run the step in its own background process asynchronously.

Waiting for :py:attr:`~datadreamer.steps.Step.output`
-----------------------------------------------------
When you run a step in the background, its output may not be immediately ready, and trying to access
:py:attr:`~datadreamer.steps.Step.output` may raise an exception until the step has completed running in
the background. To wait for a step's output to be ready, you can call :py:func:`~datadreamer.steps.wait`
on the step. This will block until the step's output is ready.

Running steps in different threads
==================================

To run multiple steps in parallel, you can group them into Python functions and run these functions in parallel using threads. You can pass
the functions to :py:func:`~datadreamer.steps.concurrent` to run them in parallel using threading.
