Contributing to DataDreamer
###########################

Contributions are welcome for bug fixes or supporting new models and techniques. This project requires Linux or macOS to run. A general outline of how to get the project running locally for development and contributing is listed below.

Setup for local development
===========================

Fork and clone the project at https://github.com/datadreamer-dev/DataDreamer. After cloning, run:

.. code-block:: bash
    
    cd DataDreamer
    git config --local core.hooksPath ./scripts/.githooks/
    ./scripts/.githooks/post-checkout


The project is now setup and ready for development. DataDreamer also comes bundled with a settings and extension configuration for the `VS Code <https://code.visualstudio.com/>`_ editor which can help setup a development IDE compatible with the project's style and formatting.


Make your changes
===========================

You can add or modify the files you want to change in the ``src`` directory. This documentation website can be helpful at finding the source code you need to modify by clicking the ``[source]`` links in the :doc:`API Reference <../datadreamer>`.

.. tip::
    If you are adding a new model or technique, you can add metadata like license, BibTeX citation, or model card information. End users will be able to access these properties through DataDreamer's automatic model card and data card generation functionality.

Test
===========================

To run the project and test your changes, you can add new test case(s) under the ``src/tests`` folder.

To run all tests you can run:

.. code-block:: bash
    
    ./scripts/run.sh

To run a single file test file you can run with an argument like:

.. code-block:: bash
    
    ./scripts/run.sh src/tests/steps/prompt/test_prompt.py

To filter tests by a string you can use the ``-k`` flag provided by ``pytest``  like:

.. code-block:: bash
    
    ./scripts/run.sh -k "TestFewShotPrompt"

.. tip::

    Each time you run ``run.sh``, it will ensure a virtual environment exists and will install the required dependencies for the project at ``~/.datadreamer_dev/``. After the dependencies are installed on your system, you may want to skip this check to make subsequent runs faster by setting an environment variable:

    .. code-block:: bash

        export PROJECT_SKIP_INSTALL_REQS=1
        ./scripts/run.sh -k "TestFewShotPrompt"
    
    You can also customize the location of the virtual environment directory by editing ``scripts/project.env``.

Type check
===========================

This project uses ``mypy`` for static type checking. To check for any type errors run:

.. code-block:: bash
    
    ./scripts/run.sh -k "mypy"

Format
===========================

Please make sure your changes follow the project style guidelines. You can check for style violations and auto-format by running:

.. code-block:: bash
    
    ./scripts/format.sh

A pre-commit hook is setup on project setup that will auto-format and lint your changes before committing.

Submit your changes
===========================

File a pull request to have your forked changes reviewed. If you are a first-time contributor, you must sign the `CLA <https://gist.github.com/AjayP13/b657de111d8d0907f48ba32eababd911>`_. The CLA can be signed via the CLA Assistant GitHub action that will auto-comment on your pull request.

Your changes will also be tested on our CI system before being approved and merged.