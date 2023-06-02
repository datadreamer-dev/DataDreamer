#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Create and activate a virtual environment for dev requirements
export PROJECT_VENV=.venv_poetry
mkdir -p $PROJECT_VENV
python3 -m venv $PROJECT_VENV
source $PROJECT_VENV/bin/activate
pip3 install -q -U pip
pip3 install -q -r src/requirements-dev.txt

export PACKAGE_NAME=$(grep "name = " pyproject.toml | head -n1 | cut -d'"' -f 2)
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
export POETRY_HOME=/nlp/data/$USER/.cache/pypoetry
mkdir -p $POETRY_HOME
export POETRY_CACHE_DIR=/nlp/data/$USER/.cache/pypoetry/cache
mkdir -p $POETRY_CACHE_DIR

python -m poetry publish "$@"
