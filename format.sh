#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Create and activate a virtual environment for dev requirements
export PROJECT_VENV=.venv_dev/dev
mkdir -p $PROJECT_VENV
python3 -m venv $PROJECT_VENV
source $PROJECT_VENV/bin/activate
pip3 install -q -U pip
pip3 install -q -r src/requirements-dev.txt

# isort format
if [[ "$(find src -maxdepth 1 -type d | wc -l)" -eq 1 ]]; then
    python3 -m isort src/*.py
else
    python3 -m isort src/*.py src/**/*.py
fi

# Black format
python3 -m black src/

# Lint
python3 -m flake8 src/
