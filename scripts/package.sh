#!/bin/bash
set -e

# Change directory to script location
cd "$(dirname "$0")/../" || exit

# Create and activate a virtual environment for the build
export PROJECT_VENV=.venv_poetry
mkdir -p $PROJECT_VENV
python3 -m venv $PROJECT_VENV
source $PROJECT_VENV/bin/activate
pip3 install -q -U pip

export PACKAGE_NAME=$(grep "include = " pyproject.toml | head -n1 | cut -d'"' -f 2 | awk '{print tolower($0)}')
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
export POETRY_HOME=/nlp/data/ajayp/.cache/pypoetry
mkdir -p $POETRY_HOME
export POETRY_CACHE_DIR=/nlp/data/ajayp/.cache/pypoetry/cache
mkdir -p $POETRY_CACHE_DIR

echo "Setting up..."
cp pyproject.toml pyproject.toml.bak
poetry add $(cat src/requirements.txt | grep -v "pandas-stubs>=")
poetry add $(cat src/requirements-cpu.txt | sed 's/+cpu//' | grep -v "find-links" | grep -v "torchvision=" | grep -v "torchaudio=" | grep -v "tensorflow=")
poetry add --group dev $(cat src/requirements-dev.txt)
mv src/ $PACKAGE_NAME/

echo "Building 'dist' folder..."
rm -rf dist || true
cp .gitignore .gitignore.bak
cat .gitignore.bak | grep -v '^\s*datadreamer\s*$' >.gitignore
poetry build
mv .gitignore.bak .gitignore

echo "Cleaning up..."
mv pyproject.toml.bak pyproject.toml
mv $PACKAGE_NAME/ src/
if [[ $* != *--keep-venv* ]]; then
    rm -rf ./.venv_poetry
fi
rm -rf poetry.lock
