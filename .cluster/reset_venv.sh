#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Reset the virtual env
rm -rf ../.venv/*/* || true
rm -rf ../.venv/ || true
rm -rf ../.venv_dev/*/* || true
rm -rf ../.venv_dev/ || true
