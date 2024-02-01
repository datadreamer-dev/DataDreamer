#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Warn
read -r -p $'Are you sure? You will lose ALL data, logs, experiments, etc. associated with this project.\nType "delete all my data" to confirm: ' response
if [[ "$response" =~ "delete all my data" ]]; then
    :
else
    exit 1
fi

# Confirm
read -r -p "Are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    :
else
    exit 1
fi

# Confirm again
read -r -p "Are you really sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Fully resetting..."
else
    exit 1
fi

# Reset all data
rm -rf ../.cluster/.lock
rm -rf ../.cluster/output/*/* || true
rm -rf ../.cluster/output/ || true
rm ../.cluster/output 2>/dev/null || true
rm -rf ../.cluster/*/.last_job || true
rm -rf ../.venv/*/* || true
rm -rf ../.venv/ || true
rm -rf ../.venv_dev/*/* || true
rm -rf ../.venv_dev/ || true
