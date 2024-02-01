#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Warn
read -r -p $'Are you sure? You will lose all data, logs, etc. associated with this project that are not tagged as an experiment.\nType "i only want experiments" to confirm: ' response
if [[ "$response" =~ "i only want experiments" ]]; then
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
    echo "Resetting..."
else
    exit 1
fi

# Reset all data
rm -rf ../.cluster/output/committed || true
rm -rf ../.cluster/output/named || true
(
    cd ../.cluster/output/dated || exit
    for x in *; do readlink -f ../experiments/* | grep -q "$x" || rm -rf "$x"; done
)
(
    cd ../.cluster/output/persistent_data || exit
    for x in *; do readlink -f ../dated/*/data/* | grep -q "$x" || rm -rf "$x"; done
)
rm -rf ../.cluster/*/.last_job || true
