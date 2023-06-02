#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

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
    echo "Canceling..."
else
    exit 1
fi

scancel -u "$USER"
