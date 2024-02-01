#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")/../" || exit

poetry publish "$@"
