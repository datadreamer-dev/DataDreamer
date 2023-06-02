#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Set environment variables
export PROJECT_VENV=.venv/prod

# Ensure direct environment variables are provided
source .cluster/direct/_direct_env.sh

# Load user-specified project environment variables
source project.env

# Set environment variables
export PROJECT_CACHE_DIR=$PROJECT_DATA/.cache

# Run the boot script
.cluster/_boot.sh "$@"
