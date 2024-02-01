#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Fetch the status of all jobs
echo "Fetching the status of all jobs..." 1>&2
qstat -u "$USER"
