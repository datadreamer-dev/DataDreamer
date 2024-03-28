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

# Cancel all jobs
for JOB_FOLDER in ../output/dated/*/; do
    if [ -f "$JOB_FOLDER"/job/type ] && [ -f "$JOB_FOLDER"/job/job_id ]; then
        if grep -q "direct" "$JOB_FOLDER"/job/type; then
            JOB_PID=$(ps -o sid= -p "$(cat "$JOB_FOLDER"/job/job_id)")
            if [ -n "$JOB_PID" ]; then
                kill -9 "$JOB_PID"
            fi
        fi
    fi
done
