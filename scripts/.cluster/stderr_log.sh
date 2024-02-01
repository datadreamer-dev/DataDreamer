#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Get the job name and task ID
if [[ "$1" =~ ':'$ ]]; then
    export FIRST_ARGUMENT="$1"
else
    export FIRST_ARGUMENT="$1:"
fi
export JOB_NAME=$(echo "$FIRST_ARGUMENT" | cut -f1 -d:)
export TASK_ID=$(echo "$FIRST_ARGUMENT" | cut -f2 -d:)

# Fetch the stderr log
if [ -z "$JOB_NAME" ]; then
    echo "Fetching the stderr log of the last job..." 1>&2
    if [ -f "./output/named/_latest/job/.array" ]; then
        if [ -z "$TASK_ID" ]; then
            echo "You must provide the task ID with a colon (:<task_id>) since this job was run as a job array." 1>&2
            exit 1
        else
            export LOG_PATH=./output/named/_latest/$TASK_ID/stderr.out
        fi
    else
        export LOG_PATH=./output/named/_latest/stderr.out
    fi
    if [ -f "$LOG_PATH" ]; then
        cat "$LOG_PATH"
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
else
    echo "Fetching the stderr log of the '$JOB_NAME' job..." 1>&2
    if [ -f "./output/named/$JOB_NAME/job/.array" ]; then
        if [ -z "$TASK_ID" ]; then
            echo "You must provide the task ID with a colon (:<task_id>) since this job was run as a job array." 1>&2
            exit 1
        else
            export LOG_PATH=./output/named/$JOB_NAME/$TASK_ID/stderr.out
        fi
    else
        export LOG_PATH=./output/named/$JOB_NAME/stderr.out
    fi
    if [ -f "$LOG_PATH" ]; then
        cat "$LOG_PATH"
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
fi
