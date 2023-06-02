#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Get the job name, task ID, and series
if [[ "$1" =~ '/'$ ]]; then
    export FIRST_ARGUMENT="$1"
else
    export FIRST_ARGUMENT="$1/"
fi
export FIRST_ARGUMENT_=$(echo "$FIRST_ARGUMENT" | cut -f1 -d/)
if [[ "$FIRST_ARGUMENT_" =~ ':'$ ]]; then
    export FIRST_ARGUMENT_="$FIRST_ARGUMENT_"
else
    export FIRST_ARGUMENT_="$FIRST_ARGUMENT_:"
fi
export SERIES_NAME=$(echo "$FIRST_ARGUMENT" | cut -f2 -d/)
export JOB_NAME=$(echo "$FIRST_ARGUMENT_" | cut -f1 -d:)
export TASK_ID=$(echo "$FIRST_ARGUMENT_" | cut -f2 -d:)

# Make sure a series name was provided
if [ -z "$SERIES_NAME" ]; then
    echo "You must provide the series name with a slash (/<series_name>)." 1>&2
    exit 1
fi

# Fetch the $SERIES_NAME series log
if [ -z "$JOB_NAME" ]; then
    echo "Fetching the $SERIES_NAME series log of the last job..." 1>&2
    if [ -f "./output/named/_latest/job/.array" ]; then
        if [ -z "$TASK_ID" ]; then
            echo "You must provide the task ID with a colon (:<task_id>) since this job was run as a job array." 1>&2
            exit 1
        else
            export LOG_PATH=./output/named/_latest/$TASK_ID/series/$SERIES_NAME.csv
        fi
    else
        export LOG_PATH=./output/named/_latest/series/$SERIES_NAME.csv
    fi
    if [ -f "$LOG_PATH" ]; then
        cat "$LOG_PATH"
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
else
    echo "Fetching the $SERIES_NAME series log of the '$JOB_NAME' job..." 1>&2
    if [ -f "./output/named/$JOB_NAME/job/.array" ]; then
        if [ -z "$TASK_ID" ]; then
            echo "You must provide the task ID with a colon (:<task_id>) since this job was run as a job array." 1>&2
            exit 1
        else
            export LOG_PATH=./output/named/$JOB_NAME/$TASK_ID/series/$SERIES_NAME.csv
        fi
    else
        export LOG_PATH=./output/named/$JOB_NAME/series/$SERIES_NAME.csv
    fi
    if [ -f "$LOG_PATH" ]; then
        perl -pe 's/((?<=,)|(?<=^)),/ ,/g;' <"$LOG_PATH" | column -t -s, | less -S
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
fi
