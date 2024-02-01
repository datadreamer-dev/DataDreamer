#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Fetch the run dir
if [ -z "$1" ]; then
    echo "Fetching the run output of the last job..." 1>&2
    if [ -e "./output/named/_latest" ]; then
        export RUN_DIR=$(readlink -f ./output/named/_latest)
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
else
    echo "Fetching the run output of the '$1' job..." 1>&2
    if [ -e "./output/named/$1" ]; then
        export RUN_DIR=$(readlink -f ./output/named/"$1")
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
fi

# Fetch the experiment dir
export JOB_NAME=$(cat "$RUN_DIR"/job/job_name)
read -r -p "What do you want to name the experiment (default: '$JOB_NAME')? " EXPERIMENT_NAME
if [ -z "$EXPERIMENT_NAME" ]; then
    export EXPERIMENT_NAME=$JOB_NAME
fi
export EXPERIMENT_DIR=./output/experiments/$EXPERIMENT_NAME

# Store the run as an experiment
if [ -e "$EXPERIMENT_DIR" ]; then
    echo "This '$EXPERIMENT_NAME' experiment already exists. Do you want to replace it?"

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
        :
    else
        exit 1
    fi

    rm "$EXPERIMENT_DIR"
fi
ln -s "$RUN_DIR" "$EXPERIMENT_DIR" && echo -e "\nDone. You can find the experiment at: $(
    cd "$EXPERIMENT_DIR" || exit
    pwd
)"
