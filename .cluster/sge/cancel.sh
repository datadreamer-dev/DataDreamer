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
    :
else
    exit 1
fi

# Fetch the status
if [ -z "$1" ]; then
    echo "Canceling the last job..." 1>&2
    if [ -f "../output/named/_latest/job/job_id" ]; then
        qdel "$(cat ../output/named/_latest/job/job_id)"
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
else
    echo "Canceling the '$1' job..." 1>&2
    if [ -f "../output/named/$1/job/job_id" ]; then
        qdel "$(cat ../output/named/"$1"/job/job_id)"
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
fi
