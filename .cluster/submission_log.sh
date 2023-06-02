#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Fetch the submission log
if [ -z "$1" ]; then
    echo "Fetching the submission log of the last job..." 1>&2
    if [ -f "./output/named/_latest/job/submission.out" ]; then
        cat ./output/named/_latest/job/submission.out
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
else
    echo "Fetching the submission log of the '$1' job..." 1>&2
    if [ -f "./output/named/$1/job/submission.out" ]; then
        cat ./output/named/"$1"/job/submission.out
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
fi
