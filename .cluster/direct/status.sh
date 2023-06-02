#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Fetch the status
if [ -z "$1" ]; then
    echo "Fetching the status of the last job..." 1>&2
    if [ -f "../output/named/_latest/job/job_id" ]; then
        JOB_PID=$(ps -o sid= -p "$(cat ../output/named/_latest/job/job_id)")
        if [ -n "$JOB_PID" ]; then
            # shellcheck disable=SC2086
            ps --forest -wwo user,sid,pid,stat,start,time,%cpu,%mem,args -g $JOB_PID 2>/dev/null
        fi
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
else
    echo "Fetching the status of the '$1' job..." 1>&2
    if [ -f "../output/named/$1/job/job_id" ]; then
        JOB_PID=$(ps -o sid= -p "$(cat "../output/named/$1/job/job_id")")
        if [ -n "$JOB_PID" ]; then
            # shellcheck disable=SC2086
            ps --forest -wwo user,sid,pid,stat,start,time,%cpu,%mem,args -g $JOB_PID 2>/dev/null
        fi
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
fi
