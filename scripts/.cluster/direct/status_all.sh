#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Fetch the status of all jobs
echo "Fetching the status of all jobs..." 1>&2
for JOB_FOLDER in ../output/dated/*/; do
    if [ -f "$JOB_FOLDER"/job/type ] && [ -f "$JOB_FOLDER"/job/job_id ] && [ -f "$JOB_FOLDER"/job/job_name ]; then
        if grep -q "direct" "$JOB_FOLDER"/job/type; then
            JOB_PID=$(ps -o sid= -p "$(cat "$JOB_FOLDER"/job/job_id)")
            if [ -n "$JOB_PID" ]; then
                echo "------------------------------------------------------------------------"
                echo "Job: $(cat "$JOB_FOLDER"/job/job_name) ($(cat "$JOB_FOLDER"/job/date))"
                echo "------------------------------------------------------------------------"
                # shellcheck disable=SC2086
                ps --forest -wwo user,sid,pid,stat,start,time,%cpu,%mem,args -g $JOB_PID 2>/dev/null
            fi
        fi
    fi
done
