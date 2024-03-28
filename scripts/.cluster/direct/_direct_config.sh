#!/bin/bash

################################################################
# To run a job array define array task IDs here,
# example: ARRAY_TASK_IDS=( 10 20 30 40 50)
# if only one value is provided, this job will not be considered a job array
ARRAY_TASK_IDS=(0)
MIN_MEMORY_PER_TASK=10G
MAX_PARALLEL_TASKS=8
################################################################

# Load environment variables for direct running
source project.env

# Ensure direct environment variables are provided
source .cluster/direct/_direct_env.sh

# Redirect to submission log
if [ "$PROJECT_INTERACTIVE" != "1" ]; then
    exec 3>&1 4>&2 >.cluster/direct/.last_job/submission.out 2>&1
fi

# Define task runner
TASK_RUNNER() {
    TASK_ID=$1
    ARRAY_TASK_IDS_LENGTH=$2

    # Source the user's bashrc
    # shellcheck disable=SC1090
    source ~/.bashrc

    # Mark that we are using direct
    export PROJECT_CLUSTER=1
    export PROJECT_CLUSTER_TYPE=direct

    # Set direct dependent environment variables
    export PROJECT_VENV=.venv/direct
    export PROJECT_CACHE_DIR=$PROJECT_DATA/.cache
    if [ "$ARRAY_TASK_IDS_LENGTH" == "1" ]; then
        # Only one task ID detected, therefore this is not a job array
        :
    else
        export PROJECT_TASK_ID=$TASK_ID
    fi

    # Store the direct last job information
    cp .cluster/direct/_direct_config.sh .cluster/direct/.last_job/resources
    echo $PROJECT_CLUSTER_TYPE >.cluster/direct/.last_job/type
    echo "$PROJECT_JOB_NAME" >.cluster/direct/.last_job/job_name
    echo $$ >.cluster/direct/.last_job/job_id
    hostname >.cluster/direct/.last_job/hostname
    echo "$PROJECT_CURRENT_DATE" >.cluster/direct/.last_job/date
    echo "$PROJECT_CURRENT_COMMIT" >.cluster/direct/.last_job/commit

    # Run the boot script
    shift
    shift
    .cluster/_boot.sh "$@"
}
export -f TASK_RUNNER

# Run
if [ "$PROJECT_INTERACTIVE" != "1" ]; then
    # shellcheck disable=SC1083
    parallel --memfree $MIN_MEMORY_PER_TASK --jobs $MAX_PARALLEL_TASKS TASK_RUNNER {.} "${#ARRAY_TASK_IDS[@]}" "$@" ::: "${ARRAY_TASK_IDS[@]}"
else
    # shellcheck disable=SC1083
    parallel --tty --memfree $MIN_MEMORY_PER_TASK --jobs $MAX_PARALLEL_TASKS TASK_RUNNER {.} "${#ARRAY_TASK_IDS[@]}" "$@" ::: "${ARRAY_TASK_IDS[@]}"
fi
