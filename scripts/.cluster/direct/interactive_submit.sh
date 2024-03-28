#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Change directory to project root location
cd ../../

# Check if there is already a job running
if ( (.cluster/direct/status_all.sh | grep -q "$USER") 2>/dev/null); then
    echo -e "WARNING: There is already a job running!\n"
    .cluster/direct/status_all.sh

    # Confirm
    echo ""
    read -r -p "Are you sure you want to submit another job? [y/N] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        :
    else
        exit 1
    fi
elif (
    # shellcheck disable=SC2009
    ps -aux | grep -v "grep" | grep -q .cluster/direct/_direct_config.sh
); then
    echo -e "WARNING: There is already a job running! It is still initializing...\n"

    # Confirm
    echo ""
    read -r -p "Are you sure you want to submit another job? [y/N] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        :
    else
        exit 1
    fi
fi

# Get the job name
if [ -z "$1" ]; then
    echo "You must submit with a job name as the first argument to this script."
    exit 1
else
    export JOB_NAME="$1"
fi

# Submit
mkdir -p .cluster/direct/.last_job
export PROJECT_CURRENT_DATE=$(TZ=America/New_York date +%Y-%m-%d-%T)
export PROJECT_CURRENT_COMMIT=$(git rev-parse --verify HEAD 2>/dev/null || echo "_uncommited")
export PROJECT_JOB_NAME="$JOB_NAME"
rm -rf .cluster/.lock
export PROJECT_INTERACTIVE=1
touch .cluster/direct/.last_job/.interactive
echo -n "" >.cluster/direct/.last_job/submission.out
shift
.cluster/direct/_direct_config.sh "$@"
