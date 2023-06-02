#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Change directory to project root location
cd ../../

# Check if there is already a job running
if ( (.cluster/sge/status_all.sh | grep -q "$USER") 2>/dev/null); then
    echo -e "WARNING: There is already a job running!\n"
    .cluster/sge/status_all.sh

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
mkdir -p .cluster/sge/.last_job
export PROJECT_CURRENT_DATE=$(TZ=America/New_York date +%Y-%m-%d-%T)
export PROJECT_CURRENT_COMMIT=$(git rev-parse --verify HEAD 2>/dev/null || echo "_uncommited")
rm -rf .cluster/.lock
export QSUB_ARGS=$(grep -E '^#\$' .cluster/sge/_qsub_config.sh | sed -E 's/\s*#\$\s*//g' | grep -v '^-o' | grep -v '^-e' | tr '\n' ' ')
export PROJECT_INTERACTIVE=1
touch .cluster/sge/.last_job/.interactive
echo -n "" >.cluster/sge/.last_job/submission.out
shift
# shellcheck disable=SC2086
qrsh -now no -cwd -pty y -V -N "$JOB_NAME" $QSUB_ARGS .cluster/sge/_qsub_config.sh "$@"
