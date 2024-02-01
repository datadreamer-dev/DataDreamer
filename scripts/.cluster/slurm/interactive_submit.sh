#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Change directory to project root location
cd ../../

# Check if there is already a job running
if ( (.cluster/slurm/status_all.sh | grep -q "$USER") 2>/dev/null); then
    echo -e "WARNING: There is already a job running!\n"
    .cluster/slurm/status_all.sh

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
mkdir -p .cluster/slurm/.last_job
export PROJECT_CURRENT_DATE=$(TZ=America/New_York date +%Y-%m-%d-%T)
export PROJECT_CURRENT_COMMIT=$(git rev-parse --verify HEAD 2>/dev/null || echo "_uncommited")
rm -rf .cluster/.lock
export SBATCH_ARGS=$(grep -E '^#SBATCH' .cluster/slurm/_sbatch_config.sh | sed -E 's/\s*#SBATCH\s*//g' | grep -v '^--output' | tr '\n' ' ')
export PROJECT_INTERACTIVE=1
touch .cluster/slurm/.last_job/.interactive
echo -n "" >.cluster/slurm/.last_job/submission.out
shift
# shellcheck disable=SC2086
srun -u --job-name="$JOB_NAME" $SBATCH_ARGS --pty .cluster/slurm/_sbatch_config.sh "$@"
