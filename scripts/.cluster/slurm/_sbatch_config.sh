#!/bin/bash
#
#SBATCH --time=60:00:00
#SBATCH --partition=p_nlp
#SBATCH --output=.cluster/slurm/.last_job/submission.out
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --mem=10G
#SBATCH --gpus=2

# Source the user's bashrc
# shellcheck disable=SC1090
source ~/.bashrc

# Mark that we are using slurm
export PROJECT_CLUSTER=1
export PROJECT_CLUSTER_TYPE=slurm

# Set slurm dependent environment variables
export PROJECT_VENV=.venv/slurm
export PROJECT_DATA=/nlp/data/$USER
export PROJECT_CACHE_DIR=$PROJECT_DATA/.cache
export PROJECT_JOB_NAME=$SLURM_JOB_NAME
export PROJECT_TASK_ID=$SLURM_ARRAY_TASK_ID

# Set up global cache directories
if [ ! -e "$PROJECT_CACHE_DIR/huggingface_cache" ]; then
    ln -s /nlp/data/huggingface_cache "$PROJECT_CACHE_DIR/huggingface_cache"
fi
if [ ! -e "$PROJECT_CACHE_DIR/sentence_transformers_cache" ]; then
    ln -s /nlp/data/huggingface_cache/sentence_transformers "$PROJECT_CACHE_DIR/sentence_transformers_cache"
fi

# Change directory to submit location
cd "$SLURM_SUBMIT_DIR" || exit

# Store the slurm last job information
cp .cluster/slurm/_sbatch_config.sh .cluster/slurm/.last_job/resources
echo $PROJECT_CLUSTER_TYPE >.cluster/slurm/.last_job/type
echo "$PROJECT_JOB_NAME" >.cluster/slurm/.last_job/job_name
if [ -z "$PROJECT_TASK_ID" ]; then
    echo "$SLURM_JOBID" >.cluster/slurm/.last_job/job_id
else
    echo "$SLURM_ARRAY_JOB_ID" >.cluster/slurm/.last_job/job_id
fi
echo "$SLURM_JOB_NODELIST" >.cluster/slurm/.last_job/nodelist
echo "$PROJECT_CURRENT_DATE" >.cluster/slurm/.last_job/date
echo "$PROJECT_CURRENT_COMMIT" >.cluster/slurm/.last_job/commit

# Run the boot script
.cluster/_boot.sh "$@"
