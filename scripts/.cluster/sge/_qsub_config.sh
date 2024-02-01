#$ -l h_rt=60:00:00
#$ -o .cluster/sge/.last_job/submission.out
#$ -e .cluster/sge/.last_job/submission.out
#$ -cwd
#$ -l mem=1G
#$ -pe parallel-onenode 4
#$ -l h=nlpgrid12

# Source the user's bashrc
# shellcheck disable=SC1090
source ~/.bashrc

# Mark that we are using sge
export PROJECT_CLUSTER=1
export PROJECT_CLUSTER_TYPE=sge

# Set sge dependent environment variables
export PROJECT_VENV=.venv/sge
export PROJECT_DATA=/nlp/data/$USER
export PROJECT_CACHE_DIR=$PROJECT_DATA/.cache
export PROJECT_JOB_NAME=$JOB_NAME
if [ "$SGE_TASK_ID" != "undefined" ]; then
    export PROJECT_TASK_ID=$SGE_TASK_ID
fi

# Set up global cache directories
if [ ! -e "$PROJECT_CACHE_DIR/huggingface_cache" ]; then
    ln -s /nlp/data/huggingface_cache "$PROJECT_CACHE_DIR/huggingface_cache"
fi
if [ ! -e "$PROJECT_CACHE_DIR/sentence_transformers_cache" ]; then
    ln -s /nlp/data/huggingface_cache/sentence_transformers "$PROJECT_CACHE_DIR/sentence_transformers_cache"
fi

# Change directory to submit location
cd "$SGE_O_WORKDIR" || exit

# Store the sge last job information
cp .cluster/sge/_qsub_config.sh .cluster/sge/.last_job/resources
echo $PROJECT_CLUSTER_TYPE >.cluster/sge/.last_job/type
echo "$PROJECT_JOB_NAME" >.cluster/sge/.last_job/job_name
if [ -z "$PROJECT_TASK_ID" ]; then
    echo "$JOB_ID" >.cluster/sge/.last_job/job_id
else
    echo "$JOB_ID" >.cluster/sge/.last_job/job_id
fi
echo "$HOSTNAME" >.cluster/sge/.last_job/nodelist
echo "$PROJECT_CURRENT_DATE" >.cluster/sge/.last_job/date
echo "$PROJECT_CURRENT_COMMIT" >.cluster/sge/.last_job/commit

# Run the boot script
.cluster/_boot.sh "$@"
