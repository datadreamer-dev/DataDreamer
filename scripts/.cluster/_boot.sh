#!/bin/bash

# Run setup
. ./.cluster/_setup.sh

# Run the python script
if [ "$PROJECT_CLUSTER_TYPE" == "slurm" ] && [ "$PROJECT_INTERACTIVE" != "1" ]; then
    srun -u .cluster/_command.sh "$@"
else
    .cluster/_command.sh "$@"
fi
