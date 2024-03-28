#!/bin/bash

# Check for the existence of environment variables
if [ -z "$PROJECT_DATA" ]; then
    echo "You must define the 'PROJECT_DATA' environment variable in project.env. Set it to a directory where output files will be written and large data can be saved while running." 1>&2
    exit 1
fi
if [ -z "$PROJECT_ACCELERATOR_TYPE" ]; then
    if [ "$(uname)" != "Darwin" ] && (env -0 | cut -z -f1 -d= | tr '\0' '\n' | grep -q "TPU_"); then
        export PROJECT_ACCELERATOR_TYPE=tpu
        export PROJECT_VISIBLE_ACCELERATOR_DEVICES=all
        echo "Detected an environment with TPUs. For this reason, accelerator device dependencies will not be installed to a virtual environment." 1>&2
        export PROJECT_DISABLE_ACCELERATOR_REQUIREMENTS=1
    elif [ -x "$(command -v nvidia-smi)" ]; then
        export PROJECT_ACCELERATOR_TYPE=cuda
        echo "Detected an environment with CUDA GPUs." 1>&2
        export PROJECT_VISIBLE_ACCELERATOR_DEVICES=$(seq --separator="," 0 $(($(nvidia-smi --list-gpus | wc -l) - 1)))
    fi
fi
