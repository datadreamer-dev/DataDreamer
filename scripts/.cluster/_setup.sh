#!/bin/bash

if [ "$PROJECT_INTERACTIVE" == "1" ]; then
    exec 3>&1 4>&2
    trap 'exec 2>&4 1>&3' 0 1 2 3
    exec 1> >(tee -a .cluster/"$PROJECT_CLUSTER_TYPE"/.last_job/submission.out) 2> >(tee -a .cluster/"$PROJECT_CLUSTER_TYPE"/.last_job/submission.out >&2)
fi

#########################
# Attain lock
source .cluster/_lock.sh
#########################

if [ "$PROJECT_CLUSTER" == "1" ]; then
    echo "Initializing job..."
fi

# Define helper functions
function create_alias() {
    if [ ! -e "$2" ]; then
        mkdir -p "$(dirname "$2")"
        ln -s "$1" "$2"
    fi
}
function create_alias_delete() {
    if [ -e "$2" ]; then
        (rm "$2" 1>/dev/null 2>/dev/null) || rm -rf "$2"
    fi
    mkdir -p "$(dirname "$2")"
    ln -s "$1" "$2"
}

# Create directories
mkdir -p "$PROJECT_DATA"
export PROJECT_DATA="$(realpath "$PROJECT_DATA")"
mkdir -p "$PROJECT_DATA"
export PROJECT_NAME=$(basename "$(pwd)")
export PROJECT_DATA_VENV=$PROJECT_DATA/.venv/$PROJECT_NAME

if [[ $PROJECT_DISABLE_ACCELERATOR_REQUIREMENTS != "1" ]]; then
    # Activate the virtual environment
    export PROJECT_DATA_VENV_DIR=$PROJECT_DATA_VENV/$PROJECT_VENV
    mkdir -p "$PROJECT_DATA_VENV_DIR"
    create_alias_delete "$PROJECT_DATA_VENV_DIR" "$PROJECT_VENV"
    python3 -m venv "$PROJECT_DATA_VENV_DIR" || exit 1
    source "$PROJECT_VENV"/bin/activate
fi

# Create cache directory
mkdir -p "$PROJECT_CACHE_DIR"

if [ "$PROJECT_CLUSTER" == "1" ]; then
    # Set environment variables
    export PROJECT_DATA_OUTPUT=$PROJECT_DATA/output/$PROJECT_NAME
    export PROJECT_DATA_LOCAL_OUTPUT=.cluster/output
    export PROJECT_DATA_OUTPUT_PERSISTENT_DATA=$PROJECT_DATA_OUTPUT/persistent_data
    export PROJECT_DATA_OUTPUT_DATED_ROOT=$PROJECT_DATA_OUTPUT/dated/$PROJECT_CURRENT_DATE
    if [ -z "$PROJECT_TASK_ID" ]; then
        export PROJECT_DATA_OUTPUT_DATED=$PROJECT_DATA_OUTPUT_DATED_ROOT
    else
        export PROJECT_DATA_OUTPUT_DATED=$PROJECT_DATA_OUTPUT_DATED_ROOT/$PROJECT_TASK_ID
    fi
    export PROJECT_DATA_OUTPUT_DATED_JOB=$PROJECT_DATA_OUTPUT_DATED_ROOT/job
    export PROJECT_DATA_OUTPUT_DATED_SERIES=$PROJECT_DATA_OUTPUT_DATED/series
    export PROJECT_DATA_OUTPUT_DATED_DATA=$PROJECT_DATA_OUTPUT_DATED/data
    export PROJECT_DATA_OUTPUT_COMMITTED_AND_DATED=$PROJECT_DATA_OUTPUT/committed/$PROJECT_CURRENT_COMMIT/$PROJECT_CURRENT_DATE
    export PROJECT_DATA_OUTPUT_NAMED=$PROJECT_DATA_OUTPUT/named
    export PROJECT_DATA_OUTPUT_NAMED_LATEST=$PROJECT_DATA_OUTPUT_NAMED/_latest
    export PROJECT_DATA_OUTPUT_NAMED_JOB_NAME=$PROJECT_DATA_OUTPUT_NAMED/$PROJECT_JOB_NAME
    export PROJECT_DATA_OUTPUT_EXPERIMENTS=$PROJECT_DATA_OUTPUT/experiments
    export PROJECT_WRITE_DIR=$PROJECT_DATA_OUTPUT_DATED_DATA
    export PROJECT_SERIES_DIR=$PROJECT_DATA_OUTPUT_DATED_SERIES
    export PROJECT_ARGS_FILE=$PROJECT_DATA_OUTPUT_DATED/args.json
    export PROJECT_CONFIG_FILE=$PROJECT_DATA_OUTPUT_DATED/config.json
    export PROJECT_NOTES_FILE=$PROJECT_DATA_OUTPUT_DATED/notes.md
    export PROJECT_INSTALL_FILE=$PROJECT_DATA_OUTPUT_DATED/install.out
    export PROJECT_STDOUT_FILE=$PROJECT_DATA_OUTPUT_DATED/stdout.out
    export PROJECT_STDERR_FILE=$PROJECT_DATA_OUTPUT_DATED/stderr.out

    # Create output directories
    mkdir -p "$PROJECT_DATA_OUTPUT"
    create_alias_delete "$PROJECT_DATA_OUTPUT" $PROJECT_DATA_LOCAL_OUTPUT
    mkdir -p "$PROJECT_DATA_OUTPUT_PERSISTENT_DATA"
    mkdir -p "$PROJECT_DATA_OUTPUT_DATED"
    create_alias_delete "$(pwd)"/.cluster/"$PROJECT_CLUSTER_TYPE"/.last_job "$PROJECT_DATA_OUTPUT_DATED_JOB"
    mkdir -p "$PROJECT_DATA_OUTPUT_DATED_SERIES"
    mkdir -p "$PROJECT_DATA_OUTPUT_DATED_DATA"
    create_alias_delete "$PROJECT_DATA_OUTPUT_DATED_ROOT" "$PROJECT_DATA_OUTPUT_COMMITTED_AND_DATED"
    create_alias_delete "$PROJECT_DATA_OUTPUT_DATED_ROOT" "$PROJECT_DATA_OUTPUT_NAMED_LATEST"
    create_alias_delete "$PROJECT_DATA_OUTPUT_DATED_ROOT" "$PROJECT_DATA_OUTPUT_NAMED_JOB_NAME"
    mkdir -p "$PROJECT_DATA_OUTPUT_EXPERIMENTS"
    echo "null" >"$PROJECT_ARGS_FILE"
    echo "null" >"$PROJECT_CONFIG_FILE"
    echo -e "**Enter any custom notes you want to save about this run below in Markdown format:**\n------------\n" >"$PROJECT_NOTES_FILE"
    touch "$PROJECT_INSTALL_FILE"
    touch "$PROJECT_STDOUT_FILE"
    touch "$PROJECT_STDERR_FILE"
    if [ -n "$PROJECT_TASK_ID" ]; then
        touch "$PROJECT_DATA_OUTPUT_DATED_JOB"/.array
    fi

    # Finished initializing
    echo "Done initializing job."

    # Hard copy the last_job folder
    (rm "$PROJECT_DATA_OUTPUT_DATED_JOB" 1>/dev/null 2>/dev/null) || rm -rf "$PROJECT_DATA_OUTPUT_DATED_JOB"
    cp -r "$(pwd)"/.cluster/"$PROJECT_CLUSTER_TYPE"/.last_job "$PROJECT_DATA_OUTPUT_DATED_JOB"

    # Redirect output from here on
    if [ "$PROJECT_INTERACTIVE" == "1" ]; then
        exec 2>&4 1>&3
        exec 3>&1 4>&2
        trap 'exec 2>&4 1>&3' 0 1 2 3
        exec 1> >(tee -a "$PROJECT_INSTALL_FILE") 2> >(tee -a "$PROJECT_INSTALL_FILE" >&2)
    else
        exec 3>&1 4>&2
        trap 'exec 2>&4 1>&3' 0 1 2 3
        exec 1>>"$PROJECT_INSTALL_FILE" 2>&1
    fi
else
    export PROJECT_WRITE_DIR=$PROJECT_DATA/data
    mkdir -p "$PROJECT_WRITE_DIR"
fi

# Set custom environment variables
export COMMAND_PRE="python3 -m"
export COMMAND_ENTRYPOINT="src.__main__"
export COMMAND_POST=""
set -o allexport
if [ -f "src/.env" ]; then
    source src/.env
fi
if [ -f "src/.secrets.env" ]; then
    source src/.secrets.env
fi
set +o allexport
PROJECT_SAFE_TASK=$(echo "$1" | tr '-' '_') # Replace dashes with underscore
COMMAND_TASK_PRE_ENV_NAME="COMMAND_TASK_${PROJECT_SAFE_TASK}_PRE"
COMMAND_TASK_ENTRYPOINT_ENV_NAME="COMMAND_TASK_${PROJECT_SAFE_TASK}_ENTRYPOINT"
COMMAND_TASK_POST_ENV_NAME="COMMAND_TASK_${PROJECT_SAFE_TASK}_POST"
eval COMMAND_TASK_PRE=\$"$COMMAND_TASK_PRE_ENV_NAME"
eval COMMAND_TASK_ENTRYPOINT=\$"$COMMAND_TASK_ENTRYPOINT_ENV_NAME"
eval COMMAND_TASK_POST=\$"$COMMAND_TASK_POST_ENV_NAME"
if [ -n "$COMMAND_TASK_PRE" ]; then
    export COMMAND_PRE=$COMMAND_TASK_PRE
fi
if [ -n "$COMMAND_TASK_ENTRYPOINT" ]; then
    export COMMAND_ENTRYPOINT=$COMMAND_TASK_ENTRYPOINT
fi
if [ -n "$COMMAND_TASK_POST" ]; then
    export COMMAND_POST=$COMMAND_TASK_POST
fi
if [[ $PROJECT_JOB_NAME == "test" ]] || [[ $PROJECT_JOB_NAME == "test_"* ]]; then
    export COMMAND_PRE="pytest"
    export COMMAND_ENTRYPOINT=""
    # shellcheck disable=SC2199
    if [[ "${@:1}" != *.py* ]]; then
        export COMMAND_POST="src/tests"
    else
        export COMMAND_POST=""
    fi
fi

if [ "$PROJECT_SKIP_INSTALL_REQS" != "1" ]; then
    # Run a preinstall script
    if [ -f "src/preinstall.sh" ]; then
        echo "Running preinstall.sh..."
        chmod +x ./src/preinstall.sh
        . ./src/preinstall.sh
        echo "Done running preinstall.sh."
    fi

    # Install requirements
    echo "Installing requirements.txt..."

    device-requirements-files-exist() {
        [ -f "$1.txt" ] || [ -f "$1.sh" ]
    }
    install-device-requirements() {
        if [ -f "$1.txt" ]; then
            :
            python3 -m pip install --upgrade pip
            pip3 install -r src/requirements.txt -r "$1.txt" -r src/requirements-test.txt
        fi
        if [ -f "$1.sh" ]; then
            chmod +x ./"$1".sh
            ./"$1".sh
        fi
    }
    if device-requirements-files-exist "src/requirements-accelerator-device" || device-requirements-files-exist "src/requirements-$PROJECT_ACCELERATOR_TYPE"; then
        if [[ $PROJECT_DISABLE_ACCELERATOR_REQUIREMENTS == "1" ]]; then
            echo "Not using any device specific requirements-*. The system Python site-packages will be used instead, make sure you have already have these requirements installed in the system site-packages."
        elif [ -n "$CUDA_VISIBLE_DEVICES" ] || [ -n "$PROJECT_VISIBLE_ACCELERATOR_DEVICES" ] || [ -n "$PROJECT_ACCELERATOR_TYPE" ]; then
            if [ -n "$PROJECT_ACCELERATOR_TYPE" ] && device-requirements-files-exist "src/requirements-$PROJECT_ACCELERATOR_TYPE"; then
                echo "Using requirements-$PROJECT_ACCELERATOR_TYPE..."
                install-device-requirements "src/requirements-$PROJECT_ACCELERATOR_TYPE"
            elif device-requirements-files-exist "src/requirements-accelerator-device"; then
                echo "Using requirements-accelerator-device..."
                install-device-requirements "src/requirements-accelerator-device"
            fi
        elif device-requirements-files-exist "src/requirements-cpu"; then
            echo "Using requirements-cpu instead of any device specific requirements-* due to no accelerator devices visible."
            install-device-requirements "src/requirements-cpu"
        fi
    fi
    echo "Done installing requirements.txt."

    # Run a postinstall script
    if [ -f "src/postinstall.sh" ]; then
        echo "Running postinstall.sh..."
        chmod +x ./src/postinstall.sh
        . ./src/postinstall.sh
        echo "Done running postinstall.sh."
    fi

    # If GitHub CI, list exact dependencies used for this run for future reference
    if [ -n "$GITHUB_ACTIONS" ] && [ ! -f "$PROJECT_VENV/piplist.txt" ]; then
        echo "======================="
        echo "Begin: Dependency List"
        echo "======================="
        pip3 list | sort
        pip3 freeze | sort >$PROJECT_VENV/piplist.txt
        echo "======================="
        echo "End: Dependency List"
        echo "======================="
    fi
fi

#########################
# Release lock
lockfile_release
#########################
