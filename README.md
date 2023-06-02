# ML Research Project Template

This is a project template for ML research projects. It makes it easy to build a Python-based project that can run across PennNLP servers that are managed with cluster management software like Slurm and Sun Grid Engine. 

The project is setup so that you can easily run the same project between different compute environments at PennNLP without modification and makes it easy to also move the project to run on AWS/GCP instances or other servers without cluster management software so that the same codebase can be shared with external researchers or open sourced. In other words, the project can be more reliabily run and reproduced in different compute environments.

Moreover, the project helps manage logs, store metrics like loss curves, and track experiments in an organized manner so that intermediate results, variants, and ablation studies can be easily found later on.

Finally, this project allows for easily publishing the code as a PyPI library for distribution.

## Features
* The project template includes starter code that reads in args via [click](https://pypi.org/project/click/) and logs using [loguru](https://pypi.org/project/loguru/).
* Helps you organize project dependencies (like Python libraries).
* Automatically creates, manages, and installs dependencies into to Python virtual environments when running.
* Comes with easy unified scripts for submitting a job, canceling a job, viewing logs, monitoring memory, etc. across different compute environments.
* Helps you organize environment variables and secret environment variables.
* Allows you to run the same project on Slurm, Sun Grid Engine, or directly on bare-metal servers with little to no modification of your scripts.
* Allows you to interactively open a `bash` shell into your environment to debug or even drop into a Python REPL in the middle of your code to debug WHILE the code is still running.
* Stores the outputs from all of your runs. Never lose data from a run. They get automatically organized so you can find them by date/time, by Git commit, or by the run name.
* Runs that are important can be saved an "experiment" so they can be more easily found later when reporting results in a paper or presentation.
* Helps you more easily run your codebase against different types of accelerator devices (CUDA, TPU, etc.).
* Helps you keep your code clean by linting and formating your code using [flake8](https://pypi.org/project/flake8/), [flake8-bugbear](https://github.com/PyCQA/flake8-bugbear), [black](https://pypi.org/project/black/), and [isort](https://pypi.org/project/isort/). It also enforces a max [cyclomatic complexity](https://en.wikipedia.org/wiki/Cyclomatic_complexity).
* The project includes a helpful VSCode extensions and configurations for automatically formatting, linting, generating Python docstrings, etc.
* Helps you write code with [GitHub Copilot](https://github.com/features/copilot).
* The project allows for easy building, packaging, and distributing on PyPI via [poetry](https://python-poetry.org/).

## Table of Contents

- [Setup and Configuration](#setup-and-configuration)
  * [Default configuration](#default-configuration)
  * [Choosing a Python version](#choosing-a-python-version)
  * [Configuring resources](#configuring-resources)
  * [Installing CUDA/TPU Python dependencies](#installing-cudatpu-python-dependencies)
- [Development Run](#development-run)
  * [Running on Slurm (NLPGPU)](#running-on-slurm-nlpgpu)
  * [Running on Sun Grid Engine (NLPGrid)](#running-on-sun-grid-engine-nlpgrid)
  * [Running directly on a server](#running-directly-on-a-server)
  * [Cluster commands](#cluster-commands)
  * [Log commands](#log-commands)
  * [Experiment commands](#experiment-commands)
  * [Reset commands](#reset-commands)
  * [Running JupyterLab](#running-jupyterlab)
  * [Running a HTTP server](#running-a-http-server)
- [Developing](#developing)
  * [Python requirements](#python-requirements)
  * [Pre-install and post-install scripts](#pre-install-and-post-install-scripts)
  * [Environment variables](#environment-variables)
  * [Secret environment variables](#secret-environment-variables)
  * [Customizing the interpreter](#customizing-the-interpreter)
  * [Adding an `__entry__` script](#adding-an-__entry__-script)
  * [Adding tasks and arguments](#adding-tasks-and-arguments)
  * [Logging](#logging)
  * [Using accelerator devices (CUDA, TPU, etc)](#using-accelerator-devices-cuda-tpu-etc)
  * [Debugging](#debugging)
  * [Reporting configuration](#reporting-configuration)
  * [Reporting series metrics](#reporting-series-metrics)
  * [Reporting to Weights & Biases](#reporting-to-weights--biases)
  * [Uploading artifacts](#uploading-artifacts)
  * [Writing output files](#writing-output-files)
  * [Using a job array task ID](#using-a-job-array-task-id)
  * [Serving a port with `ngrok`](#serving-a-port-with-ngrok)
  * [Serving a port with `cloudflared`](#serving-a-port-with-cloudflared)
  * [Outputting results to LaTeX][#outputting-results-to-latex]
- [Production Run](#production-run)
  * [Project environment variables](#project-environment-variables)
- [Building and Publishing to PyPI](#building-and-publishing-to-pypi)

## Setup and Configuration

There are a few quick things you'll have to do when using this project.

### Default configuration

Some configuration is already done for you.

By default, all data is written to `/nlp/data/$USER/output` to prevent overflowing your home directory. This directory is symlinked to `.cluster/output` for convenience. This directory is encoded in `.cluster/slurm/_sbatch_config.sh` and `.cluster/sge/_qsub_config.sh`.

By default, this project comes with [numpy](https://pypi.org/project/numpy/), [torch](https://pytorch.org/get-started/locally/), [tensorflow](https://pypi.org/project/tensorflow/), [jax](https://pypi.org/project/jax/), [transformers](https://pypi.org/project/transformers/), [datasets](https://pypi.org/project/datasets/), and [huggingface-hub](https://pypi.org/project/huggingface-hub/), and some others.

### Choosing a Python version


If you use [`pyenv`](https://github.com/pyenv/pyenv) to manage the version of Python you are using, you can edit `.python-version` to select which Python version to run and build the project against. Other files that utilize the Python version that may need to be updated are `README.md`, `.vscode/settings.json`, and `pyproject.toml`.

### Configuring resources

If you are running in a cluster (Slurm or Sun Grid Engine), you must configure computational resources required by your application (memory / CPUs / GPUs / etc).

You can do this at `.cluster/slurm/_sbatch_config.sh` and `.cluster/sge/_qsub_config.sh`.

You don't have to manually manage and allocate resources when directly running on a server, but it's worth looking at some of the configuration options in `.cluster/direct/_direct_config.sh` that may check for minimum amounts of resources.

### Installing CUDA/TPU Python dependencies

Libraries like PyTorch, TensorFlow, and JAX might require a specific versions of their libraries on your environment (due to needing a particular version of CUDA).

Check `src/requirements-accelerator-device.txt` to make sure these requirements are compatible with the server you are running if you are running on cluster management software or directly on a server.

If you are doing a production run on a machine that already has PyTorch, TensorFlow, CUDA, etc. installed, you can run this project without having it install these libraries. See [Project Environment Variables](#project-environment-variables) for more information how to do that.

## Development Run

When running during development, you can run on cluster management software, report configuration and metrics, save and track runs and experiments, run array jobs, debug, etc. This is all useful when performing research and recording experiment data for research.

### Running on Slurm (NLPGPU)

To run on SLURM you will use scripts found under `.cluster/slurm/`.


### Running on Sun Grid Engine (NLPGrid)

To run on Sun Grid Engine you will use scripts found under `.cluster/sge/`.

### Running directly on a server

To run directly on a server you will use scripts found under `.cluster/direct/`.

Note: The server must have Python 3.10+, `bash`, `curl`, `tar`, and [`parallel`](https://www.gnu.org/software/parallel/). While Linux is supported, macOS is not supported for running directly on a server.

### Cluster commands 

To run a command in the background:
```bash
.cluster/<cluster_type>/submit.sh <job_name> <task_name> [<args>...]
```

To run a command interactively:
```bash
.cluster/<cluster_type>/interactive_submit.sh <job_name> <task_name> [<args>...]
```

To check the status of the last job or a specific job:
```bash
.cluster/<cluster_type>/status.sh [<job_name>]
```

To check the status of all jobs:
```bash
.cluster/<cluster_type>/status_all.sh
```

To check the cancel the last job or a specific job:
```bash
.cluster/<cluster_type>/cancel.sh [<job_name>]
```

To check the cancel all jobs:
```bash
.cluster/<cluster_type>/cancel_all.sh
```

To open a shell to the machine of the last job or a specific job:
```bash
.cluster/<cluster_type>/shell.sh [<job_name>]
```

To open `htop` on the machine of the last job or a specific job:
```bash
.cluster/<cluster_type>/htop.sh [<job_name>]
```

To open `nvidia-smi` on the machine of the last job or a specific job:
```bash
.cluster/<cluster_type>/nvidia-smi.sh [<job_name>]
```

### Log commands

There are various commands for displaying logs. For job array jobs on a cluster, you must also specify the `<task_id>` of the specific job array task.

To see the submission log from submitting the job to the cluster of the last job or a specific job:
```bash
.cluster/submission_log.sh [<job_name>]
```

To see the install log from installing requirements of the last job or a specific job:
```bash
.cluster/install_log.sh [<job_name>[:<task_id>]]
```

To see the arguments of the last job or a specific job:
```bash
.cluster/args_log.sh [<job_name>[:<task_id>]]
```

To see the standard out log of the last job or a specific job:
```bash
.cluster/stdout_log.sh [<job_name>[:<task_id>]]
```

To see the standard error log of the last job or a specific job:
```bash
.cluster/stderr_log.sh [<job_name>[:<task_id>]]
```

To see the reported configuration log of the last job or a specific job:
```bash
.cluster/config_log.sh [<job_name>[:<task_id>]]
```

To see the reported series metrics log of the last job or a specific job:
```bash
.cluster/series_log.sh [<job_name>[:<task_id>]/<series_name>]
```

### Experiment commands

To tag the last job or a specific job as an experiment:
```bash
.cluster/tag.sh [<job_name>]
```

### Reset commands

To reset the virtual environment built for a cluster:
```bash
.cluster/<cluster_type>/reset_venv.sh
```

To reset and clean all data from runs (except runs tagged as experiments which will be kept):
```bash
.cluster/reset.sh
```

To do a full reset and clean all data from runs (including experiments) and reset all virtual environments:
```bash
.cluster/full_reset.sh
```

### Running JupyterLab

To run JupyterLab against the project:
```bash
.cluster/<cluster_type>/submit.sh jupyter_run jupyter [--hostname subdomain.example.com] [--password password]
```

### Running a HTTP server

To run a HTTP server against the project:
```bash
.cluster/<cluster_type>/submit.sh http_run http-server [--hostname subdomain.example.com]
```

## Developing

### Python requirements

You can define Python requirements in `src/requirements.txt` with requirements that depend on accelerator devices in `src/requirements-accelerator-device.txt` and requirements that are CPU alternatives in `src/requirements-cpu.txt`.

### Pre-install and post-install scripts

You can add scripts that will be run before and after installing Python requirements by adding scripts at `src/preinstall.sh` and `src/postinstall.sh`.

The output of these logs will be logged to the install log.

You can run code specific to a certain task by using an if-statement (`<task_name>` here should use underscores instead of dashes):
```bash
if [ "$PROJECT_SAFE_TASK" == "<task_name>" ]; then
    ...
fi
```

### Environment variables

You can add custom environment variables under `src/.env`.

### Secret environment variables

You can add secret environment variables (for API keys or secret tokens, etc.) by adding a file `src/.secrets.env`. This file will not get checked in by `git`.

### Customizing the interpreter

If you want to customize the command used to run the `__main__.py` script, you can define `COMMAND_PRE` or `COMMAND_POST` in `src/.env`. By default, `COMMAND_PRE='python3'` and `COMMAND_POST` is empty. 

If you want to customize the command used to run the `__main__.py` script for specific tasks, you can define `COMMAND_TASK_<task_name>_PRE` and `COMMAND_TASK_<task_name>_POST` environment variables (`<task_name>` here should use underscores instead of dashes).

### Adding an `__entry__` script

You can execute Python code in the program before anything is run (even before CLI args are parsed by `click`), by adding an `src/__entry__.py` file. Code in this file will be executed as early as possible. 

### Adding tasks and arguments

You can register new tasks in `src/__cli__.py` and adding a new module under `src/tasks/`. See the `hello_world` task as an example. 

The tasks use [click](https://click.palletsprojects.com/en/8.1.x/) for argument parsing.

If running on a PennNLP cluster, you can copy input files to the SSD drive for faster read speeds like so:
```python3
from project.pennnlp import copy_files_to_ssd
new_A_file_path, new_B_file_path, ... =  copy_files_to_ssd(A_file_path, B_file_path, ...)
```

### Logging

We use [loguru](https://loguru.readthedocs.io/en/stable/api/logger.html) to log. You can import the logger with:
```python3
from loguru import logger
```

and you can log with:
```python3
logger.info("Hello.")
```

You can set the `LOGURU_LEVEL` environment variable in `src/.env` to control the level of logging.

You can find the various levels of logging [here](https://loguru.readthedocs.io/en/stable/api/logger.html#levels).

Use ANSI Colors VSCode extension to view logs in preview mode (use the `F1` key in VSCode).

### Using accelerator devices (CUDA, TPU, etc)

The following methods help access accelerator devices for various libraries:

```python3
from project import get_jax_cpu_device
from project import get_jax_device
from project import get_jax_devices
from project import get_tf_cpu_device
from project import get_tf_device
from project import get_tf_devices
from project import get_torch_cpu_device
from project import get_torch_device
from project import get_torch_devices
```

### Debugging

The following methods / imports help with debugging:
```python3
from project import debugger
from project import bash
from project import context
```

Call `debugger()` to open a Python REPL at the current location. Call `bash()` to open a bash shell at the current location. Use `context` as a global dictionary to store global information helpful for debugging you may need to pass around module to module.

### Reporting configuration

You can report any configuration options of the current job with:
```python3
from project import reporter
reporter.set(key, value) # where key can be a nested key using dot notation
```

You can get configuration options of the current job with:
```python3
from project import reporter
reporter.get(key, value) # where key can be a nested key using dot notation
```

### Reporting series metrics

You can report a series of metrics to a CSV file like so:
```python3
from project import reporter
reporter.series("training_metrics", {
    "epoch": epoch,
    "training_loss": loss,
    "validation_loss": validation_loss,
    "validation_accuracy": accuracy,
})
```

### Reporting to Weights & Biases

First, make sure you have set `WANDB_ENTITY`, `WANDB_MODE`, and `WANDB_API_KEY` in `src/.secrets.env`.

When [logging](#logging), [reporting configuration](#reporting-configuration), and [reporting series metrics](#reporting-series-metrics), the data will automatically be synced to Weights & Biases.

### Uploading artifacts

To upload an artifact (a file or directory) to Google Drive, you can set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account JSON file and do:
```python3
from project import reporter
reporter.upload("path_to_file", "google_drive_target_folder_identifier", service="drive")
```

To upload an artifact (a file or directory) to AWS S3, you can set the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables and do:
```python3
from project import reporter
reporter.upload("path_to_file", "s3_uri", service="s3")
```

To upload an artifact (a file or directory) to Weights & Biases, you can follow the setup in [Reporting to Weights & Biases](#reporting-to-weights--biases) and do:
```python3
from project import reporter
reporter.upload("path_to_file", type="dataset|model") # Choose either 'dataset' or 'model' as the type depending on what you are uploading
```

### Writing output files

You can write files to the path stored in the `PROJECT_WRITE_DIR` environment variable. The path in the `PROJECT_CACHE_DIR` environment variable can be used as a cache directory for downloading models to (for examples the cache directory for the `transformers` library).

If you want to create a persistent directory (one that keeps data between jobs, for example a persistent directory may store training checkpoints so the training can be resumed in a future job from the latest checkpoint):
```python3
from project import get_persistent_dir

# The directory will remain persistent between jobs that have the exact same value for the `config_key` in the reported configuration.
dir_path = get_persistent_dir(name_of_the_directory, config_key) 
```

The [`jsonlines`](https://pypi.org/project/jsonlines/), [`h5py`](https://pypi.org/project/h5py/), [`sqlitedict`](https://pypi.org/project/sqlitedict/), and [`dataclasses-sqlitedict`](https://pypi.org/project/dataclasses-sqlitedict/) packages are available for reading and writing files.

### Using a job array task ID

If you are running a job array, you can access the current job array task ID by reading the `PROJECT_TASK_ID` environment variable in Python.

### Serving a port with `cloudflared`
If you want to use a custom hostname, be sure to login with:
```bash
curl -L https://github.com/cloudflare/cloudflared/releases/download/2023.5.1/cloudflared-linux-amd64 -o /tmp/cloudflared && chmod +x /tmp/cloudflared && /tmp/cloudflared tunnel login && rm -rf /tmp/cloudflared
```

Otherwise, you don't need to login.

Then, use:
```python
from project import run_cloudflared
url = run_cloudflared(port, hostname=None)
```

### Serving a port with `ngrok`

First, make sure you have set `NGROK_AUTHTOKEN` in `src/.secrets.env`.

Then, use:
```python
from project import run_ngrok
url = run_ngrok(port, hostname=None)
```

### Outputting results to LaTeX

The [`latextable`](https://pypi.org/project/latextable/) and [`tikzplotlib`](https://pypi.org/project/tikzplotlib/) packages are available to easily export data and plots to LaTeX format. Alternatively, [this guide](https://ercanozturk.org/2017/12/16/python-matplotlib-plots-in-latex/) for using LaTex directly in `matplotlib` and exporting to PDF may be an alternative to `tikzplotlib`.

## Production Run

A production run is meant for running the project by an end user. An end user is someone who might not be running on PennNLP servers, does not need naming of jobs, log management, experimentation tracking, etc. If you are handing the project off to external researchers to use or open sourcing it, you should include instructions to guide them to do a production run and not have them directly run scripts within `.cluster`.

A production run can be run with:
```bash
./run.sh <task_name> [<args>...]
```

The main dependencies are Python 3.10+, `bash`, `curl`, and `tar`. Both macOS and Linux should be supported for production runs.

### Project environment variables

When doing a production run you must specify in `project.env`:
```
export PROJECT_DATA="<path_to_data_dir>" # The path to a directory where (possibly large) data files can be written.
```

The following can be specified if you want to be explicit, otherwise these can be inferred from the environment:
```
export PROJECT_ACCELERATOR_TYPE="<accelerator_type>" # The type of accelerator device: cpu, cuda, tpu, etc.
export PROJECT_VISIBLE_ACCELERATOR_DEVICES="<device_indexes>" # A comma separated list of visible accelerator devices.
export PROJECT_DISABLE_ACCELERATOR_REQUIREMENTS="1" # If set, accelerator device Python requirements will not be installed. Use if running in an environment where PyTorch / TensorFlow / JAX / etc. are already pre-installed on the system and you want to use that version.
export PROJECT_DISABLE_TUNNEL="1" # If set, binaries needed for tunneling a port like `ngrok` and `cloudflared` will not be installed. 
```

If you want to override the accelerator device requirements in `src/requirements-accelerator-device.txt`, add a `src/requirements-<PROJECT_ACCELERATOR_TYPE>.txt` file with custom requirements. If present, `src/requirements-<PROJECT_ACCELERATOR_TYPE>.txt` will always be installed instead of `src/requirements-accelerator-device.txt`.

## Building and Publishing to PyPI

This project can build, package, and publish with [poetry](https://python-poetry.org/).

Simply use `./package.sh` to build and package the project into distributable wheels. This command will output wheels to the `dist/` folder in the project root.

Then, use `./package_publish.sh <args>` to publish the package to PyPI where are the `<args>` are the [arguments of the `poetry publish` subcommand](https://python-poetry.org/docs/cli/#publish).

The metadata for the package can be found and modified in `pyproject.toml`.
