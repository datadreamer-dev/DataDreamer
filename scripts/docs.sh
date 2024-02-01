#!/bin/bash

# Mac-specific aliases
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Use brew install coreutils gnu-sed
    shopt -s expand_aliases
    alias sed='gsed'
    alias cp='gcp'
fi

# Change directory to script location
cd "$(dirname "$0")/../" || exit

# Mark that we are in a sphinx build
export SPHINX_BUILD="1"

# Create and activate a virtual environment for docs requirements
export PROJECT_VENV=.venv_dev/docs
mkdir -p $PROJECT_VENV
python3 -m venv $PROJECT_VENV
source $PROJECT_VENV/bin/activate
pip3 install -q -U pip
pip3 install -r src/requirements.txt
pip3 install -r src/requirements-dev.txt

# Change to docs directory
cd ./docs/ || exit

# Get the project folder name and the Python package name
export PROJECT_NAME=$(basename "$(realpath ../)" | awk '{print tolower($0)}')
export PACKAGE_NAME=$(grep name ../pyproject.toml | head -n1 | cut -d'"' -f 2 | awk '{print tolower($0)}')
export REPOSITORY_URL=$(grep repository ../pyproject.toml | head -n1 | cut -d'"' -f 2)
export EXCLUDE_PATHS=$(grep autodoc_exclude_paths ../pyproject.toml | head -n1 | cut -d'"' -f 2) # Space-separated, fnmatch style
export GITHUB_USERNAME=$(echo "$REPOSITORY_URL" | cut -d'/' -f 4)
export GITHUB_REPOSITORY=$(echo "$REPOSITORY_URL" | cut -d'/' -f 5)

# Customize index.rst
sed -i 's@# documentation@'"$PROJECT_NAME documentation"'@' ./source/index.rst
sed -i 's@API Reference <#>@'"API Reference <$PACKAGE_NAME>"'@' ./source/index.rst
sed -i 's@https://pypi.org/project/#@'"https://pypi.org/project/$PACKAGE_NAME"'@' ./source/index.rst
sed -i 's@https://github.com/#/#@'"$REPOSITORY_URL"'@' ./source/index.rst
sed -i 's@#/#@'"$GITHUB_USERNAME/$GITHUB_REPOSITORY"'@' ./source/_templates/sidebar/brand.html

# Select a port and kill it
export PORT_FILE="/tmp/$PROJECT_NAME-docs-port"
if [ ! -e "$PORT_FILE" ]; then
    export PORT=$(python3 -c "import socket; sock = socket.socket(); sock.bind(('', 0)); print(sock.getsockname()[1])")
    echo "$PORT" >"$PORT_FILE"
else
    export PORT=$(cat "$PORT_FILE")
fi
lsof -ti:43761 | xargs kill -9 &>/dev/null

# Clear old build files
rm -rf ./build/
shopt -s extglob
(cd ./source/ && rm !(index.rst|conf.py|docstrings.py|__init__.py)) 2>/dev/null

# Rename the `src` directory temporarily to the true module name
rm -rf "../$PACKAGE_NAME" &>/dev/null
function docs_cleanup {
    rm -rf "../$PACKAGE_NAME" &>/dev/null
}
trap docs_cleanup EXIT
cp -rl "$(realpath ../src/)" "../$PACKAGE_NAME"

# Auto documentation from the Python source files
export SPHINX_APIDOC_OPTIONS="members,show-inheritance"
set -o noglob
# shellcheck disable=SC2086
(cd "../$PACKAGE_NAME/" && sphinx-apidoc -d 1 -fMeT -t ../docs/source/_templates/apidoc -o ../docs/source . $EXCLUDE_PATHS)
set +o noglob

# Run coverage
sphinx-build -b coverage ./source ./build

# Auto documentation from the Python source files
unset SPHINX_APIDOC_OPTIONS
set -o noglob
# shellcheck disable=SC2086
(cd "../$PACKAGE_NAME/" && sphinx-apidoc -d 1 -fMeT -t ../docs/source/_templates/apidoc -o ../docs/source . $EXCLUDE_PATHS)
set +o noglob

# Build and serve the documentation
if [[ $* == *--no-watch* ]]; then
    # Build the documentation HTML pages
    make html
    sed -i "s/Python Module Index/Index of Python Modules/g" ./build/html/py-modindex.html
    if [[ $* == *--no-serve* ]]; then
        :
    else
        # Serve the documentation
        python3 -m http.server -d ./build/html "$PORT" 2>/dev/null
    fi
else
    # Auto-document, auto-build, and serve the documentation
    sphinx-autobuild ./source ./build/html --port "$PORT" --pre-build "/bin/bash -c '(cd ../$PACKAGE_NAME/ && sphinx-apidoc -d 1 -fMeT -t ../docs/source/_templates/apidoc -o ../docs/source . $EXCLUDE_PATHS)'" --watch ../src/
fi
