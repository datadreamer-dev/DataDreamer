#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Reset the virtual env
rm -rf ../../.venv/direct/*
rm -rf ../../.venv/direct/
