import os
import sys

# Detect environments
RUNNING_IN_PYTEST = os.path.basename(sys.argv[0]) == "pytest"
RUNNING_IN_CLUSTER = "PROJECT_CLUSTER" in os.environ
