#!/bin/bash

# Write the command arguments to a file
export PROJECT_COMMAND_PATH=$(mktemp /tmp/projectcmd.XXXXXX)
(
  printf '%s' "$COMMAND_PRE"
  printf " "
  printf '%s' "$COMMAND_ENTRYPOINT"
  printf " "

  # For each argument
  for ARG in "$@"; do
    printf "$'"
    # echo the argument, except:
    # * Replace backslashes with escaped backslashes
    # * Replace single quotes with escaped single quotes
    echo -n "$ARG" | sed -e "s/\\\\/\\\\\\\\/g;" | sed -e "s/'/\\\\'/g;"
    # echo `'`
    printf "' "
  done

  printf " "
  printf '%s' "$COMMAND_POST"
) >"$PROJECT_COMMAND_PATH"
chmod +x "$PROJECT_COMMAND_PATH"

# Run the script
if [ "$PROJECT_CLUSTER" == "1" ]; then
  if [ "$PROJECT_INTERACTIVE" == "1" ]; then
    exec 2>&4 1>&3
    script -efq "$PROJECT_STDOUT_FILE" -c "$PROJECT_COMMAND_PATH 2> >(tee -a $PROJECT_STDERR_FILE >&2)"
  else
    $PROJECT_COMMAND_PATH 1>>"$PROJECT_STDOUT_FILE" 2>>"$PROJECT_STDERR_FILE"
  fi
else
  $PROJECT_COMMAND_PATH
fi
