#!/bin/bash

# Adapted from: https://stackoverflow.com/questions/1715137/what-is-the-best-way-to-ensure-only-one-instance-of-a-bash-script-is-running

function lockfile_waithold() {
   declare -ir time_beg=$(date '+%s')
   declare -ir time_max=7200

   while ! (
      set -o noclobber
      echo -e "DATE:$(date)\nUSER:$(whoami)\nPID:$$" \  >.cluster/.lock
   ) 2>/dev/null; do
      if [ $(($(date '+%s') - ${time_beg})) -gt ${time_max} ]; then
         echo "Error: waited too long for lock file .cluster/.lock" 1>&2
         return 1
      fi
      sleep 2
   done

   return 0
}

function lockfile_release() {
   rm -f .cluster/.lock
}

if ! lockfile_waithold; then
   exit 1
fi
trap lockfile_release EXIT
