#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")" || exit

# Fetch the nodelist
if [ -z "$1" ]; then
    echo "Fetching the compute environment of the last job..." 1>&2
    if [ -f "../output/named/_latest/job/nodelist" ]; then
        export NODELIST=$(cat ../output/named/_latest/job/nodelist)
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
else
    echo "Fetching the compute environment of the '$1' job..." 1>&2
    if [ -f "../output/named/$1/job/nodelist" ]; then
        export NODELIST=$(cat ../output/named/"$1"/job/nodelist)
    else
        echo "Job does not exist. If you just submitted the job, try again in a few seconds." 1>&2
        exit 1
    fi
fi

echo -e "Opening a shell to the $NODELIST compute environment...\n(note: the shell will only remain open for 3 hours, this is a time limit to prevent hanging resources)" 1>&2
sleep 2
qrsh -now no -cwd -pty y -N shell -l h_rt=3:00:00 -pe parallel-onenode 1 -l mem=100M -l h="$NODELIST" /bin/bash -c "echo ""; cd ../../; /bin/bash -i"
