#!/bin/bash

source config.sh

COMMON_NAME="$(basename $0)(${PROBLEM},${METHOD})"

# Define a function to run the generalized script with specific parameters
run_job() {
    export N=$1
    export STEPS=$2
    export TE=$3
    export G=$4
    export COUPLING=${5:-XX}
    export TAU=${6:-0.1}

    sbatch --job-name="${COMMON_NAME}" --export=ALL JobCooling.sh
}

# Run jobs with different parameters
for N in $(seq 10 10 10); do
    if [ "$METHOD" = "MPO" ]; then
        run_job $N 1000 2.0 0.3 XX 0.1
    else
        run_job $N 1000 2.0 0.3 XX
    fi
done
