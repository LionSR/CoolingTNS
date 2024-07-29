#!/bin/bash

source config.sh

COMMON_NAME="$(basename $0)(${PROBLEM},${METHOD},${SEARCH_METHOD})"

# Define a function to run the generalized script with specific parameters
run_job() {
    export N=$1
    export STEPS=$2
    export COUPLING=${3:-XX}
    export TAU=${4:-0.1}  # Only used for MPO method

    sbatch --job-name="${COMMON_NAME}" --export=ALL JobOptimizeCooling.sh
}

# Run jobs with different parameters
for N in $(seq 10 10 100); do
    if [ "$METHOD" = "MPO" ]; then
        run_job $N 200 XX 0.1
    else
        run_job $N 200 XX
    fi
done

