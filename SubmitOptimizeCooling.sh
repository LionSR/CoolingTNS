#!/bin/bash

source config.sh

COMMON_NAME="$(basename $0)(${PROBLEM},${METHOD},${SEARCH_METHOD})"

# Define a function to run the generalized script with specific parameters
run_job() {
    export N=$1
    export STEPS=$2
    export COUPLING=${3:-XX}
    export TAU=${4:-0.1}
    export DMAX=${5:-20}

    sbatch --job-name="${COMMON_NAME}" --export=ALL --array=0-10 JobOptimizeCooling.sh
}

# Run jobs with different parameters
for N in $(seq 10 10 100); do
    for DMAX in 20 40 60; do
        if [ "$METHOD" = "MPO" ]; then
            run_job $N 200 XX 0.1 $DMAX
        else
            run_job $N 200 XX 0.1 $DMAX
        fi
    done
done

