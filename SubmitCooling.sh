#!/bin/bash

source config.sh

COMMON_NAME="$(basename $0)(${PROBLEM},${METHOD})"

# Define a function to run the generalized script with specific parameters
run_job() {
    export N=$1
    export STEPS_VALUE=$2
    export TE_VALUE=$3
    export G_VALUE=$4
    additional_params="g${G_VALUE}te${TE_VALUE}"
    sbatch --job-name="${COMMON_NAME}" --export=ALL JobCooling.sh
}

# Run jobs with different parameters
for N in $(seq 10 10 100); do
    run_job $N 1000 2.0 0.3
done
