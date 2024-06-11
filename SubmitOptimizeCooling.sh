#!/bin/bash

export PROBLEM=niIsing
export METHOD=MPS
export DMAX=60
export SEARCH_METHOD=Bayesian
export NUM_TRIALS=20
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS=1
COMMON_NAME="$(basename $0)(${PROBLEM},${METHOD},${SEARCH_METHOD})"

# Define a function to run the generalized script with specific parameters
run_job() {
    export N=$1  # Note this change; N is now passed as a parameter
    export STEPS_VALUE=$2
    sbatch --job-name="${COMMON_NAME}" --export "N=$N,PROBLEM=$PROBLEM,METHOD=$METHOD,STEPS_VALUE=$2,DMAX=$DMAX,SEARCH_METHOD=$SEARCH_METHOD,NUM_TRIALS=$NUM_TRIALS" JobOptimizeCooling.sh
}

# Run jobs with different parameters

# run_job 10 200

for N in $(seq 10 10 100); do
    run_job $N 200
done

