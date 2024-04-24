#!/bin/bash

export PROBLEM=niIsing
export METHOD=MPO
COMMON_NAME="$(basename $0)(${PROBLEM})"

# Define a function to run the generalized script with specific parameters
run_job() {
    export N=$1  # Note this change; N is now passed as a parameter
    export STEPS_VALUE=$2
    export TE_VALUE=$3
    export G_VALUE=$4
    sbatch --job-name="${COMMON_NAME}" --export "N=$N,PROBLEM=$PROBLEM,METHOD=$METHOD,STEPS_VALUE=$2,TE_VALUE=$3,G_VALUE=$4" JobCoolingMPO.sh
}

# Run jobs with different parameters
for N in $(seq 10 10 100); do
    run_job $N 500 2.0 0.3
done
