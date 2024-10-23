#!/bin/bash

source config.sh

# Define DMAX array
# DMAX_ARRAY=(20 40 60 80)
DMAX_ARRAY=(20)

# Define a function to run the generalized script with specific parameters
run_job() {
    export N=$1
    export STEPS=$2
    export TE=$3
    export G=$4
    export COUPLING=${5:-XX}
    export DMAX=$6

    if [ "$METHOD" = "MPO" ]; then
        export TAU=${7:-0.1}
    fi

    PE_ARRAY=0-10

    COMMON_NAME="$(basename $0)(${PROBLEM},${METHOD},N${N},Dmax${DMAX})"
    sbatch --job-name="${COMMON_NAME}" --export=ALL --array=${PE_ARRAY} JobCooling.sh
}

# Run jobs with different parameters
for N in $(seq 10 10 10); do
    for DMAX in "${DMAX_ARRAY[@]}"; do
        if [ "$METHOD" = "MPO" ]; then
            run_job $N 1000 2.0 0.3 XX $DMAX 0.1
        else
            run_job $N 1000 2.0 0.3 XX $DMAX
        fi
    done
done
