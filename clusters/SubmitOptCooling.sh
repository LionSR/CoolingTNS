#!/bin/bash

source config.sh

COMMON_NAME="$(basename $0)(${PROBLEM},${METHOD},${SEARCH_METHOD},N${N},Dmax${DMAX})"

# Define DMAX array
DMAX_ARRAY=(20 40 60 80)

# Define a function to run the generalized script with specific parameters
run_job() {
    export N=$1
    export STEPS=$2
    export COUPLING=${3:-XX}
    export DMAX=$4

    if [ "$METHOD" = "MPO" ]; then
        export TAU=${5:-0.1}
    fi

    PE_ARRAY=0-10

    COMMON_NAME="$(basename $0)(${PROBLEM},${METHOD},N${N},Dmax${DMAX})"
    sbatch --job-name="${COMMON_NAME}" --export=ALL --array=${PE_ARRAY} JobOptCooling.sh
}

# Run jobs with different parameters
for N in $(seq 10 10 100); do
    for DMAX in "${DMAX_ARRAY[@]}"; do
        if [ "$METHOD" = "MPO" ]; then
            run_job $N 200 XX $DMAX 0.1
        else
            run_job $N 200 XX $DMAX
        fi
    done
done

