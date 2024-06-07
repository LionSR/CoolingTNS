#!/bin/bash

# Set the number of threads for OpenBLAS and Julia
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS=1

# Take input N
if [ -z "$1" ]; then
    echo "Usage: $0 <N>"
    exit 1
fi

N=$1

# Loop over peInt from 0 to 10 and run each in the background
for peInt in $(seq 0 10)
do
    echo "Starting optimization for N=$N, peInt=$peInt"
    julia OptimizeCoolingMPS.jl --N=$N --search_method=Bayesian --num_trials=20 --steps=200 --cutoff=1e-5 --Dmax=40 --peInt=$peInt &
done

# Wait for all background jobs to finish
wait

echo "All optimizations completed."