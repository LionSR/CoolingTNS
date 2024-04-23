#!/bin/bash

# Set the number of threads for OpenBLAS and Julia
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS=1

# Loop over N from 10 to 100 and run each in the background
for N in $(seq 10 10 100)
do
    echo "Starting optimization for N=$N"
    julia runCoolingMPS.jl --N=$N --search_method=Bayesian --num_trials=20 --steps=200 &
done

# Wait for all background jobs to finish
wait

echo "All optimizations completed."