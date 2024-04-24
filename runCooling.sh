#!/bin/zsh

# Set the number of threads for OpenBLAS and Julia
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS=1

# Loop over N from 10 to 100 and run each in the background
for N in $(seq 10 10 100)
do
    echo "Starting optimization for N=$N"
    # julia runCoolingMPS.jl --N=$N --steps=50 &
    # julia runCoolingMPS.jl --N=$N --steps=50 --pe=0.001 &
    julia runCoolingMPO.jl --N=$N --steps=50 --cutoff=1e-4 --Dmax=20 &
done

# Wait for all background jobs to finish
wait

echo "All optimizations completed."