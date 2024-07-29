#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -t 0-100:00
#SBATCH --array=0-10

source config.sh

export TASK_ID=$SLURM_ARRAY_TASK_ID
export JOB_ID=$SLURM_ARRAY_JOB_ID
export JID=$SLURM_JOB_ID
export ID=${JOB_ID:-$JID}
export PE=$TASK_ID

OUTFILE=$(create_outfile "LogOptimize" "OptimizeCooling")
export SLURM_JOB_OUTPUT="${OUTFILE}.out"
export SLURM_JOB_ERROR="${OUTFILE}.err"

module_load
print_node_info

# Run Julia script with parameters
srun --export=ALL --output="${SLURM_JOB_OUTPUT}" --error="${SLURM_JOB_ERROR}" julia --sysimage /u/siruilu/.julia/sysimages/sys_itensors.so optimizeCoolingMPS.jl --method=$METHOD --problem=$PROBLEM --N=$N --steps=${STEPS_VALUE} --Dmax=${DMAX} --search_method=${SEARCH_METHOD} --num_trials=${NUM_TRIALS} --peInt=$PE

wait
