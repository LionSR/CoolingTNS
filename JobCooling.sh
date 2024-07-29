#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000
#SBATCH --cpus-per-task=2
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

OUTFILE=$(create_outfile "Log" "Cooling")
export SLURM_JOB_OUTPUT="${OUTFILE}.out"
export SLURM_JOB_ERROR="${OUTFILE}.err"

module_load
print_node_info

# Common parameters
COMMON_PARAMS="--method=$METHOD --problem=$PROBLEM --N=$N --steps=$STEPS --te=$TE --g=$G --peInt=$PE --coupling=$COUPLING"

# Method-specific parameters
if [ "$METHOD" = "MPO" ]; then
    METHOD_PARAMS="--tau=$TAU"
else
    METHOD_PARAMS="--Dmax=$DMAX"
fi

# Run Julia script with parameters
srun --export=ALL --output="${SLURM_JOB_OUTPUT}" --error="${SLURM_JOB_ERROR}" \
    julia --sysimage /u/siruilu/.julia/sysimages/sys_itensors.so \
    Cooling${METHOD}.jl $COMMON_PARAMS $METHOD_PARAMS

wait
