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

# Hamiltonian parameters
HAM_PARAMS="--problem=$PROBLEM --N=$N"
if [ "$PROBLEM" = "Ising" ]; then
    HAM_PARAMS="$HAM_PARAMS --J=$J --h=$H"
elif [ "$PROBLEM" = "niIsing" ]; then
    HAM_PARAMS="$HAM_PARAMS --J=$J --hx=$HX --hz=$HZ"
fi

# Simulation parameters
SIM_PARAMS="--method=$METHOD --steps=$STEPS --peInt=$PE"
if [ "$METHOD" = "MPO" ]; then
    SIM_PARAMS="$SIM_PARAMS --tau=$TAU"
else
    SIM_PARAMS="$SIM_PARAMS --Dmax=$DMAX"
fi

# Coupling parameters
COUPLING_PARAMS="--te=$TE --g=$G --coupling=$COUPLING"

# Run Julia script with parameters
srun --export=ALL --output="${SLURM_JOB_OUTPUT}" --error="${SLURM_JOB_ERROR}" \
    julia --sysimage /u/siruilu/.julia/sysimages/sys_itensors.so \
    Cooling${METHOD}.jl $HAM_PARAMS $SIM_PARAMS $COUPLING_PARAMS

wait
