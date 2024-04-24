#!/bin/bash

#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000 # Memory
#SBATCH --cpus-per-task=2
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH -t 0-100:00 # Runtime in D-HH:MM format
#SBATCH --array=0-10

export TASK_ID=$SLURM_ARRAY_TASK_ID
export JOB_ID=$SLURM_ARRAY_JOB_ID
export JID=$SLURM_JOB_ID
export ID=${JOB_ID:-$JID}
export N=$N
export PE=$TASK_ID
export OUTFILE="Log/${ID}_NoisyCooling${PROBLEM}Ns${N}Nb${N}_g${G_VALUE}te${TE_VALUE}_peInt${PE}"
export SLURM_JOB_OUTPUT="${OUTFILE}.out"
export SLURM_JOB_ERROR="${OUTFILE}.err"

# Load modules
module purge
module load anaconda/3/2023.03
module load julia/1.10
module load mkl/2023.1

# Print node info
echo "I ran on:"
cd $SLURM_SUBMIT_DIR
echo $SLURM_NODELIST

alias julia="/tqo/u/system/soft/SLE_15/packages/x86_64/julia/1.10.1/bin/julia"
alias julia_itensors="julia --sysimage /u/siruilu/.julia/sysimages/sys_itensors.so "

# Run Julia script with parameters
srun --export=ALL --output="${SLURM_JOB_OUTPUT}" --error="${SLURM_JOB_ERROR}" julia --sysimage /u/siruilu/.julia/sysimages/sys_itensors.so CoolingMPO.jl --method=$METHOD --problem=$PROBLEM --N=$N --steps=${STEPS_VALUE} --te=${TE_VALUE} --g=${G_VALUE} --peInt=$PE

wait
