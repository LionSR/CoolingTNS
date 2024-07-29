#!/bin/bash

# Common configuration
export JULIA_DEPOT_PATH=/ptmp/mpq/srlu/Julia
export PROBLEM=${PROBLEM:-niIsing}
export METHOD=${METHOD:-MPS}
export DMAX=${DMAX:-40}
export SEARCH_METHOD=${SEARCH_METHOD:-Bayesian}
export NUM_TRIALS=${NUM_TRIALS:-20}
export OPENBLAS_NUM_THREADS=1
export JULIA_NUM_THREADS=1

# Function to create output file name
create_outfile() {
    local prefix=$1
    local suffix=$2
    echo "${prefix}/${ID}_${METHOD}_${suffix}${PROBLEM}Ns${N}Nb${N}_Dmax${DMAX}_${additional_params}_peInt${PE}"
}

# Load modules
module_load() {
    module purge
    module load anaconda/3/2023.03
    module load julia/1.10
    module load mkl/2023.1
}

# Print node info
print_node_info() {
    echo "I ran on:"
    cd $SLURM_SUBMIT_DIR
    echo $SLURM_NODELIST
}

# Set Julia alias
alias julia="/tqo/u/system/soft/SLE_15/packages/x86_64/julia/1.10.4/bin/julia"
alias julia_itensors="julia --sysimage /u/siruilu/.julia/sysimages/sys_itensors.so "
