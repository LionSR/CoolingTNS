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
    local ham_name_part="Ham${PROBLEM}Ns${N}Nb${N}"
    local coupling_name_part="Coupling${COUPLING}g${G}te${TE}steps${STEPS}"
    local sim_name_part="Sim${METHOD}Dmax${DMAX}tau${TAU}"
    
    if [ "$PE" -gt 0 ]; then
        sim_name_part="${sim_name_part}peInt${PE}"
    fi
    
    echo "${prefix}/${ID}_${suffix}_${ham_name_part}_${coupling_name_part}_${sim_name_part}"
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

