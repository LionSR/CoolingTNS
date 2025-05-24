# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoolingTNS is a Julia-based quantum physics simulation framework for studying cooling protocols in spin systems using tensor network methods. It implements various algorithms to simulate the dynamics of quantum systems coupled to thermal baths.

## Common Development Commands

### Running Simulations

```bash
# Basic cooling simulation
julia Cooling.jl --N 10 --problem niIsing --method MPS --coupling XX --g 0.1 --te 10.0 --steps 100

# With precompiled sysimage (faster startup)
julia --sysimage /u/siruilu/.julia/sysimages/sys_itensors.so Cooling.jl [args]

# Hyperparameter optimization
julia optCooling.jl --search_method Bayesian --num_trials 20 --N 10 --problem niIsing
```

### Common Parameters
- `--N`: Number of spins in the system
- `--problem`: Problem type (Ising, niIsing, Rydberg)
- `--method`: Simulation method (MPS, MPO, TrotterMPS)
- `--coupling`: Coupling type (XX, YY, ZZ, XY, XZ, YZ)
- `--g`: Coupling strength
- `--te`: Total evolution time
- `--steps`: Number of time steps
- `--Dmax`: Maximum bond dimension for MPS

### HPC Cluster Submission

```bash
# Submit cooling job to SLURM
sbatch SubmitCooling.sh

# Submit optimization job
sbatch SubmitOptCooling.sh
```

## High-Level Architecture

### Core Simulation Flow

1. **Parameter Setup** (`setup_common_parameters`): Parses command-line arguments and creates parameter structures
2. **Problem Initialization** (`setup_problem_*`): Creates quantum states, Hamiltonians, and evolution operators based on the chosen method
3. **Evolution** (`run_cooling_*`): Performs time evolution using the selected tensor network method
4. **Analysis**: Computes observables like fidelity, energy, and mutual information

### Key Architectural Components

**Tensor Network Methods**:
- **MPS (Matrix Product States)**: Efficient representation of quantum states with controlled entanglement
- **MPO (Matrix Product Operators)**: Operator representation for time evolution
- **TrotterMPS**: Trotter decomposition combined with MPS for time evolution

**Physical Models**:
- **Ising**: Standard transverse field Ising model
- **niIsing**: Non-integrable Ising model with additional terms
- **Rydberg**: Rydberg atom arrays with dressed interactions

**System-Bath Coupling**:
The framework supports various coupling operators between system and bath spins:
- Diagonal couplings: XX, YY, ZZ
- Off-diagonal couplings: XY, XZ, YZ

**Optimization Framework**:
Uses Hyperopt.jl for Bayesian optimization of coupling parameters to maximize cooling efficiency (final ground state fidelity).

### Module Structure

- `src/ham.jl`: Hamiltonian construction for different models
- `src/cooling_functions_*.jl`: Core evolution algorithms for each method
- `src/utils_*.jl`: Utility functions for tensor operations
- `src/policy.jl`: Time-dependent coupling policies
- `src/noise.jl`: Noise models for realistic simulations
- `src/dmrg.jl`: DMRG ground state calculations
- `src/plotting.jl`: Visualization utilities

### Data Flow

1. Results are saved as HDF5 files with structured metadata
2. MATLAB reference implementations in `ExactDiagonalization/` validate tensor network results
3. Plotting scripts (`plotCooling.jl`, `plotOptCooling.jl`) generate publication-quality figures

## Development Notes

- The project uses ITensors.jl for tensor network operations
- MKL is loaded on Linux for optimized linear algebra
- Thread safety: Set `JULIA_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` for cluster runs
- Results directory structure: `{ID}_{type}_{Ham}_{Coupling}_{Sim}`

## Exact Diagonalization Reference Implementation

The `ExactDiagonalization/` directory contains MATLAB reference implementations for validating tensor network results:

### Key Scripts
- `MatlabRho/CoolingMultiBath.m`: Main script (now uses Julia naming conventions)
- `MatlabRho/EvolveMultiBath.m`: Time evolution using matrix exponentiation
- `MatlabRho/plotCoolingMultiBath.m`: Plotting script for visualization
- `MatlabRho/test_resonant_cooling.m`: Test script demonstrating usage
- `MatlabRho/test_julia_consistency.m`: Shows MATLAB-Julia parameter correspondence

### Running MATLAB Simulations
```matlab
% MATLAB now uses Julia naming conventions
N = 5;                  % System size
J = 1.0; hx = -1.05; hz = 0.5;  % Hamiltonian parameters
coupling = "XX";        % Coupling type (was coupling_types)
steps = 1000;           % Number of cooling iterations (was Niter)
g = 0.1;                % Coupling strength
te = 5.0;               % Total evolution time per step (was t)

% Run simulation
CoolingMultiBath;
```

### Key Implementation Details
- **Resonant Cooling**: Default sets `delta = -gap` where gap is the system energy gap
- **Bath Initialization**: Bath spins initialized in ground state (|1⟩ for Δ<0, |0⟩ for Δ>0)
- **Time Evolution**: Uses matrix exponentiation with optimized algorithms (expmv/expokit)
- **Observables**: Tracks energy, ground state overlap, purity, and bath magnetization
- **Multi-State Analysis**: Tests multiple initial states (θ = -0.5π, 0π, 0.5π)

### Results Format
- Saved as `.mat` files with naming: `CI_MB_N{N}J{J}hx{hx}hz{hz}_{coupling}{steps}delta{delta}g{g}te{te}.mat`
- Includes energy evolution, ground state overlap, purity, population dynamics
- Automatically generates PDF plots for visualization

## Consistency Features

### Initial State Options
The Julia implementation now supports multiple initial state types to match MATLAB capabilities:

```bash
# Product state (default)
julia Cooling.jl --init_state product ...

# Identity/maximally mixed state (like MATLAB default for MPO)
julia Cooling.jl --init_state identity --method MPO ...

# Theta-parameterized states
julia Cooling.jl --init_state theta --theta -0.5  # All down
julia Cooling.jl --init_state theta --theta 0.0   # X+ state
julia Cooling.jl --init_state theta --theta 0.5   # All up
```

### Parameter Name Consistency
Both MATLAB and Julia now use the same parameter names:
- `delta` - bath detuning (MATLAB previously used `Delta`)
- `te` - total evolution time per step (MATLAB previously used `t`)
- `steps` - number of cooling iterations (MATLAB previously used `Niter`)
- `coupling` - coupling type (MATLAB previously used `coupling_types`)
- `g`, `N`, `J`, `hx`, `hz` - unchanged