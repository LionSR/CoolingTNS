# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoolingTNS is a Julia-based quantum physics simulation framework for studying cooling protocols in spin systems. It implements various algorithms to simulate the dynamics of quantum systems coupled to thermal baths, supporting both tensor network methods and exact diagonalization.

## Common Development Commands

### Running Simulations

```bash
# Basic cooling simulation with tensor networks
julia Cooling.jl --N 10 --problem niIsing --method MPS --coupling XX --g 0.1 --te 10.0 --steps 100

# Exact diagonalization (small systems)
julia Cooling.jl --N 6 --problem niIsing --method ED --ed_method density_matrix --coupling XX --g 0.1 --te 5.0 --steps 50

# With precompiled sysimage (faster startup)
julia --sysimage /u/siruilu/.julia/sysimages/sys_itensors.so Cooling.jl [args]

# Hyperparameter optimization
julia optCooling.jl --search_method Bayesian --num_trials 20 --N 10 --problem niIsing
```

### Common Parameters
- `--N`: Number of spins in the system
- `--problem`: Problem type (Ising, niIsing, Rydberg)
- `--method`: Simulation method (MPS, MPO, TrotterMPS, ED)
- `--coupling`: Coupling type (XX, YY, ZZ, XY, XZ, YZ)
- `--g`: Coupling strength
- `--te`: Total evolution time
- `--steps`: Number of time steps
- `--Dmax`: Maximum bond dimension for tensor network methods
- `--ed_method`: ED simulation type (density_matrix, monte_carlo)
- `--init_state`: Initial state type (product, identity, theta)
- `--theta`: Parameterized initial state angle (in units of π)

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
2. **Problem Initialization**: Creates quantum states, Hamiltonians, and evolution operators based on the chosen backend
3. **Evolution**: Performs time evolution using the selected simulation method
4. **Analysis**: Computes observables like fidelity, energy, and mutual information

### Key Architectural Components

**Simulation Backends**:
- **ED (Exact Diagonalization)**: Full quantum dynamics for small systems (N ≤ 10)
- **MPS (Matrix Product States)**: Efficient representation with controlled entanglement
- **MPO (Matrix Product Operators)**: Density matrix evolution with tensor networks
- **TrotterMPS**: Trotter decomposition combined with MPS

**Simulation Methods**:
- **Density Matrix**: Full quantum state tracking (MPO, ED with density matrix)
- **Monte Carlo Wavefunction**: Stochastic trajectories (MPS, TrotterMPS, ED with MC)

**Physical Models**:
- **Ising**: Standard transverse field Ising model
- **niIsing**: Non-integrable Ising model with additional terms
- **Rydberg**: Rydberg atom arrays with dressed interactions

**System-Bath Coupling**:
The framework supports various coupling operators between system and bath spins:
- Diagonal couplings: XX, YY, ZZ
- Off-diagonal couplings: XY, XZ, YZ

**Optimization Framework**:
Uses Hyperopt.jl for Bayesian optimization of coupling parameters to maximize cooling efficiency.

### Module Structure

- `src/ham.jl`: Hamiltonian construction for different models
- `src/cooling_functions_*.jl`: Core evolution algorithms for each backend
- `src/cooling_functions_ed.jl`: Exact diagonalization with Yao.jl
- `src/utils_*.jl`: Utility functions for tensor operations
- `src/policy.jl`: Time-dependent coupling policies
- `src/noise.jl`: Noise models for realistic simulations
- `src/dmrg.jl`: DMRG ground state calculations
- `src/plotting.jl`: Visualization utilities

### Data Flow

1. Results are saved as HDF5 files with structured metadata
2. Common interface allows easy comparison between different backends
3. Plotting scripts (`plotCooling.jl`, `plotOptCooling.jl`) generate publication-quality figures

## Development Notes

- The project uses ITensors.jl for tensor network operations
- MKL is loaded on Linux for optimized linear algebra
- Thread safety: Set `JULIA_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` for cluster runs
- Results directory structure: `{ID}_{type}_{Ham}_{Coupling}_{Sim}`

## Simulation Backends

### Exact Diagonalization (ED)
- Uses Yao.jl for quantum state manipulation
- ExponentialUtilities.jl for efficient time evolution
- KrylovKit.jl for ground state calculations
- Supports both density matrix and Monte Carlo wavefunction methods
- Limited to small systems (N ≤ 10) due to exponential scaling

### Tensor Network Methods
- **MPS**: Matrix Product States with TDVP time evolution
- **MPO**: Matrix Product Operators with TEBD evolution
- **TrotterMPS**: Trotter gates with MPS compression
- Can handle larger systems with controlled approximation

## Initial State Options

The framework supports multiple initial state types:

```bash
# Product state (default)
julia Cooling.jl --init_state product ...

# Identity/maximally mixed state
julia Cooling.jl --init_state identity --method MPO ...

# Theta-parameterized states
julia Cooling.jl --init_state theta --theta -0.5  # All down |111...⟩
julia Cooling.jl --init_state theta --theta 0.0   # X+ state |+++...⟩
julia Cooling.jl --init_state theta --theta 0.5   # All up |000...⟩
```

## Testing

Run tests with:
```bash
julia --project=. tests/runtests.jl
```

Individual test files:
- `tests/test_cooling_backends.jl`: Compare different simulation backends
- `tests/test_initial_states.jl`: Test initial state preparation
- `tests/test_hamiltonians.jl`: Verify Hamiltonian construction
- `tests/test_observables.jl`: Check observable calculations

# Important Instructions
- Focus on Julia implementation best practices
- Use multiple dispatch for clean interfaces
- Leverage Julia's type system for performance
- Keep the codebase DRY with proper abstractions
- Test edge cases and ensure consistency across backends