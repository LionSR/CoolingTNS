# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoolingTNS is a Julia-based quantum physics simulation framework for studying cooling protocols in spin systems using tensor network methods and exact diagonalization. It implements various algorithms to simulate the dynamics of quantum systems coupled to thermal baths.

## User's Code Style Preferences

### Architectural Philosophy
- **Pure Dispatch Architecture**: Everything uses Julia's multiple dispatch. No if-else blocks for method selection.
- **Graceful Code Organization**: Clean separation of concerns with modular files, each having a single responsibility.
- **Substance in Dispatch Functions**: No empty wrappers - dispatch functions contain actual implementations.
- **DRY Principles**: Share common elements gracefully without duplication.
- **Type-Based Method Selection**: Let Julia's type system handle all method routing at compile time.

### Dispatch Guidelines
- **No Conditionals**: Replace all if-else logic with type-based dispatch
- **Backend Dispatch**: Use TNBackend/EDBackend types rather than string comparisons
- **Triple/Quadruple Dispatch**: HamiltonianModel × SimulationMethod × EvolutionMethod × Backend
- **Unified Interfaces**: Functions like `find_ground_state` dispatch on backend type rather than having separate `find_ground_state_dmrg` and `find_ground_state_ed`
- **Clean Include Structure**: Modular files that include their dependencies and are included by higher-level dispatchers

### Code Gracefulness Requirements
- **Minimal Complexity**: Each function does one thing well
- **Maximum Readability**: Self-documenting code with clear type signatures
- **Performance-Driven Design**: Use Julia's strengths (type stability, specialization)
- **Comprehensive Type Coverage**: Every combination of types should have a method
- **Predictable Interfaces**: Consistent argument order and naming across dispatch methods

## Common Development Commands

### Running Simulations

```bash
# Basic tensor network simulation
julia Cooling.jl --N 10 --problem niIsing --backend TN --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.1 --te 10.0 --steps 100

# Exact diagonalization for small systems
julia Cooling.jl --N 6 --problem niIsing --backend ED --sim_method density_matrix --evolution_method continuous --coupling XX --g 0.1 --te 5.0 --steps 50

# Trotter evolution with tensor networks
julia Cooling.jl --N 8 --problem Ising --backend TN --sim_method monte_carlo --evolution_method trotter --tau 0.1 --coupling YY --g 0.2 --te 2.0 --steps 20

# With precompiled sysimage (faster startup)
julia --sysimage /u/siruilu/.julia/sysimages/sys_itensors.so Cooling.jl [args]

# Hyperparameter optimization (DEPRECATED - needs refactoring)
julia optCooling.jl --search_method Bayesian --num_trials 20 --N 10 --problem niIsing
```

### Current Parameters
- `--N`: Number of spins in the system
- `--problem`: Problem type (Ising, niIsing, Rydberg)
- `--backend`: Simulation backend (TN, ED)
- `--sim_method`: Simulation method (density_matrix, monte_carlo)
- `--evolution_method`: Evolution method (continuous, trotter)
- `--coupling`: Coupling type (XX, YY, ZZ, XY, XZ, YZ)
- `--g`: Coupling strength
- `--te`: Total evolution time per step
- `--steps`: Number of cooling iterations
- `--tau`: Time step for Trotter evolution
- `--Dmax`: Maximum bond dimension for tensor networks
- `--init_state`: Initial state type (product, identity, theta)
- `--theta`: Parameterized initial state angle (in units of π)
- `--n_trajectories`: Number of trajectories for Monte Carlo method
- `--peInt`: Noise strength (×10⁻³)

### Testing Commands

```bash
# Run quick test on macOS (using gtimeout instead of timeout)
gtimeout 60 julia --startup-file=no -t 1 Cooling.jl --N 4 --problem niIsing --backend TN --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.1 --te 0.5 --steps 5 --Dmax 10

# Run quick test on Linux
timeout 60 julia --startup-file=no -t 1 Cooling.jl --N 4 --problem niIsing --backend TN --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.1 --te 0.5 --steps 5 --Dmax 10
```

### HPC Cluster Submission

```bash
# Submit cooling job to SLURM
sbatch SubmitCooling.sh

# Submit optimization job (DEPRECATED - needs refactoring)
sbatch SubmitOptCooling.sh
```

## High-Level Architecture

### Dispatch-Based Architecture

The codebase uses a clean multiple dispatch architecture:

1. **Backend Types**: `TNBackend` (Tensor Networks) and `EDBackend` (Exact Diagonalization)
2. **Simulation Methods**: `DensityMatrix()` and `MonteCarloWavefunction()`
3. **Evolution Methods**: `ContinuousEvolution()` and `TrotterEvolution()`
4. **Hamiltonian Models**: `IsingModel()`, `NiIsingModel()`, `RydbergModel()`

### Core Simulation Flow

1. **Parameter Setup** (`setup_common_parameters`): Creates typed parameter structures
2. **Problem Initialization** (`setup_problem`): Dispatches on backend and parameters
3. **Initial State** (`setup_initial_state`): Dispatches on backend and state type
4. **Evolution** (`run_cooling`): Dispatches on all type parameters
5. **Analysis**: Computes observables with backend-specific implementations

### Module Structure

**Core Dispatch Files**:
- `src/cooling_interface.jl`: Main interface with substance implementations
- `src/hamiltonian_dispatch.jl`: Includes all Hamiltonian-related dispatchers
- `src/system_hamiltonian_dispatch.jl`: System-only Hamiltonian construction
- `src/system_bath_hamiltonian_dispatch.jl`: System+bath Hamiltonian construction
- `src/ground_state_dispatch.jl`: Unified ground state computation
- `src/setup_system_dispatch.jl`: System setup with backend dispatch
- `src/evolution_dispatch.jl`: Time evolution dispatch
- `src/initial_state_dispatch.jl`: Initial state preparation
- `src/trotter_dispatch.jl`: Trotter circuit construction

**Backend Implementation Files** (contain legacy functions that need dispatch refactoring):
- `src/cooling_functions_ed.jl`: ED-specific implementations (run_cooling_ed_density_matrix, run_cooling_ed_monte_carlo)
- `src/cooling_functions_mps.jl`: MPS implementations (run_cooling_mps)
- `src/cooling_functions_mpo.jl`: MPO implementations (run_cooling_mpo)
- `src/cooling_functions_trotter_mps.jl`: Trotter+MPS implementations (run_cooling_trotter_mps)

**Utilities**:
- `src/parameter_types.jl`: Type definitions for parameters
- `src/coupling_utils.jl`: Coupling operator parsing
- `src/utils.jl`: General utilities and file I/O
- `src/utils_*.jl`: Backend-specific utilities
- `src/plotting.jl`: Visualization
- `src/noise.jl`: Noise models
- `src/policy.jl`: Time-dependent policies
- `src/argparse.jl`: Command-line argument parsing

### Physical Models

All models are implemented with dispatch on `HamiltonianParameters{Model}`:
- **Ising**: Transverse field Ising model H = J∑ZZ + h∑X
- **niIsing**: Non-integrable Ising H = J∑ZZ + hx∑X + hz∑Z
- **Rydberg**: Rydberg atoms with van der Waals interactions

### System-Bath Layout

The framework uses alternating qubit layout: [s₁, b₁, s₂, b₂, ..., sₙ, bₙ]
- System qubits at odd indices: 1, 3, 5, ...
- Bath qubits at even indices: 2, 4, 6, ...

## Development Guidelines

### Current Refactoring Needs

1. **Initial State Dispatch**: Move `setup_initial_state` implementations to their own file (`src/initial_state_setup_dispatch.jl`)
2. **Remove Legacy Functions**: Functions like `setup_problem_mps`, `setup_problem_mpo` are outdated
3. **Optimization Scripts**: `optCooling.jl` and `plotOptCooling.jl` need refactoring to use new dispatch architecture
4. **Backend Implementation Files**: Need to be refactored to pure dispatch instead of monolithic functions

### Adding New Features

1. **New Backend**: Create a new backend type and implement all required dispatch methods
2. **New Model**: Add model type and implement Hamiltonian construction dispatches
3. **New Evolution Method**: Add evolution type and implement in appropriate dispatch files
4. **New Observable**: Add dispatch methods for each backend type

### Code Quality

- **Type Stability**: Ensure all functions are type-stable for performance
- **No Type Piracy**: Only extend functions you own or explicitly import
- **Consistent Interfaces**: Maintain argument order across dispatch methods
- **Documentation**: Each dispatch method should have a docstring
- **Testing**: Add tests for each new dispatch combination

### Performance Considerations

- ITensors.jl for tensor network operations
- Yao.jl for quantum circuit/state manipulation in ED
- KrylovKit.jl for eigenvalue problems
- ExponentialUtilities.jl for matrix exponentials
- MKL on Linux for optimized BLAS/LAPACK

## Testing

```bash
# Run all tests
julia --project=. test/runtests.jl

# Run specific test file
julia --project=. test/test_cooling_interface.jl
```

Test files verify:
- Consistency across backends
- Correct dispatch resolution
- Type stability
- Edge cases and error handling

## Common Patterns

### Backend-Agnostic Code
```julia
# Let dispatch handle backend differences
function compute_observable(state::QuantumState{B}, obs) where B<:CoolingBackend
    # Dispatches to appropriate implementation
    return measure(state, obs, B())
end
```

### Adding Dispatch Methods
```julia
# System Hamiltonian
function construct_system_hamiltonian(
    ham_params::HamiltonianParameters{YourModel}, 
    backend::TNBackend, 
    sites
)
    # Implementation for your model on TN backend
end

# Ground State
function find_ground_state(H_sys, backend::YourBackend, args...)
    # Implementation for your backend
end
```

### Type Hierarchies
```julia
abstract type HamiltonianModel end
struct IsingModel <: HamiltonianModel end
struct NiIsingModel <: HamiltonianModel end

abstract type CoolingBackend end
struct TNBackend <: CoolingBackend end
struct EDBackend <: CoolingBackend end
```

## Known Issues and TODOs

- **optCooling.jl**: Still uses old string-based method selection, needs dispatch refactoring
- **plotOptCooling.jl**: Needs update to work with new parameter types
- **ED N extraction**: Currently extracting N from Hamiltonian size is hacky (see TODO in cooling_interface.jl:142)
- **Initial state setup**: Should be moved to separate dispatch file for consistency
- **Legacy wrapper removal**: Many files still have Dict-based wrappers for backward compatibility

## Platform-Specific Notes

### macOS
- Use `gtimeout` instead of `timeout` for command timeouts
- Install with: `brew install coreutils`

### Linux
- Use standard `timeout` command
- MKL loaded automatically for better performance

## Memory Notes

- Removed all references to old `--method` argument (replaced by `--backend`)
- Eliminated `--ed_method` (now `--sim_method` works for all backends)
- No more backend-specific functions like `find_ground_state_dmrg`
- Pure dispatch architecture throughout - no string comparisons
- All empty wrappers removed - substance in dispatch functions
- Legacy functions like `setup_problem_mps` are deprecated