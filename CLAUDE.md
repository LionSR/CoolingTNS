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

# Tensor network

## MPS with monte carlo and continuous evolution
julia Cooling.jl --N 10 --problem Ising --backend TN --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.1 --te 10.0 --steps 20

## Trotter evolution with MPS
julia Cooling.jl --N 10 --problem Ising --backend TN --sim_method monte_carlo --evolution_method trotter --tau 0.1 --coupling XX --g 0.2 --te 2.0 --steps 20

## Tensor network with MPO
julia Cooling.jl --N 10 --problem Ising --backend TN --sim_method density_matrix --evolution_method trotter --tau 0.1 --coupling XX --g 0.2 --te 2.0 --steps 20


# Exact diagonalization

## continuous evolution of density matrix
julia Cooling.jl --N 4 --problem Ising --backend ED --sim_method density_matrix --evolution_method continuous --coupling XX --g 0.1 --te 0.5 --steps 5

## continuous evolution of monte carlo wavefunction
julia Cooling.jl --N 4 --problem Ising --backend ED --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.1 --te 0.5 --steps 5

## trotter evolution of density matrix
julia Cooling.jl --N 4 --problem Ising --backend ED --sim_method density_matrix --evolution_method trotter --tau 0.1 --coupling XX --g 0.1 --te 0.5 --steps 5

## ED with periodic BC and k-space measurements (only for Ising model)
julia Cooling.jl --N 6 --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling XX --g 0.3 --te 2.0 --steps 20 --J 1.0 --h 2.0

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
- `--theta`: Dimensionless theta-code parameter for the initial state
  (-0.5 -> |0>, 0 -> |+>, 0.5 -> |1>)
- `--n_trajectories`: Number of trajectories for Monte Carlo method
- `--peInt`: Noise strength (×10⁻³)
- `--bc`: Boundary conditions (open, periodic, antiperiodic). ED and TN
  continuous Ising-family MPO construction honor them. TN Trotter currently
  supports open boundaries only and rejects non-open boundaries. Rydberg
  dynamics should use open boundaries until a non-open Rydberg convention is
  specified.
- `--J`: Ising coupling strength (default 1.0)
- `--h`: Transverse field strength (default 1.0)

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

### Module Structure (Unified Dispatch Architecture)

**Core Unified Files**:
- `src/cooling_evolution.jl`: Main cooling evolution with unified TN+ED dispatch
- `src/system_hamiltonian.jl`: System Hamiltonian construction (TN+ED unified)
- `src/system_bath_hamiltonian.jl`: System+bath Hamiltonian construction (TN+ED unified) 
- `src/ground_state.jl`: Unified ground state computation (TN+ED)
- `src/initial_state.jl`: Initial state preparation (TN+ED unified)
- `src/setup.jl`: Problem setup with backend dispatch
- `src/ed_backend.jl`: Clean Float64-only ED backend (no Yao dependencies)

**Support Files**:
- `src/parameter_types.jl`: Type definitions for parameters
- `src/cooling_types.jl`: CoolingProblem and QuantumState types
- `src/coupling_utils.jl`: Coupling operator parsing
- `src/utils.jl`: General utilities and file I/O
- `src/utils_mps.jl` / `src/utils_mpo.jl`: TN-specific utilities
- `src/plotting.jl`: Visualization
- `src/noise.jl`: Noise models
- `src/policy.jl`: Time-dependent policies
- `src/argparse.jl`: Command-line argument parsing
- `src/state_manipulation.jl`: Dispatched state operations
- `src/bath_measurements.jl`: Bath measurement functions
- `src/trotter.jl`: Trotter evolution support
- `src/evolution.jl`: Evolution utilities
- `src/setup_system.jl`: System setup utilities

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

### Data Flow

1. Results are saved as HDF5 files with backend type in filename (SimTN or SimED)
2. NO method names (MPS/MPO/TrotterMPS) in filenames anymore
3. MATLAB reference implementations in `ExactDiagonalization/` validate tensor network results
4. Plotting scripts (`plotCooling.jl`, `plotOptCooling.jl`) generate publication-quality figures

### File Naming Convention

Files are named: `Cooling_Ham{model}_Coupling{type}_Sim{backend}Dmax{D}`
- Backend: SimTN or SimED (NOT SimMPS/SimMPO/SimTrotterMPS)
- No method information in filenames

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

- ITensors.jl and ITensorMPS.jl for tensor network operations
- KrylovKit.jl for eigenvalue problems and sparse matrix operations
- LinearAlgebra.jl and SparseArrays.jl for ED backend matrix operations
- Complex matrices for quantum states, real sparse matrices for operators
- Cached evolution operators in ED backend to avoid repeated diagonalization
- MKL on Linux for optimized BLAS/LAPACK

### K-Space Measurements

For ED simulations with periodic/antiperiodic boundary conditions:
- Automatically computes momentum distribution n_k using Jordan-Wigner transformation
- Generates plots showing n_k and e_k = ε_k × n_k evolution during cooling
- Only enabled for Ising model (integrable system)
- Ground state and its n_k should be computed numerically for accurate comparison

### Getting doucmentations from Julia packages:
Use ITensors.jl or ITensorMPS.jl, you can get the documentation of a function by running:
```bash
# Get function documentation from a package
julia -e 'using PackageName; @doc function_name'
```


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

- **Monte Carlo trajectories**: Cooling is stochastic; energy need not decrease trajectory-by-trajectory (TN and ED). Validate using ensemble averages (see `scripts/diagnostics/physics_investigation_report.jl`).
- **optCooling.jl**: Still uses old string-based method selection, needs dispatch refactoring  
- **TN Backend Measurements**: Missing measurement functions for some TN method combinations
- **Precompilation**: Long precompilation times due to ITensors/Yao dependencies eating tokens during debugging

## Platform-Specific Notes

### macOS
- Use `gtimeout` instead of `timeout` for command timeouts
- Install with: `brew install coreutils`

### Linux
- Use standard `timeout` command
- MKL loaded automatically for better performance

## Implementation Status

### ✅ Completed Features

**Architecture Overhaul:**
- **Pure Dispatch Architecture**: Completely implemented using Julia's multiple dispatch 
- **Unified File Structure**: Eliminated all duplicate `*_ed.jl` files - everything now in single unified files
- **Clean ED Backend**: Complex matrix support for proper quantum mechanics
- **Type-Based Routing**: All method selection uses types, no string comparisons

**Backend Implementations:**
- **TNBackend**: Full tensor network support with ITensors.jl
- **EDBackend**: Clean exact diagonalization using LinearAlgebra + SparseArrays + KrylovKit
- **Unified Interfaces**: Same dispatch signatures work for both backends
- **Multiple Method Support**: DensityMatrix + MonteCarloWavefunction × ContinuousEvolution + TrotterEvolution

**New Features:**
- **Boundary Conditions**: Support for periodic and anti-periodic BC in ED backend
- **K-Space Measurements**: Momentum distribution measurements for PBC/APBC using Jordan-Wigner transformation
- **Evolution Caching**: Cached evolution operators for ED backend performance
- **Complex Jordan-Wigner**: Proper complex fermionic operators with real Pauli matrices

**File Organization:**
- All legacy duplicate files removed (`system_hamiltonian_ed.jl`, `cooling_evolution_ed.jl`, etc.)
- Single files with unified TN+ED dispatch: `system_hamiltonian.jl`, `cooling_evolution.jl`, `ground_state.jl`, `initial_state.jl`
- Clean module structure with no Yao dependencies in ED backend

### ⚠️ Known Issues

**Physics Problems:**
- **ED Cooling Rate**: Cooling is very slow with current parameters - may need stronger coupling or longer evolution times
- **TN Measurements**: Some TN backend measurement combinations missing

**Performance:**
- **Precompilation Time**: Long compilation due to ITensors dependencies
- **ED Scaling**: ED backend limited to small systems (N ≤ 12) due to exponential scaling

### 🔧 Development Guidelines

**For ED Backend Usage:**
- Use small systems (N ≤ 10) for reasonable performance
- Density matrix method more reliable than Monte Carlo for ED
- Enable periodic/antiperiodic BC for k-space measurements
- Use cached evolution operators for better performance

**Jordan-Wigner Convention:**
- |↑⟩ = vacuum (no fermion), |↓⟩ = occupied (one fermion)
- a = (X - Y_real)/2 = σ^+ (annihilation)
- a† = (X + Y_real)/2 = σ^- (creation)
- Real Pauli Y matrix: Y_real = [0 -1; 1 0]

**Architecture Maintenance:**
- Keep unified dispatch pattern - no new duplicate files
- All new features use multiple dispatch on backend types
- Complex matrices for quantum states, real matrices for operators when possible
- Follow established type hierarchy patterns


### Rules about *.tex LaTeX notes

1. **Equation References**: Always use `\ref{eq:label}` or `\cref{eq:label}` instead of hardcoded equation numbers:
   ```latex
   % Bad: According to equation (82), the constant term is...
   % Good: According to Eq.~\ref{eq:transformed_hamiltonian}, the constant term is...
   % Also good: According to \cref{eq:transformed_hamiltonian}, the constant term is...
   ```

2. **Citing Specific Results**: When referring to equations from the notes in code comments:
   ```julia
   # From Eq. \ref{eq:mode_energy} in MapToSpin.tex
   ε_k = sqrt(1 + sin(2θ) * cos(2π*k/N))
   
   # NOT: From equation (260) in the notes
   ```

3. **Label Conventions**: Use descriptive labels that won't change if equations are reordered:
   - `\label{eq:spin_hamiltonian}` for the spin Hamiltonian
   - `\label{eq:JW_transformation}` for Jordan-Wigner transformation
   - `\label{eq:mode_energy}` for mode energies
   
4. **Cross-referencing**: When implementing formulas from the notes, always include the LaTeX label:
   ```julia
   # Computing ground state energy from Eq. \ref{eq:gs_energy}
   # where the APBC sum runs over half-integer k values
   ```

### Investigation Philosophy

When investigating physics problems:

1. **Start Simple**: Begin with minimal implementations to verify core concepts before adding complexity
2. **Test Systematically**: When discrepancies arise, test across multiple parameter values (different N, θ, etc.) to identify patterns
3. **Validate Known Limits**: Check special cases where analytical results are known (e.g., θ=π/2 for pure transverse field)
4. **Compare Observable by Observable**: Verify each physical quantity separately (e.g., gaps vs absolute energies)

### Debugging Best Practices

1. **No Hardcoded Conclusions**: Never use statements like `println("These match!")`. Instead:
   ```julia
   if abs(value1 - value2) < tolerance
       println("✓ Values match within tolerance")
   else
       println("✗ Mismatch: $(abs(value1 - value2))")
   end
   ```

2. **Avoid Magic Numbers**: Don't hardcode numerical values from previous runs:
   ```julia
   # Bad: println("Discrepancy: 0.369")
   # Good: println("Discrepancy: $discrepancy")
   ```

3. **Use Descriptive Variables**: Create meaningful variable names for comparisons:
   ```julia
   # Instead of: println("Ratio: $(discrepancy/ε_π)")
   ratio_to_special_mode = discrepancy/ε_π
   println("Ratio to π-mode energy: $ratio_to_special_mode")
   ```

4. **Systematic Output**: Structure output to be machine-readable when scanning parameters:
   ```julia
   @printf("%.3f\t%.6f\t%.6f\n", param, result1, result2)
   ```

### Physics Validation Approach

When comparing analytical and numerical results:

1. **Check Symmetries First**: Verify conserved quantities (e.g., parity sectors)
2. **Test Gap Structure**: Energy differences are often more robust than absolute energies
3. **Scan Parameter Space**: Look for patterns across different values of N, θ, coupling strengths
4. **Identify Special Limits**: Find parameter values where the problem simplifies (e.g., θ=π/2 removes interactions)
