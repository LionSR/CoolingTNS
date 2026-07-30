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
# Only `--search_method Random` is implemented; other values fall back with a warning.
julia optCooling.jl --search_method Random --num_trials 20 --N 10 --problem niIsing
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
- `--init-state`: Initial state type (product, identity, theta, ground).
  The parsed metadata key remains `init_state`; `--init_state` is accepted as a
  legacy command-line alias.
- `--theta`: Dimensionless theta-code parameter for the initial state
  (-0.5 -> |0>, 0 -> |+>, 0.5 -> |1>)
- `--n_trajectories`: Number of trajectories for Monte Carlo method
- `--peInt`: Noise strength (×10⁻³)
- `--bc`: Boundary conditions (open, periodic, antiperiodic). ED and TN
  continuous Ising-family MPO construction honor them. TN Trotter currently
  supports open boundaries only and rejects non-open boundaries. Rydberg
  dynamics should use open boundaries until a non-open Rydberg convention is
  specified.
- `--measure_modes`: Record Bogoliubov mode observables `h_k` and occupations
  for integrable Ising k-space diagnostics with periodic or antiperiodic
  boundary conditions.
- `--J`: Ising coupling strength (default 1.0)
- `--h`: Transverse field strength (default -2.0). Note the default sits in the
  paramagnetic phase (|h| > J at J = 1), not at the critical point; scripts that
  assume h = 1.0 must pass it explicitly.

### Testing Commands

```bash
# Run quick test on macOS (using gtimeout instead of timeout)
gtimeout 60 julia --startup-file=no -t 1 Cooling.jl --N 4 --problem niIsing --backend TN --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.1 --te 0.5 --steps 5 --Dmax 10

# Run quick test on Linux
timeout 60 julia --startup-file=no -t 1 Cooling.jl --N 4 --problem niIsing --backend TN --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.1 --te 0.5 --steps 5 --Dmax 10
```

### HPC Cluster Submission

In this repository the scripts live in `clusters/`, but they are **not** run from
there. `clusters/upload_scripts_to_remote.sh` rsyncs the `.sh` files flat into
the cluster's project root, so on the cluster `SubmitCooling.sh`, `config.sh`,
`JobCooling.sh`, and `Cooling.jl` all sit side by side. Both the relative
`source config.sh` in the submit script and the relative `Cooling.jl` in
`JobCooling.sh`'s `srun` line depend on that flat layout — running the submit
script from inside `clusters/` locally would make `$SLURM_SUBMIT_DIR` the wrong
directory and the job would not find `Cooling.jl`.

`SubmitCooling.sh` is a driver: it loops over parameters and calls
`sbatch --array=... JobCooling.sh` itself (`clusters/SubmitCooling.sh:25`), so it
is run directly on the login node rather than submitted.

⚠️ **The cluster scripts are stale and will not run as-is.** They predate two
refactors and have not been updated:

1. `clusters/JobCooling.sh` (lines 30 and 36) passes `--method=$METHOD`, but
   `src/argparse.jl` has no `--method`; it takes the typed triple `--backend`,
   `--sim_method`, `--evolution_method`. ArgParse rejects the job before the
   simulation starts.
2. `METHOD` still carries the retired names (`config.sh` defaults it to `MPS`;
   `SubmitCooling.sh` branches on `MPO` / `TrotterMPS`), and `config.sh:19`
   builds `Sim${METHOD}Dmax...` filenames — the `SimMPS`/`SimMPO`/`SimTrotterMPS`
   convention that the File Naming Convention section below says was replaced by
   `SimTN` / `SimED`.

Fixing this means deciding how the old `METHOD` values map onto the typed triple,
which is a product decision — do not treat the snippet below as a working recipe
until `clusters/` is updated.

```bash
# On the cluster, from the project root (see clusters/upload_scripts_to_remote.sh)
# — currently broken, see the two points above.
bash SubmitCooling.sh
bash SubmitOptCooling.sh   # optimization driver (DEPRECATED - needs refactoring)
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

`src/CoolingTNS.jl` holds the authoritative include list and load order; the
groupings below follow it.

**Core Unified Files**:
- `src/cooling_evolution.jl`: Main cooling evolution with unified TN+ED dispatch
- `src/system_hamiltonian.jl`: System Hamiltonian construction (TN+ED unified)
- `src/system_bath_hamiltonian.jl`: System+bath Hamiltonian construction (TN+ED unified) 
- `src/ground_state.jl`: Unified ground state computation (TN+ED)
- `src/initial_state.jl`: Initial state preparation (TN+ED unified)
- `src/setup.jl`: Problem setup with backend dispatch
- `src/native_gate_cooling.jl`: Hardware-native Rydberg gate-compiled cooling
  circuit, unified across ED and TN (structure written once, backend-specific
  steps reached by dispatch)

**Types and Schema**:
- `src/parameter_types.jl`: Type definitions for parameters
- `src/cooling_types.jl`: CoolingProblem and QuantumState types
- `src/result_keys.jl`: Canonical result-dictionary keys and schema labels
- `src/multi_frequency_schedules.jl`: Single source of truth for multi-frequency
  detuning schedule names

**Support Files**:
- `src/coupling_utils.jl`: Coupling operator parsing
- `src/interleaved_layout.jl`: Single source of truth for the interleaved
  system-bath site convention (`s_i` at `2i-1`, `b_i` at `2i`)
- `src/utils.jl`: General utilities and file I/O
- `src/utils_mps.jl` / `src/utils_mpo.jl`: TN-specific utilities
- `src/noise.jl`: Noise models
- `src/argparse.jl`: Command-line argument parsing
- `src/state_manipulation.jl`: Dispatched state operations
- `src/bath_measurements.jl`: Bath measurement functions
- `src/trotter.jl`: Trotter evolution support
- `src/evolution.jl`: Evolution utilities
- `src/setup_system.jl`: System setup utilities
- `src/multi_frequency.jl`: Multi-frequency (multi-Δ) cooling helpers —
  detuning selection and low-lying gap computation

**Mode / k-space Analysis**:
- `src/mode_analysis.jl`: Analytic Ising mode structure — parameter mapping
  between the notes' θ-form and the code's (J, h), dispersion, and k-grids
- `src/dispersion.jl`: Compatibility dispersion and k-space helpers used by the
  plotting scripts, built on `mode_analysis.jl` conventions
- `src/tn_mode_observables.jl`: MPS/MPO mode observables via split-string
  correlators

**ED Backend**:
- `src/ed_backend.jl`: ED backend using complex state vectors and density
  matrices (`ComplexF64`), built on LinearAlgebra + SparseArrays + KrylovKit
- `src/ed_backend_complex_jw.jl`: The single source of truth for the complex
  Jordan-Wigner transform in the `MapToSpin.tex` convention. It is a
  domain-specific module, not an ED copy of a TN file — no TN counterpart
  exists or should be created.
- `src/cooling_evolution_ed_shared.jl`: Shared ED helper routines (bath ground
  state, combined-state preparation, ED measurement recording) factored out for
  DRY reuse *within* the ED path. The dispatch entry points that call them live
  in the unified `cooling_evolution.jl`; this is not a per-backend twin of it.

**Plotting** (outside `src/`, loaded on demand):
- `scripts/plotting/plotting.jl`: Visualization. It is *not* part of the
  `CoolingTNS` module — `Cooling.jl`, `optCooling.jl`, `plotCooling.jl`, and
  `plotOptCooling.jl` `include` it at run time when plotting is requested.

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

1. Results are saved as HDF5 files with backend and simulation method in the
   filename (`SimTNDM`, `SimTNMC`, `SimEDDM`, `SimEDMC` — see File Naming
   Convention below)
2. NO method names (MPS/MPO/TrotterMPS) in filenames anymore
3. The ED backend is the reference that validates tensor network results:
   `test/test_correctness.jl` and `test/test_ed_tn_density_channel.jl` are the
   TN-vs-ED cross-validators
4. Plotting scripts (`plotCooling.jl`, `plotOptCooling.jl`) generate publication-quality figures

### File Naming Convention

Built by `create_filename` in `src/utils.jl` — read it for the authoritative
rule; do not hand-construct paths from this summary. The name is three
underscore-joined groups, `Cooling_{ham}_{coupling}_{sim}`, with no underscores
inside a group.

The sim group starts `Sim{backend}{sim_method}` — backend `TN`/`ED`, sim_method
`DM`/`MC` — so the four stems are `SimTNDM`, `SimTNMC`, `SimEDDM`, `SimEDMC`.
Suffixes are then appended **conditionally**, which is what makes exact paths
hard to guess:

| Suffix | Appended when |
|---|---|
| `Dmax{D}` | TN backend **and** `Dmax != 100` (the default is omitted) |
| `tau{τ}` | `evolution_method isa TrotterEvolution` |
| `pe{n}` | `pe > 0`, as `round(pe * 1000)` |

- The retired `SimMPS` / `SimMPO` / `SimTrotterMPS` names are gone.
- **Simulation method is still in the name** (`DM`/`MC`).
- The evolution method has no literal name in the filename, but it is still
  distinguishable: Trotter runs carry a `tau` suffix and continuous runs do not.

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
- Automatically computes the raw Fourier occupation \(\tilde n_k=\langle\tilde a_k^\dagger \tilde a_k\rangle\) using the notes-aligned Jordan-Wigner transformation
- Separately records Bogoliubov mode observables \(h_k\), whose energy contributions are \(E_k=(\Lambda/2)\,\mathrm{coeff}_k\langle h_k\rangle\)
- Do not identify the raw Fourier occupation \(\tilde n_k\) with a Bogoliubov mode energy; `plot_ek_evolution.jl` deliberately refuses Fourier occupations as energies
- Only enabled for Ising model (integrable system)
- In the chosen-operator vacuum, Bogoliubov occupations \(n_k^{\mathrm{Bog}}=(h_k+1)/2\) vanish; raw Fourier occupations \(\tilde n_k\) instead follow the Jordan-Wigner convention in `Notes/NotesED/MapToSpin.tex`

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
- **TN density_matrix + continuous is unsupported**: `evolve_cooling_step` for
  `CoolingProblem{TNBackend}` with
  `UnifiedSimulationParameters{DensityMatrix,ContinuousEvolution}` raises an
  error in `src/cooling_evolution.jl` — ITensors' TDVP does not evolve an MPO.
  This is the one missing TN combination; the other three (MC+continuous,
  MC+trotter, DM+trotter) have evolution and measurement methods. Use one of
  those instead.
- **Precompilation**: Long precompilation times due to ITensors dependencies eating tokens during debugging

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
- **Optimization Driver**: `optCooling.jl` uses the typed backend and simulation-parameter interface; `MPS` and `MPO` are accepted only as backward-compatible input aliases before typed dispatch.

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
- Clean module structure with no external circuit-simulator dependency in the ED backend

### ⚠️ Known Issues

**Physics Problems:**
- **ED Cooling Rate**: Cooling is very slow with current parameters - may need stronger coupling or longer evolution times

**Performance:**
- **Precompilation Time**: Long compilation due to ITensors dependencies
- **ED Scaling**: The binding constraint is memory, and it is far tighter than
  the "N ≤ 12" this file used to claim. The system-bath layout doubles the qubit
  count, so ED works in dimension `2^(2N)`, and the hot structures are *dense*:
  `prepare_combined_state_ed` allocates `zeros(ComplexF64, 2^(2N), 2^(2N))` for
  the density-matrix path, and `_get_eigendecomp` calls `eigen(Hermitian(Matrix(H)))`
  on the same dimension for every evolution path (ED "Trotter" also uses the full
  `exp(-iHt)` in sub-steps). One such matrix costs:

  | N | dim `2^(2N)` | dense ComplexF64 matrix |
  |---|---|---|
  | 6 | 4096 | 0.25 GiB |
  | 7 | 16384 | 4 GiB |
  | 8 | 65536 | 64 GiB |
  | 10 | 1048576 | 16 TiB |
  | 12 | 16777216 | 4 PiB |

  Nothing in the code enforces a ceiling, so an oversized run simply exhausts
  memory.

### 🔧 Development Guidelines

**For ED Backend Usage:**
- N ≤ 6 is the comfortable working range (0.25 GiB per dense matrix); N = 7
  (4 GiB) is the practical limit on a workstation and N = 8 (64 GiB) needs a
  large-memory node. Do not follow the old "N ≤ 10 / N ≤ 12" guidance — see
  **ED Scaling** above for the derivation. Use the TN backend beyond this.
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
   # From the BdG block near Eq. \ref{eq:bdg_block} in MapToSpin.tex,
   # converted to code units
   # with θ = atan(h, J)
   ε_k = 2sqrt(J^2+h^2) * sqrt(1 - sin(2θ) * cos(2π*k/N))
   
   # NOT: From equation (260) in the notes
   ```

3. **Label Conventions**: Use descriptive labels that won't change if equations are reordered:
   - `\label{eq:spin_hamiltonian}` for the spin Hamiltonian
   - `\label{eq:JW_transformation}` for Jordan-Wigner transformation
   - `\label{eq:bdg_block}` for the BdG block
   - `\label{eq:hk_final}` for the spin-correlator expression of mode observables
   
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
