module CoolingTNS

# Import ITensors package
using ITensors
using ITensorMPS
using Printf
using SparseArrays

# For convenience, here we should just include all the files that are still being used

include("parameter_types.jl")      # Define parameter types first
include("cooling_types.jl")        # CoolingProblem and QuantumState types
include("result_keys.jl")          # Public result dictionary keys

include("utils.jl")
include("coupling_utils.jl")
include("utils_mps.jl")
include("utils_mpo.jl")

# Analytical dispersion relations (pure math, no plotting deps)
include("dispersion.jl")

include("policy.jl")
include("argparse.jl")
include("noise.jl")

# Include ED backend
include("ed_backend.jl")
include("ed_backend_complex_jw.jl")  # Complex JW (notes convention) — single source of truth
include("mode_analysis.jl")          # Parameter mapping, dispersion, k-grid
include("tn_mode_observables.jl")    # MPS mode observables using split-string correlators
include("multi_frequency.jl")       # Multi-frequency (multi-Δ) cooling helpers

include("bath_measurements.jl")    # Dispatched bath measurement functions
include("state_manipulation.jl")   # Dispatched state manipulation functions

# Core implementations (all using dispatch by default)
include("system_hamiltonian.jl")    # System Hamiltonian construction
include("ground_state.jl")          # Ground state computation
include("setup_system.jl")          # System setup using the above
include("system_bath_hamiltonian.jl") # System-bath coupling
include("trotter.jl")               # Trotter circuit construction
include("initial_state.jl")         # Initial state preparation
include("evolution.jl")             # Time evolution functions
include("cooling_evolution_ed_shared.jl") # Shared ED backend functions
include("cooling_evolution.jl")     # Cooling evolution
include("setup.jl")                 # setup_problem implementations


export setup_problem, run_cooling, run_cooling_multi_freq, setup_initial_state
export CoolingProblem, QuantumState
export CoolingBackend, EDBackend, TNBackend
export SimulationMethod, DensityMatrix, MonteCarloWavefunction
export EvolutionMethod, ContinuousEvolution, TrotterEvolution
# Export new parameter types and functions
export CouplingParameters, SimulationParameters, CoolingResults
export BasicCouplingParameters, MultiFrequencyCouplingParameters, OptimizationCouplingParameters
export UnifiedSimulationParameters
export HamiltonianParameters, IsingParameters, NiIsingParameters, RydbergParameters
export HamiltonianModel, IsingModel, NiIsingModel, RydbergModel
export DensityMatrixResults, MonteCarloResults, TensorNetworkResults
export create_coupling_params, create_sim_params, create_results
export to_dict
export setup_common_parameters, create_filename, save_results
export get_backend, get_sim_method, get_evolution_method, mean_last_window, relative_energy
export parse_coupling, coupling_operator_terms, get_bath_operator
# Result dictionary keys
export RESULT_ENERGY, RESULT_GROUND_STATE_OVERLAP, RESULT_PURITY
export RESULT_BATH_MAGNETIZATION, RESULT_BATH_SAMPLE_MAGNETIZATION
export RESULT_MOMENTUM_DISTRIBUTION, RESULT_K_VALUES, RESULT_MOMENTUM_GF, RESULT_MOMENTUM_GF_SOURCE
export RESULT_MODE_GF, RESULT_MODE_HK, RESULT_MODE_NK, RESULT_MODE_K_INDICES, RESULT_MODE_ENERGIES
export RESULT_DELTA_LIST, RESULT_TE_LIST, RESULT_DELTA_VALUES
export RESULT_SCHEDULE, RESULT_RANDOMIZE_TIMES, RESULT_N_TRAJECTORIES
export RESULT_KEYS
# Dispersion relations used by plotting helpers; implementations follow mode_analysis.jl
export generate_k_values, compute_energy_dispersion, compute_ground_state_occupation

# Mode analysis (canonical parameter bridge and dispersion)
export theta_from_Jh, Jh_from_theta, energy_scale
export mode_energy, mode_energy_Jh, w_k_coefficient, r_k_coefficient
export bogoliubov_angle, coeff_k, vacuum_energy, vacuum_energy_Jh
export allowed_k_indices, fermionic_bc, parity_operator_code
export mode_occupation_from_hk

# Multi-frequency cooling helpers
export uniform_delta_grid, compute_excitation_gaps, spectral_delta_values

# Complex JW (notes convention)
export jordan_wigner_transform_complex, pauli_y_complex
export measure_momentum_distribution_ed_clean
# Mode energy observables (Phase 2+3)
export measure_hk, measure_all_mode_energies, measure_state_parity

end
