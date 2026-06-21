module CoolingTNS

# Import ITensors package
using ITensors
using ITensorMPS
using Printf
using SparseArrays
using HDF5

# For convenience, here we should just include all the files that are still being used

include("multi_frequency_schedules.jl") # Schedule names used by parameter types
include("parameter_types.jl")      # Define parameter types first
include("cooling_types.jl")        # CoolingProblem and QuantumState types
include("result_keys.jl")          # Public result dictionary keys

include("utils.jl")
include("coupling_utils.jl")
include("interleaved_layout.jl")
include("utils_mps.jl")
include("utils_mpo.jl")

# Analytical dispersion relations (pure math, no plotting deps)
include("dispersion.jl")

include("policy.jl")
include("argparse.jl")
include("noise.jl")

# Include ED backend
include("ed_backend.jl")
include("mode_analysis.jl")          # Parameter mapping, dispersion, k-grid
include("ed_backend_complex_jw.jl")  # Complex JW (notes convention) — single source of truth
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
export setup_tn_multifrequency_problem_from_system
export tdvp_sweep_observer
export initial_product_angle
export theta_code_from_initial_product_angle, theta_site_amplitudes
export CoolingProblem, QuantumState
export CoolingBackend, EDBackend, TNBackend
export SimulationMethod, DensityMatrix, MonteCarloWavefunction
export tn_method_maxdim, tn_trotter_maxdim
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
export setup_common_parameters, create_filename, save_results, HDF5_PARSED_ARGS_GROUP
export get_backend, get_sim_method, get_evolution_method, mean_last_window, relative_energy
export create_sim_params_from_args, normalize_optimization_args!
export parse_coupling, coupling_operator_terms, get_bath_operator
export bath_ground_state_amplitudes, get_bath_ground_state
export interleaved_total_sites
export interleaved_system_site, interleaved_bath_site
export interleaved_system_sites, interleaved_bath_sites
export interleaved_system_indices, interleaved_bath_indices
export interleaved_bit_position
export interleaved_system_bit, interleaved_bath_bit
export interleaved_system_bits, interleaved_bath_bits
export interleaved_basis_state, interleaved_system_basis_state
# Result dictionary keys
export RESULT_ENERGY, RESULT_RELATIVE_ENERGY, RESULT_GROUND_STATE_OVERLAP, RESULT_PURITY
export RESULT_BATH_MAGNETIZATION, RESULT_BATH_SAMPLE_MAGNETIZATION
export RESULT_MOMENTUM_DISTRIBUTION, RESULT_K_VALUES, RESULT_MOMENTUM_GF, RESULT_MOMENTUM_GF_SOURCE
export RESULT_MODE_GF, RESULT_MODE_GF_SOURCE
export RESULT_MODE_HK, RESULT_MODE_NK, RESULT_MODE_K_INDICES, RESULT_MODE_ENERGIES
export RESULT_MODE_MEASUREMENT_CYCLES
export RESULT_MODE_HK_TRAJECTORIES, RESULT_MODE_NK_TRAJECTORIES
export RESULT_MODE_HK_STDERR, RESULT_MODE_NK_STDERR
export RESULT_DELTA_LIST, RESULT_TE_LIST, RESULT_DELTA_VALUES
export RESULT_SCHEDULE, RESULT_RANDOMIZE_TIMES
export RESULT_REQUESTED_STEPS, RESULT_COMPLETED_STEPS, RESULT_STOP_REASON
export RESULT_N_TRAJECTORIES
export RESULT_ENERGY_TRAJECTORIES, RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES
export RESULT_ENERGY_STD, RESULT_GROUND_STATE_OVERLAP_STD
export RESULT_BOND_DIMS, RESULT_TRUNCATION_ERRORS, RESULT_TRUNCATION_ERROR_HISTORY_STATUS
export TRUNCATION_ERROR_HISTORY_NOT_RECORDED
export TRUNCATION_ERROR_HISTORY_LEGACY_MISSING, TRUNCATION_ERROR_HISTORY_MEASURED
export TRUNCATION_ERROR_HISTORY_EMPTY
export RESULT_RENYI_ENTROPY, RESULT_FINAL_STATE
export RESULT_KEYS
# Dispersion relations used by plotting helpers; implementations follow mode_analysis.jl
export generate_k_values, compute_energy_dispersion, compute_ground_state_occupation

# Mode analysis (canonical parameter bridge and dispersion)
export theta_from_Jh, Jh_from_theta, energy_scale
export mode_energy, mode_energy_Jh, mode_energies_Jh, w_k_coefficient, r_k_coefficient
export obc_bdg_matrices, obc_bdg_matrix, obc_mode_energies, obc_mode_energies_Jh
export bogoliubov_angle, coeff_k, vacuum_energy, vacuum_energy_Jh
export allowed_k_indices, fermionic_bc, parity_operator_code
export is_generic_mode, generic_k_indices
export mode_occupation_from_hk, ising_energy_from_mode_hk
export supports_ising_fourier_observables
export ising_mode_detuning_preserves_px, ising_mode_detuning_has_special_modes
export ising_mode_detuning_reference
export bath_detuning_energy, nearest_bath_resonance_indices

# Multi-frequency cooling helpers
export uniform_delta_grid, multi_frequency_cycle_choice, multi_frequency_cycle_sequence
export MULTI_FREQUENCY_SCHEDULES
export parse_multi_frequency_schedule, validate_multi_frequency_schedule
export multi_frequency_schedule_token
export compute_excitation_gaps, spectral_delta_values

# Complex JW (notes convention)
export jordan_wigner_transform_complex, pauli_y_complex
export measure_momentum_distribution_ed_clean
# Bogoliubov mode observables and positive quasiparticle gaps
export measure_hk, measure_all_mode_observables, measure_all_mode_energies, measure_state_parity

end
