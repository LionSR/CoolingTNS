"""
    cooling_evolution.jl

Cooling evolution implementations using multiple dispatch.
"""

using ITensors
using ITensorMPS
using ITensors: apply
using LinearAlgebra
using SparseArrays
using Random
using Statistics

# ED backend types and shared functions are already included by CoolingTNS.jl

# ============================================================================
# Main Cooling Evolution Interface
# ============================================================================

"""
    run_cooling(problem::CoolingProblem, state::QuantumState, coupling_params, sim_params, ham_params)

Run cooling simulation with type dispatch. This is the main entry point that delegates
to specialized implementations based on backend and method types.
"""
function run_cooling(problem::CoolingProblem{B}, state::QuantumState{B,S,E}, 
                    coupling_params, sim_params, 
                    ham_params; measure_modes::Bool=false) where {B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    
    # Common setup
    steps = coupling_params.steps
    te = coupling_params.te
    pe = sim_params.pe
    
    # Initialize measurement arrays
    measurements = initialize_measurements(problem, state, steps; measure_modes=measure_modes, ham_params=ham_params)
    
    # Initial measurements
    perform_measurements!(measurements, 1, problem, state, ham_params)
    
    # Print initial status
    print_cooling_status(1, measurements, ham_params, state)
    
    # Main cooling loop
    for step in 2:steps+1
        # Prepare system+bath state
        combined_state = prepare_combined_state(problem, state)
        
        # Evolve the combined system
        evolved_state = evolve_cooling_step(problem, combined_state, te, sim_params, ham_params)
        
        # Apply noise if requested
        if pe > 0
            evolved_state = apply_noise(evolved_state, problem, pe)
        end
        
        # Process bath and update system state
        state_and_bath_info = process_bath_and_update(problem, evolved_state, state, sim_params)
        if isa(state_and_bath_info, Tuple)
            state, bath_info = state_and_bath_info
        else
            state = state_and_bath_info
            bath_info = nothing
        end
        
        # Perform measurements
        perform_measurements!(measurements, step, problem, state, ham_params, bath_info)
        
        # Print progress
        if step % 10 == 0 || step == steps + 1
            print_cooling_status(step, measurements, ham_params, state)
        end
    end
    
    println("Cooling completed")
    
    # Return results
    return compile_results(measurements, sim_params)
end

# ============================================================================
# Multi-frequency Cooling (multi-Δ)
# ============================================================================

"""
    run_cooling_multi_freq(problem, state, mf_params::MultiFrequencyCouplingParameters, sim_params, ham_params; measure_modes=false)

Run a multi-frequency cooling protocol, cycling the bath detuning Δ through
`mf_params.delta_values` according to `mf_params.schedule`. Optionally randomizes
the evolution time each step (`randomize_times=true`).

The returned results dictionary includes:

- `delta_list[step]`: Δ used for the evolution that produced measurement `step`
  (with `delta_list[1]=NaN` for the initial measurement).
- `te_list[step]`: evolution time used at that step (similarly `NaN` at step 1).
- `delta_values`: the set of Δ values used in the protocol.
"""
function run_cooling_multi_freq(
    problem::CoolingProblem{B},
    state::QuantumState{B,S,E},
    mf_params::MultiFrequencyCouplingParameters,
    sim_params,
    ham_params;
    measure_modes::Bool=false,
) where {B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}

    steps = mf_params.steps
    pe = sim_params.pe

    measurements = initialize_measurements(problem, state, steps; measure_modes=measure_modes, ham_params=ham_params)
    measurements[RESULT_DELTA_LIST] = fill(NaN, steps + 1)
    measurements[RESULT_TE_LIST] = fill(NaN, steps + 1)
    measurements[RESULT_DELTA_VALUES] = copy(mf_params.delta_values)
    measurements[RESULT_SCHEDULE] = string(mf_params.schedule)
    measurements[RESULT_RANDOMIZE_TIMES] = mf_params.randomize_times

    # Initial measurements
    perform_measurements!(measurements, 1, problem, state, ham_params)
    print_cooling_status(1, measurements, ham_params, state)

    R = length(mf_params.delta_values)

    for step in 2:steps+1
        # Pick frequency
        r = _pick_delta_index(step, R, mf_params.schedule)
        delta_r = mf_params.delta_values[r]

        # Pick evolution time
        te_step = mf_params.randomize_times ? (rand() * 2 * mf_params.te) : mf_params.te

        measurements[RESULT_DELTA_LIST][step] = delta_r
        measurements[RESULT_TE_LIST][step] = te_step

        # Per-step coupling parameters (encodes Δ_r and possibly step-specific time)
        coupling_step = BasicCouplingParameters(mf_params.coupling, mf_params.g, steps, te_step, delta_r)

        # Prepare system+bath state
        combined_state = prepare_combined_state(problem, state)

        # Evolve with the step-dependent coupling parameters
        evolved_state = if te_step <= 0
            combined_state
        else
            evolve_cooling_step_dynamic(problem, combined_state, coupling_step, te_step, sim_params, ham_params)
        end

        # Apply noise if requested
        if pe > 0
            evolved_state = apply_noise(evolved_state, problem, pe)
        end

        # Process bath and update system state
        state_and_bath_info = process_bath_and_update(problem, evolved_state, state, sim_params)
        if isa(state_and_bath_info, Tuple)
            state, bath_info = state_and_bath_info
        else
            state = state_and_bath_info
            bath_info = nothing
        end

        # Perform measurements
        perform_measurements!(measurements, step, problem, state, ham_params, bath_info)

        # Print progress
        if step % 10 == 0 || step == steps + 1
            print_cooling_status(step, measurements, ham_params, state)
        end
    end

    println("Cooling completed")
    return compile_results(measurements, sim_params)
end

# Dispatch hook: allow run_cooling(..., mf_params::MultiFrequencyCouplingParameters, ...) to work.
function run_cooling(
    problem::CoolingProblem{B},
    state::QuantumState{B,S,E},
    coupling_params::MultiFrequencyCouplingParameters,
    sim_params,
    ham_params;
    measure_modes::Bool=false,
) where {B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    return run_cooling_multi_freq(problem, state, coupling_params, sim_params, ham_params; measure_modes=measure_modes)
end

# Internal helper: pick which Δ index to use at this step.
function _pick_delta_index(step::Int, R::Int, schedule::Symbol)
    schedule == :round_robin && return mod1(step - 1, R)
    schedule == :random && return rand(1:R)
    throw(ArgumentError("Unknown multi-frequency schedule=$schedule (expected :round_robin or :random)"))
end

# Evolve with step-dependent coupling parameters (e.g. changing bath detuning Δ).
evolve_cooling_step_dynamic(::CoolingProblem, _, ::BasicCouplingParameters, _, _, _) =
    error("evolve_cooling_step_dynamic not implemented for this backend/method combination")

function evolve_cooling_step_dynamic(
    problem::CoolingProblem{TNBackend},
    ψ_sb::MPS,
    coupling_step::BasicCouplingParameters,
    te_step::Float64,
    sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, ContinuousEvolution},
    ham_params,
)
    sites = problem.extra.sites

    δ = coupling_step.delta
    δ === nothing && throw(ArgumentError("Multi-frequency TDVP evolution requires coupling_step.delta"))
    δ = Float64(δ)

    # Cache H_step(Δ) since the Hamiltonian does not depend on the randomized time.
    cache = get(problem.extra, :H_cache, nothing)
    H_step = if cache isa AbstractDict
        get!(cache, δ) do
            construct_system_bath_hamiltonian(ham_params, problem.backend, sites, coupling_step)
        end
    else
        construct_system_bath_hamiltonian(ham_params, problem.backend, sites, coupling_step)
    end

    return evolve_state(ham_params, sim_params, problem.backend, H_step, ψ_sb, te_step, sites)
end

function evolve_cooling_step_dynamic(
    problem::CoolingProblem{TNBackend},
    ψ_sb::MPS,
    coupling_step::BasicCouplingParameters,
    te_step::Float64,
    sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, TrotterEvolution},
    ham_params,
)
    sites = problem.extra.sites

    δ = coupling_step.delta
    δ === nothing && throw(ArgumentError("Multi-frequency Trotter evolution requires coupling_step.delta"))
    δ = Float64(δ)

    cache = get(problem.extra, :gates_cache, nothing)
    gates = if cache isa AbstractDict
        get!(cache, δ) do
            build_trotter_circuit_interleaved(ham_params, problem.backend, sites, coupling_step, sim_params)
        end
    else
        build_trotter_circuit_interleaved(ham_params, problem.backend, sites, coupling_step, sim_params)
    end

    return evolve_state(ham_params, sim_params, problem.backend, gates, ψ_sb, te_step, sites; gates=gates)
end

function evolve_cooling_step_dynamic(
    problem::CoolingProblem{TNBackend},
    ρ_sb::MPO,
    coupling_step::BasicCouplingParameters,
    te_step::Float64,
    sim_params::UnifiedSimulationParameters{DensityMatrix, TrotterEvolution},
    ham_params,
)
    sites = problem.extra.sites

    δ = coupling_step.delta
    δ === nothing && throw(ArgumentError("Multi-frequency Trotter evolution requires coupling_step.delta"))
    δ = Float64(δ)

    cache = get(problem.extra, :gates_cache, nothing)
    gates = if cache isa AbstractDict
        get!(cache, δ) do
            build_trotter_circuit_interleaved(ham_params, problem.backend, sites, coupling_step, sim_params)
        end
    else
        build_trotter_circuit_interleaved(ham_params, problem.backend, sites, coupling_step, sim_params)
    end

    ρ_evolved = evolve_state(ham_params, sim_params, problem.backend, gates, ρ_sb, te_step, sites)
    ρ_evolved /= tr(ρ_evolved)
    return ρ_evolved
end

function evolve_cooling_step_dynamic(
    problem::CoolingProblem{EDBackend},
    state_total::Union{EDStateVector, EDDensityMatrix},
    coupling_step::BasicCouplingParameters,
    te_step::Float64,
    sim_params::UnifiedSimulationParameters,
    ham_params,
)
    δ = coupling_step.delta
    δ === nothing && throw(ArgumentError("Multi-frequency ED evolution requires coupling_step.delta"))
    δ = Float64(δ)

    # Cache H_step(Δ) since the Hamiltonian does not depend on the randomized time.
    cache = get(problem.extra, :H_cache, nothing)
    H_step = if cache isa AbstractDict
        get!(cache, δ) do
            construct_system_bath_hamiltonian(ham_params, problem.backend, 2 * ham_params.N, coupling_step)
        end
    else
        construct_system_bath_hamiltonian(ham_params, problem.backend, 2 * ham_params.N, coupling_step)
    end

    tau = sim_params.evolution_method isa TrotterEvolution ? sim_params.tau : nothing
    return evolve_cooling_step_ed(H_step, state_total, te_step, tau)
end

# ============================================================================
# Helper Functions - Generic Interface
# ============================================================================

"""Initialize measurement arrays based on backend and method"""
function initialize_measurements(problem::CoolingProblem, state::QuantumState, steps::Int;
                                 measure_modes::Bool=false, ham_params=nothing)
    # Basic measurements available for all backends
    measurements = Dict{String, Any}(
        RESULT_ENERGY => zeros(Float64, steps + 1),
        RESULT_GROUND_STATE_OVERLAP => zeros(Float64, steps + 1)
    )
    
    # Add backend-specific measurements
    add_backend_measurements!(measurements, problem, state, steps)
    
    # Add mode energy measurements if requested (ED + Ising + PBC/APBC only)
    if measure_modes
        add_mode_measurements!(measurements, problem, state, steps, ham_params)
    end
    
    return measurements
end

# Default: no additional measurements
add_backend_measurements!(_, _, _, _) = nothing

# TN Density matrix methods get purity
function add_backend_measurements!(measurements, ::CoolingProblem{TNBackend}, ::QuantumState{TNBackend,DensityMatrix,E}, steps) where E
    measurements[RESULT_PURITY] = zeros(Float64, steps + 1)
    measurements[RESULT_BATH_MAGNETIZATION] = zeros(Float64, steps + 1)
end

# ED Density matrix methods get purity + k-space measurements
function add_backend_measurements!(measurements, problem::CoolingProblem{EDBackend}, ::QuantumState{EDBackend,DensityMatrix,E}, steps) where E
    measurements[RESULT_PURITY] = zeros(Float64, steps + 1)
    measurements[RESULT_BATH_MAGNETIZATION] = zeros(Float64, steps + 1)

    # Add k-space measurements for ED with periodic/antiperiodic BC (only for Ising model)
    add_kspace_measurements!(measurements, problem, steps)
end

# TN Monte Carlo methods track bath samples
function add_backend_measurements!(measurements, ::CoolingProblem{TNBackend}, ::QuantumState{TNBackend,MonteCarloWavefunction,E}, steps) where E
    measurements[RESULT_BATH_SAMPLE_MAGNETIZATION] = zeros(Float64, steps + 1)
end

# ED Monte Carlo tracks bath magnetization + k-space
function add_backend_measurements!(measurements, problem::CoolingProblem{EDBackend}, ::QuantumState{EDBackend,MonteCarloWavefunction,E}, steps) where E
    measurements[RESULT_BATH_MAGNETIZATION] = zeros(Float64, steps + 1)
    add_kspace_measurements!(measurements, problem, steps)
end

# Helper for k-space measurements (ED only, Ising PBC/APBC)
function add_kspace_measurements!(measurements, problem::CoolingProblem{EDBackend}, steps)
    if haskey(problem.extra, :ham_params)
        ham_params = problem.extra.ham_params
        if _supports_ising_fourier_observables(ham_params)
            N = ham_params.N
            measurements[RESULT_MOMENTUM_DISTRIBUTION] = zeros(Float64, steps + 1, N)
            measurements[RESULT_K_VALUES] = zeros(Float64, N)
        end
    end
end

function _mode_measurement_ham_params(problem::CoolingProblem, ham_params)
    return ham_params !== nothing ? ham_params : (haskey(problem.extra, :ham_params) ? problem.extra.ham_params : nothing)
end

function _add_ising_mode_measurements!(measurements, problem::CoolingProblem, ham_params)
    hp = _mode_measurement_ham_params(problem, ham_params)
    _supports_ising_fourier_observables(hp) || return nothing

    # Determine gF from the ground state's parity sector.
    # This ensures consistent mode measurement even for mixed states.
    px = measure_state_parity(problem.ϕ₀, hp.N)
    measurements[RESULT_MODE_GF] = _reference_fermionic_bc(hp.bc, px)

    # Preallocate arrays after the first measurement call, when the mode count is known.
    measurements[RESULT_MODE_HK] = nothing
    measurements[RESULT_MODE_NK] = nothing
    measurements[RESULT_MODE_K_INDICES] = nothing
    measurements[RESULT_MODE_ENERGIES] = nothing
    return nothing
end

function _supports_tn_cooling_fourier_observables(ham_params)
    _supports_ising_fourier_observables(ham_params) || return false
    # Current TN Ising Hamiltonian and Trotter builders omit the periodic/APBC
    # boundary bond, so automatic TN cooling mode diagnostics would mix an
    # open-chain evolved state with a periodic Fourier grid. See issue #124.
    return false
end

# Helper for mode energy measurements ⟨h_k⟩ (Ising PBC/APBC)
function add_mode_measurements!(measurements, problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend}, steps, ham_params)
    _add_ising_mode_measurements!(measurements, problem, ham_params)
end

function add_mode_measurements!(measurements, problem::CoolingProblem{TNBackend},
                                state::QuantumState{TNBackend,MonteCarloWavefunction,E},
                                steps, ham_params) where E<:EvolutionMethod
    hp = _mode_measurement_ham_params(problem, ham_params)
    if _supports_ising_fourier_observables(hp)
        @warn "TN cooling Fourier-mode measurements are disabled until TN Ising Hamiltonians honor periodic/APBC boundary conditions" issue=124
    end
    return nothing
end

# Fallback: mode measurements not supported for non-ED backends
add_mode_measurements!(measurements, problem, state, steps, ham_params) = nothing

"""Perform measurements on current state"""
function perform_measurements!(measurements, step::Int, problem::CoolingProblem, state::QuantumState, ham_params, bath_info=nothing)
    # Delegate to backend-specific measurement
    perform_backend_measurements!(measurements, step, problem, state, ham_params, bath_info)
end

"""Print cooling progress"""
function print_cooling_status(step::Int, measurements, ham_params, state::QuantumState)
    N = ham_params.N
    E = measurements[RESULT_ENERGY][step]
    overlap = measurements[RESULT_GROUND_STATE_OVERLAP][step]
    
    status = "Step $step: energy/N=$(E/N), overlap=$overlap"
    
    # Add method-specific info
    if haskey(measurements, RESULT_PURITY)
        status *= ", purity=$(measurements[RESULT_PURITY][step])"
    end
    
    println(status)
end

"""Compile results into standard format"""
function compile_results(measurements, sim_params)
    results = copy(measurements)
    
    # Add simulation metadata
    if sim_params isa UnifiedSimulationParameters{MonteCarloWavefunction, E} where E
        if sim_params.n_trajectories > 1
            results[RESULT_N_TRAJECTORIES] = sim_params.n_trajectories
        end
    end
    
    return results
end

"""
    create_results(measurements, sim_params)

Public alias for `compile_results`. CoolingTNS represents simulation outputs as a
`Dict{String,Any}` of measurement arrays; this helper copies the dictionary and
adds lightweight metadata (e.g. `n_trajectories` for Monte Carlo runs).
"""
create_results(measurements, sim_params) = compile_results(measurements, sim_params)

# ============================================================================
# Backend-Specific Implementations
# ============================================================================

# --- Tensor Network + Monte Carlo (unified for both evolution methods) ---

# Unified TN+MC prepare_combined_state for both Continuous and Trotter evolution
function prepare_combined_state(problem::CoolingProblem{TNBackend}, state::QuantumState{TNBackend,MonteCarloWavefunction,E}) where E<:EvolutionMethod
    sites = problem.extra.sites
    coupling = problem.extra.coupling_params.coupling
    return appendzeros_MPS(state.state, sites, coupling)
end

function evolve_cooling_step(problem::CoolingProblem{TNBackend}, ψ_sb::MPS, te::Float64, 
                           sim_params::UnifiedSimulationParameters{MonteCarloWavefunction,ContinuousEvolution}, 
                           ham_params)
    # Use the generic evolve_state from evolution.jl
    sites = problem.extra.sites
    return evolve_state(ham_params, sim_params, problem.backend, problem.H_sys_bath, ψ_sb, te, sites)
end

function apply_noise(ψ::MPS, problem::CoolingProblem{TNBackend}, pe::Float64)
    # Get sites from problem.extra
    sites = problem.extra.sites
    ψ_noisy = apply_depolarizing_noise(ψ, sites, pe)
    orthogonalize!(ψ_noisy, 2)
    return ψ_noisy
end

# Unified TN+MC bath processing for both Continuous and Trotter evolution
function process_bath_and_update(problem::CoolingProblem{TNBackend}, ψ_evolved::MPS,
                               state::QuantumState{TNBackend,MonteCarloWavefunction,E},
                               sim_params) where E<:EvolutionMethod
    sites = problem.extra.sites
    N_sys = length(sites) ÷ 2

    # `ψ_evolved` is discarded after bath sampling, so we can use the mutating
    # version to avoid an extra MPS copy.
    v_b, ψ_s = sample_bath!(ψ_evolved)

    if length(ψ_s) != N_sys
        @warn "After sampling bath, MPS has unexpected length" expected=N_sys actual=length(ψ_s)
    end

    truncate!(ψ_s; cutoff=sim_params.cutoff)
    normalize!(ψ_s)

    return QuantumState(state.backend, state.sim_method, state.evolution_method, ψ_s), v_b
end

# Unified TN+MC measurements for both Continuous and Trotter evolution
function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{TNBackend},
                                     state::QuantumState{TNBackend,MonteCarloWavefunction,E},
                                     ham_params, bath_info=nothing) where E<:EvolutionMethod
    ψ_s = state.state
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀

    if length(ψ_s) != length(H_sys)
        @warn "Dimension mismatch in measurements" MPS_length=length(ψ_s) H_sys_length=length(H_sys) phi0_length=length(ϕ₀) step=step
    end

    if step == 1 || length(ψ_s) == length(H_sys)
        measurements[RESULT_ENERGY][step] = real(inner(ψ_s', H_sys, ψ_s))
        measurements[RESULT_GROUND_STATE_OVERLAP][step] = abs2(inner(ψ_s, ϕ₀))
    else
        @warn "Skipping measurement due to dimension mismatch"
        measurements[RESULT_ENERGY][step] = measurements[RESULT_ENERGY][step-1]
        measurements[RESULT_GROUND_STATE_OVERLAP][step] = measurements[RESULT_GROUND_STATE_OVERLAP][step-1]
    end

    if haskey(measurements, RESULT_BATH_SAMPLE_MAGNETIZATION) && bath_info !== nothing
        measurements[RESULT_BATH_SAMPLE_MAGNETIZATION][step] = compute_bath_magnetization(problem.backend, state, bath_info, ham_params.N)
    end

    if haskey(measurements, RESULT_MODE_HK) && _supports_tn_cooling_fourier_observables(ham_params)
        if length(ψ_s) == ham_params.N
            gF_kwarg = haskey(measurements, RESULT_MODE_GF) ? measurements[RESULT_MODE_GF] : nothing
            k_indices, hk_values, εk_values = measure_all_mode_energies(ψ_s, ham_params; gF=gF_kwarg)
            n_modes = length(k_indices)

            if measurements[RESULT_MODE_HK] === nothing
                n_steps_total = size(measurements[RESULT_ENERGY], 1)
                measurements[RESULT_MODE_HK] = fill(NaN, n_steps_total, n_modes)
                measurements[RESULT_MODE_NK] = fill(NaN, n_steps_total, n_modes)
                measurements[RESULT_MODE_K_INDICES] = k_indices
                measurements[RESULT_MODE_ENERGIES] = εk_values
            end

            measurements[RESULT_MODE_HK][step, :] .= hk_values
            measurements[RESULT_MODE_NK][step, :] .= mode_occupation_from_hk(hk_values)
        else
            @warn "Skipping mode measurement due to dimension mismatch" MPS_length=length(ψ_s) expected_length=ham_params.N step=step
            if step > 1 && measurements[RESULT_MODE_HK] !== nothing
                measurements[RESULT_MODE_HK][step, :] .= measurements[RESULT_MODE_HK][step-1, :]
                measurements[RESULT_MODE_NK][step, :] .= measurements[RESULT_MODE_NK][step-1, :]
            end
        end
    end
end

# --- Exact Diagonalization + Density Matrix (shared for Continuous/Trotter) ---

function prepare_combined_state(problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend,DensityMatrix,E}) where E<:EvolutionMethod
    N_bath = problem.extra.ham_params.N
    coupling = problem.extra.coupling_params.coupling
    ρ_sys = _system_state_for_measurement(state.state, N_bath)
    return prepare_combined_state_ed(ρ_sys, N_bath, coupling)
end

function process_bath_and_update(::CoolingProblem{EDBackend}, ρ_evolved::EDDensityMatrix,
                               state::QuantumState{EDBackend,DensityMatrix,E}, _) where E<:EvolutionMethod
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ρ_evolved), nothing
end

# Evolution differs by tau
function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ρ_total::EDDensityMatrix, te::Float64,
                           ::UnifiedSimulationParameters{DensityMatrix,ContinuousEvolution}, _)
    return evolve_cooling_step_ed(problem.H_sys_bath, ρ_total, te, nothing)
end

function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ρ_total::EDDensityMatrix, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,TrotterEvolution}, _)
    return evolve_cooling_step_ed(problem.H_sys_bath, ρ_total, te, sim_params.tau)
end

function apply_noise(ρ::EDDensityMatrix, ::CoolingProblem{EDBackend}, pe::Float64)
    dim = 2^ρ.n_qubits
    ρ_noisy_data = (1 - pe) * ρ.data + pe * Matrix{Float64}(I, dim, dim) / dim
    return EDDensityMatrix(ρ_noisy_data, ρ.n_qubits)
end

function apply_noise(ψ::EDStateVector, ::CoolingProblem{EDBackend}, pe::Float64)
    return apply_depolarizing_ed(ψ, pe, collect(1:ψ.n_qubits))
end

# --- Exact Diagonalization + Monte Carlo (shared for Continuous/Trotter) ---

function prepare_combined_state(problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend,MonteCarloWavefunction,E}) where E<:EvolutionMethod
    coupling = problem.extra.coupling_params.coupling
    return prepare_combined_state_ed(state.state, problem.extra.ham_params.N, coupling)
end

function process_bath_and_update(problem::CoolingProblem{EDBackend}, ψ_evolved::EDStateVector,
                               state::QuantumState{EDBackend,MonteCarloWavefunction,E}, _) where E<:EvolutionMethod
    ψ_sys, bath_outcomes = process_bath_ed_monte_carlo(ψ_evolved, problem.extra.ham_params.N)
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ψ_sys), bath_outcomes
end

# Evolution differs by tau
function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ψ_total::EDStateVector, te::Float64,
                           ::UnifiedSimulationParameters{MonteCarloWavefunction,ContinuousEvolution}, _)
    return evolve_cooling_step_ed(problem.H_sys_bath, ψ_total, te, nothing)
end

function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ψ_total::EDStateVector, te::Float64,
                           sim_params::UnifiedSimulationParameters{MonteCarloWavefunction,TrotterEvolution}, _)
    return evolve_cooling_step_ed(problem.H_sys_bath, ψ_total, te, sim_params.tau)
end

# --- ED Backend Measurements (unified) ---

function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{EDBackend},
                                     state::QuantumState{EDBackend,S,E}, ham_params, bath_info=nothing) where {S<:SimulationMethod, E<:EvolutionMethod}
    perform_measurements_ed(measurements, step, state.state, problem.H_sys, problem.ϕ₀, ham_params, bath_info)
end

# --- Tensor Network + Monte Carlo + Trotter Evolution ---

function evolve_cooling_step(problem::CoolingProblem{TNBackend}, ψ_sb::MPS, te::Float64,
                           sim_params::UnifiedSimulationParameters{MonteCarloWavefunction,TrotterEvolution},
                           ham_params)
    sites = problem.extra.sites

    # Use interleaved gates (same circuit as DM+Trotter for consistency)
    interleaved_gates = get(problem.extra, :interleaved_gates, nothing)
    if interleaved_gates === nothing
        coupling_params = problem.extra.coupling_params
        interleaved_gates = build_trotter_circuit_interleaved(ham_params, problem.backend, sites, coupling_params, sim_params)
    end

    return evolve_state(ham_params, sim_params, problem.backend, problem.H_sys_bath, ψ_sb, te, sites; gates=interleaved_gates)
end

# --- Tensor Network + Density Matrix (unified for both evolution methods) ---

# Unified TN+DM prepare_combined_state for both Continuous and Trotter
function prepare_combined_state(problem::CoolingProblem{TNBackend}, state::QuantumState{TNBackend,DensityMatrix,E}) where E<:EvolutionMethod
    sites = problem.extra.sites
    coupling = problem.extra.coupling_params.coupling
    return appendzeros_MPO(state.state, sites, coupling)
end

function evolve_cooling_step(problem::CoolingProblem{TNBackend}, ρ_sb::MPO, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,ContinuousEvolution},
                           ham_params)
    error("Continuous evolution for density matrices (MPO) is not supported by TDVP in ITensors. Please use either:
    1. monte_carlo + continuous (uses MPS with TDVP)
    2. density_matrix + trotter (uses MPO with gates)
    3. monte_carlo + trotter (uses MPS with gates)")
end

function evolve_cooling_step(problem::CoolingProblem{TNBackend}, ρ_sb::MPO, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,TrotterEvolution},
                           ham_params)
    sites = problem.extra.sites

    # Get or create interleaved Trotter gates (all act on adjacent sites)
    interleaved_gates = get(problem.extra, :interleaved_gates, nothing)
    if interleaved_gates === nothing
        coupling_params = problem.extra.coupling_params
        interleaved_gates = build_trotter_circuit_interleaved(ham_params, problem.backend, sites, coupling_params, sim_params)
    end

    # Delegate to evolve_state (shared Trotter evolution logic)
    ρ_evolved = evolve_state(ham_params, sim_params, problem.backend, interleaved_gates, ρ_sb, te, sites)
    ρ_evolved /= tr(ρ_evolved)

    return ρ_evolved
end

# Unified TN+DM bath processing for both evolution methods
function process_bath_and_update(problem::CoolingProblem{TNBackend}, ρ_evolved::MPO,
                               state::QuantumState{TNBackend,DensityMatrix,E},
                               _sim_params) where E<:EvolutionMethod
    sites = problem.extra.sites
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]

    ρ_s = partial_trace_bath(ρ_evolved, sites, sites_sys)
    ρ_s = 0.5 * (ρ_s + dag(swapprime(ρ_s, 0, 1)))  # Enforce Hermiticity
    ρ_s /= tr(ρ_s)

    return QuantumState(state.backend, state.sim_method, state.evolution_method, ρ_s), nothing
end

# Unified TN+DM measurements for both Continuous and Trotter evolution
function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{TNBackend},
                                     state::QuantumState{TNBackend,DensityMatrix,E},
                                     _ham_params, _bath_info=nothing) where E<:EvolutionMethod
    ρ_s = state.state
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀

    measurements[RESULT_ENERGY][step] = real(inner(ρ_s, H_sys))
    measurements[RESULT_GROUND_STATE_OVERLAP][step] = real(inner(ρ_s, projector_mpo(ϕ₀)))
    measurements[RESULT_PURITY][step] = real(tr(apply(ρ_s, ρ_s)))
end
