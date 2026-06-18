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

If `step_observer` is supplied, it is called once after the initial measurement
with `stage=:initial`, once after preparing the system-bath state with
`stage=:prepared`, once after each system-bath evolution with `stage=:evolved`,
and once after each updated-system measurement with `stage=:updated`. The
callback receives a named tuple containing `stage`, `step`, `state`,
`evolved_state`, `measurements`, `delta`, `te`, and `bath_info`. This is
intended for diagnostics that need transient data, such as system-bath bond
dimensions, without duplicating the cooling loop.

The optional `evolution_kwargs` named tuple is forwarded to the low-level
evolution routine for each cooling step. It is intended for diagnostics such as
TDVP observers or output levels; physical parameters should remain in
`sim_params` and `mf_params`.
"""
function run_cooling_multi_freq(
    problem::CoolingProblem{B},
    state::QuantumState{B,S,E},
    mf_params::MultiFrequencyCouplingParameters,
    sim_params,
    ham_params;
    measure_modes::Bool=false,
    step_observer=nothing,
    evolution_kwargs=(;),
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
    notify_step_observer(
        step_observer,
        (stage=:initial, step=1, state=state, evolved_state=nothing, measurements=measurements,
         delta=NaN, te=NaN, bath_info=nothing),
    )
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
        notify_step_observer(
            step_observer,
            (stage=:prepared, step=step, state=state, evolved_state=combined_state,
             measurements=measurements, delta=delta_r, te=te_step, bath_info=nothing),
        )

        # Evolve with the step-dependent coupling parameters
        evolved_state = if te_step <= 0
            combined_state
        else
            evolve_cooling_step_dynamic(
                problem, combined_state, coupling_step, te_step, sim_params, ham_params;
                evolution_kwargs...,
            )
        end
        notify_step_observer(
            step_observer,
            (stage=:evolved, step=step, state=state, evolved_state=evolved_state,
             measurements=measurements, delta=delta_r, te=te_step, bath_info=nothing),
        )

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
        notify_step_observer(
            step_observer,
            (stage=:updated, step=step, state=state, evolved_state=nothing, measurements=measurements,
             delta=delta_r, te=te_step, bath_info=bath_info),
        )

        # Print progress
        if step % 10 == 0 || step == steps + 1
            print_cooling_status(step, measurements, ham_params, state)
        end
    end

    println("Cooling completed")
    return compile_results(measurements, sim_params)
end

notify_step_observer(::Nothing, _) = nothing

function notify_step_observer(step_observer, info)
    step_observer(info)
    return nothing
end

# Dispatch hook: allow run_cooling(..., mf_params::MultiFrequencyCouplingParameters, ...) to work.
function run_cooling(
    problem::CoolingProblem{B},
    state::QuantumState{B,S,E},
    coupling_params::MultiFrequencyCouplingParameters,
    sim_params,
    ham_params;
    measure_modes::Bool=false,
    step_observer=nothing,
    evolution_kwargs=(;),
) where {B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    return run_cooling_multi_freq(
        problem,
        state,
        coupling_params,
        sim_params,
        ham_params;
        measure_modes=measure_modes,
        step_observer=step_observer,
        evolution_kwargs=evolution_kwargs,
    )
end

# Internal helper: pick which Δ index to use at this step.
function _pick_delta_index(step::Int, R::Int, schedule::Symbol)
    schedule == :round_robin && return mod1(step - 1, R)
    schedule == :random && return rand(1:R)
    throw(ArgumentError("Unknown multi-frequency schedule=$schedule (expected :round_robin or :random)"))
end

# Evolve with step-dependent coupling parameters (e.g. changing bath detuning Δ).
evolve_cooling_step_dynamic(
    ::CoolingProblem, state, ::BasicCouplingParameters, te_step, sim_params, ham_params;
    kwargs...,
) =
    error("evolve_cooling_step_dynamic not implemented for this backend/method combination")

function _interleaved_step_gates_builder(
    problem::CoolingProblem{TNBackend},
    ham_params,
    sim_params::UnifiedSimulationParameters,
    sites,
    coupling_params::CouplingParameters,
    cache_key_prefix=nothing,
)
    cache = get(problem.extra, :trotter_step_gates_cache, nothing)
    return function (dt::Float64)
        key = cache_key_prefix === nothing ? dt : (cache_key_prefix, dt)
        if cache isa AbstractDict
            return get!(cache, key) do
                build_trotter_circuit_interleaved(
                    ham_params,
                    problem.backend,
                    sites,
                    coupling_params,
                    with_trotter_tau(sim_params, dt),
                )
            end
        end
        return build_trotter_circuit_interleaved(
            ham_params,
            problem.backend,
            sites,
            coupling_params,
            with_trotter_tau(sim_params, dt),
        )
    end
end

function evolve_cooling_step_dynamic(
    problem::CoolingProblem{TNBackend},
    ψ_sb::MPS,
    coupling_step::BasicCouplingParameters,
    te_step::Float64,
    sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, ContinuousEvolution},
    ham_params,
    ;
    kwargs...,
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

    return evolve_state(ham_params, sim_params, problem.backend, H_step, ψ_sb, te_step, sites; kwargs...)
end

function evolve_cooling_step_dynamic(
    problem::CoolingProblem{TNBackend},
    ψ_sb::MPS,
    coupling_step::BasicCouplingParameters,
    te_step::Float64,
    sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, TrotterEvolution},
    ham_params,
    ;
    kwargs...,
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
    step_gates = _interleaved_step_gates_builder(problem, ham_params, sim_params, sites, coupling_step, δ)

    return evolve_state(ham_params, sim_params, problem.backend, gates, ψ_sb, te_step, sites;
                        gates=gates, step_gates=step_gates, kwargs...)
end

function evolve_cooling_step_dynamic(
    problem::CoolingProblem{TNBackend},
    ρ_sb::MPO,
    coupling_step::BasicCouplingParameters,
    te_step::Float64,
    sim_params::UnifiedSimulationParameters{DensityMatrix, TrotterEvolution},
    ham_params,
    ;
    kwargs...,
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
    step_gates = _interleaved_step_gates_builder(problem, ham_params, sim_params, sites, coupling_step, δ)

    ρ_evolved = evolve_state(ham_params, sim_params, problem.backend, gates, ρ_sb, te_step, sites;
                             step_gates=step_gates, kwargs...)
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
    ;
    kwargs...,
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
    
    # Add mode energy measurements if requested for states whose backend has a
    # convention-matched Ising Fourier observable implementation.
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
        if supports_ising_fourier_observables(ham_params)
            # The step-1 measurement materializes these placeholders before
            # results are compiled or saved.
            measurements[RESULT_MOMENTUM_DISTRIBUTION] = nothing
            measurements[RESULT_K_VALUES] = nothing
        end
    end
end

function _measurement_ham_params(problem::CoolingProblem, ham_params)
    ham_params !== nothing && return ham_params
    return haskey(problem.extra, :ham_params) ? problem.extra.ham_params : nothing
end

function _add_ising_mode_measurement_slots!(measurements, ::CoolingProblem,
                                            state::QuantumState, ham_params)
    supports_ising_fourier_observables(ham_params) || return false

    # Use the state parity when it selects a definite fermionic sector.  If the
    # state is mixed between parity sectors, fall back to the even-parity
    # reference grid and record that provenance.  This avoids using the
    # approximate TN DMRG ground state, whose parity need not be exact at finite
    # bond dimension.
    px = measure_state_parity(state.state, ham_params.N)
    sector = _reference_parity_sector_with_source(px)
    measurements[RESULT_MODE_GF] = fermionic_bc(ham_params.bc, sector.parity)
    measurements[RESULT_MODE_GF_SOURCE] = string(sector.source)

    measurements[RESULT_MODE_HK] = nothing
    measurements[RESULT_MODE_NK] = nothing
    measurements[RESULT_MODE_K_INDICES] = nothing
    measurements[RESULT_MODE_ENERGIES] = nothing
    return true
end

function _record_ising_mode_measurements!(measurements, step::Int, state, ham_params)
    haskey(measurements, RESULT_MODE_HK) || return false
    supports_ising_fourier_observables(ham_params) || return false

    gF_kwarg = haskey(measurements, RESULT_MODE_GF) ? measurements[RESULT_MODE_GF] : nothing
    k_indices, hk_values, εk_values = measure_all_mode_energies(state, ham_params; gF=gF_kwarg)
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
    return true
end

function _copy_previous_ising_mode_measurements!(measurements, step::Int)
    step > 1 || return false
    hk = get(measurements, RESULT_MODE_HK, nothing)
    nk = get(measurements, RESULT_MODE_NK, nothing)
    (hk isa AbstractMatrix && nk isa AbstractMatrix) || return false
    hk[step, :] .= hk[step - 1, :]
    nk[step, :] .= nk[step - 1, :]
    return true
end

# Helper for mode energy measurements ⟨h_k⟩ (Ising PBC/APBC)
function add_mode_measurements!(measurements, problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend}, steps, ham_params)
    _add_ising_mode_measurement_slots!(
        measurements,
        problem,
        state,
        _measurement_ham_params(problem, ham_params),
    )
end

function add_mode_measurements!(measurements, problem::CoolingProblem{TNBackend},
                                state::QuantumState{TNBackend}, steps, ham_params)
    _add_ising_mode_measurement_slots!(
        measurements,
        problem,
        state,
        _measurement_ham_params(problem, ham_params),
    )
end

# Fallback: mode measurements not supported for this backend/state pair.
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

    system_state_is_measurable = step == 1 || length(ψ_s) == length(H_sys)
    if system_state_is_measurable
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

    if system_state_is_measurable
        _record_ising_mode_measurements!(measurements, step, ψ_s, ham_params)
    else
        _copy_previous_ising_mode_measurements!(measurements, step)
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
    # Canonical cooling noise: independent local Pauli depolarization on every
    # system-bath qubit, matching the ED and TN Monte Carlo wavefunction paths.
    return apply_depolarizing_ed(ρ, pe, 1:ρ.n_qubits)
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
    coupling_params = problem.extra.coupling_params

    # Use interleaved gates (same circuit as DM+Trotter for consistency)
    interleaved_gates = get(problem.extra, :interleaved_gates, nothing)
    if interleaved_gates === nothing
        interleaved_gates = build_trotter_circuit_interleaved(ham_params, problem.backend, sites, coupling_params, sim_params)
    end
    step_gates = _interleaved_step_gates_builder(problem, ham_params, sim_params, sites, coupling_params)

    return evolve_state(ham_params, sim_params, problem.backend, problem.H_sys_bath, ψ_sb, te, sites;
                        gates=interleaved_gates, step_gates=step_gates)
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
    coupling_params = problem.extra.coupling_params

    # Get or create interleaved Trotter gates (all act on adjacent sites)
    interleaved_gates = get(problem.extra, :interleaved_gates, nothing)
    if interleaved_gates === nothing
        interleaved_gates = build_trotter_circuit_interleaved(ham_params, problem.backend, sites, coupling_params, sim_params)
    end
    step_gates = _interleaved_step_gates_builder(problem, ham_params, sim_params, sites, coupling_params)

    # Delegate to evolve_state (shared Trotter evolution logic)
    ρ_evolved = evolve_state(ham_params, sim_params, problem.backend, interleaved_gates, ρ_sb, te, sites;
                             step_gates=step_gates)
    ρ_evolved /= tr(ρ_evolved)

    return ρ_evolved
end

# Unified TN+DM bath processing for both evolution methods
function process_bath_and_update(problem::CoolingProblem{TNBackend}, ρ_evolved::MPO,
                               state::QuantumState{TNBackend,DensityMatrix,E},
                               _sim_params) where E<:EvolutionMethod
    sites = problem.extra.sites
    N = length(sites) ÷ 2
    sites_sys = interleaved_system_indices(sites, N)
    sites_bath = interleaved_bath_indices(sites, N)

    ρ_b = partial_trace_system(ρ_evolved, sites, sites_bath)
    ρ_b /= tr(ρ_b)
    bath_mag = compute_bath_magnetization(problem.backend, state, ρ_b, sites_bath)

    ρ_s = partial_trace_bath(ρ_evolved, sites, sites_sys)
    ρ_s = 0.5 * (ρ_s + dag(swapprime(ρ_s, 0, 1)))  # Enforce Hermiticity
    ρ_s /= tr(ρ_s)

    return QuantumState(state.backend, state.sim_method, state.evolution_method, ρ_s), bath_mag
end

# Unified TN+DM measurements for both Continuous and Trotter evolution
function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{TNBackend},
                                     state::QuantumState{TNBackend,DensityMatrix,E},
                                     ham_params, bath_info=nothing) where E<:EvolutionMethod
    ρ_s = state.state
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀

    measurements[RESULT_ENERGY][step] = real(inner(ρ_s, H_sys))
    measurements[RESULT_GROUND_STATE_OVERLAP][step] = real(inner(ρ_s, projector_mpo(ϕ₀)))
    measurements[RESULT_PURITY][step] = real(tr(apply(ρ_s, ρ_s)))
    if haskey(measurements, RESULT_BATH_MAGNETIZATION) && bath_info !== nothing
        measurements[RESULT_BATH_MAGNETIZATION][step] = bath_info
    end

    _record_ising_mode_measurements!(measurements, step, ρ_s, ham_params)
end
