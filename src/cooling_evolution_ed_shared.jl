"""
    cooling_evolution_ed_shared.jl

Shared functions for ED backend cooling evolution to follow DRY principles.
"""

# ============================================================================
# Shared ED Backend Functions
# ============================================================================

"""
    get_bath_ground_state_ed(N_bath::Int, coupling::String) -> EDStateVector

Create the ED bath ground state from the shared bath Hamiltonian convention.
"""
function get_bath_ground_state_ed(N_bath::Int, coupling::String)
    _, one_site = bath_ground_state_amplitudes(coupling)
    data = copy(one_site)
    for _ in 2:N_bath
        data = kron(data, one_site)
    end
    return EDStateVector(data, N_bath)
end

"""
    interleave_system_bath_ed(ψ_sys::EDStateVector, ψ_bath::EDStateVector) -> EDStateVector

Create interleaved system+bath state: [sys₁, bath₁, sys₂, bath₂, ...]
from sequential system and bath states.
"""
function interleave_system_bath_ed(ψ_sys::EDStateVector, ψ_bath::EDStateVector)
    N = ψ_sys.n_qubits
    @assert ψ_bath.n_qubits == N "System and bath must have same number of qubits"

    N_total = interleaved_total_sites(N)
    dim_total = 2^N_total
    data = zeros(ComplexF64, dim_total)

    for sys_idx in 0:(2^N - 1)
        for bath_idx in 0:(2^N - 1)
            combined_idx = interleaved_basis_state(sys_idx, bath_idx, N)
            data[combined_idx + 1] = ψ_sys.data[sys_idx + 1] * ψ_bath.data[bath_idx + 1]
        end
    end

    return EDStateVector(data, N_total)
end

"""
    prepare_combined_state_ed(state::EDStateVector, N_bath::Int, coupling::String="XX")

Prepare system+bath state vector for ED backend with INTERLEAVED layout.
Bath initialized in appropriate ground state based on coupling type.
Layout: [sys₁, bath₁, sys₂, bath₂, ...] matching Hamiltonian convention.
"""
function prepare_combined_state_ed(state::EDStateVector, N_bath::Int, coupling::String="XX")
    ψ_bath = get_bath_ground_state_ed(N_bath, coupling)
    return interleave_system_bath_ed(state, ψ_bath)
end

"""
    prepare_combined_state_ed(state::EDDensityMatrix, N_bath::Int, coupling::String="XX")

Prepare system+bath density matrix for ED backend with INTERLEAVED layout.
Bath initialized in appropriate ground state based on coupling type.
Layout: [sys₁, bath₁, sys₂, bath₂, ...] matching Hamiltonian convention.
"""
function prepare_combined_state_ed(state::EDDensityMatrix, N_bath::Int, coupling::String="XX")
    # For density matrix, create from pure state interleaving then trace if needed
    # Or directly interleave the density matrices

    N = state.n_qubits
    @assert N_bath == N "System and bath must have same number of qubits"

    ψ_bath = get_bath_ground_state_ed(N_bath, coupling)
    ρ_bath = state_to_density_ed(ψ_bath)

    N_total = interleaved_total_sites(N)
    dim_total = 2^N_total
    data = zeros(ComplexF64, dim_total, dim_total)

    # Interleave density matrices
    for sys_i in 0:(2^N - 1), sys_j in 0:(2^N - 1)
        for bath_i in 0:(2^N - 1), bath_j in 0:(2^N - 1)
            combined_i = interleaved_basis_state(sys_i, bath_i, N)
            combined_j = interleaved_basis_state(sys_j, bath_j, N)
            data[combined_i + 1, combined_j + 1] = state.data[sys_i + 1, sys_j + 1] * ρ_bath.data[bath_i + 1, bath_j + 1]
        end
    end

    return EDDensityMatrix(data, N_total)
end

"""
    evolve_cooling_step_ed(H::AbstractMatrix, state::Union{EDStateVector, EDDensityMatrix},
                          te::Float64, tau::Union{Float64, Nothing}=nothing)

Evolve ED states for both continuous (tau=nothing) and Trotter (tau specified) evolution.
Uses dispatch on state type via evolve_ed.
"""
function evolve_cooling_step_ed(H::AbstractMatrix, state::Union{EDStateVector, EDDensityMatrix},
                               te::Float64, tau::Union{Float64, Nothing}=nothing)
    tau === nothing && return evolve_ed(H, state, te)

    n_steps, dt = trotter_time_slices(te, tau)
    n_steps == 0 && return state
    evolved = state
    for _ in 1:n_steps
        evolved = evolve_ed(H, evolved, dt)
    end
    return evolved
end

"""
    process_bath_ed_monte_carlo(state::EDStateVector, N_sys::Int, N_bath::Int)

Shared function to measure and collapse bath for Monte Carlo methods.
Returns (system_state, bath_outcomes).
"""
function process_bath_ed_monte_carlo(state::EDStateVector, N_bath::Int)
    bath_qubits = interleaved_bath_sites(N_bath)
    
    # Measure bath qubits and collapse
    ψ_sys, bath_outcomes = measure_ed!(state, bath_qubits)
    
    return ψ_sys, bath_outcomes
end

"""
    _momentum_measurement_gF!(measurements, state, ham_params) -> Int

Return and cache the fermionic boundary sector used for ED momentum
diagnostics. If the current system state has definite ``P_x`` parity, the
momentum grid is fixed from that state on the first call. If the state has no
unique parity sector, the grid falls back to the same even-parity reference
sector used by mode diagnostics. Later calls reuse the cached grid so all
cooling steps share the same momentum axis.
"""
function _momentum_measurement_gF!(measurements,
                                   state::Union{EDStateVector, EDDensityMatrix},
                                   ham_params)
    if haskey(measurements, RESULT_MOMENTUM_GF)
        get!(measurements, RESULT_MOMENTUM_GF_SOURCE, "precomputed")
        return measurements[RESULT_MOMENTUM_GF]
    end

    N = ham_params.N
    px = measure_state_parity(state, N)
    sector = _reference_parity_sector_with_source(px)
    gF = fermionic_bc(ham_params.bc, sector.parity)
    measurements[RESULT_MOMENTUM_GF_SOURCE] = string(sector.source)

    measurements[RESULT_MOMENTUM_GF] = gF
    return gF
end

"""
    _system_state_for_measurement(state, N_sys)

Return the ED system state used for system observables. System-only states are
returned unchanged; an interleaved system-bath density matrix is reduced by
tracing out the bath.
"""
_system_state_for_measurement(state::EDStateVector, ::Int) = state

function _system_state_for_measurement(ρ::EDDensityMatrix, N_sys::Int)
    if ρ.n_qubits == interleaved_total_sites(N_sys)
        return trace_out_bath_ed(ρ, N_sys)
    elseif ρ.n_qubits == N_sys
        return ρ
    else
        throw(DimensionMismatch(
            "ED density-matrix measurements expected $N_sys or $(interleaved_total_sites(N_sys)) qubits, " *
            "got $(ρ.n_qubits)"
        ))
    end
end

"""
    _ensure_momentum_storage!(measurements, k_values, n_k)

Allocate ED momentum-distribution storage from the measured k-grid on the first
call, and assert on later calls that both the grid length and grid values are
unchanged.
"""
function _ensure_momentum_storage!(measurements, k_values, n_k)
    n_modes = length(k_values)
    length(n_k) == n_modes || throw(DimensionMismatch(
        "Momentum measurement returned $(length(n_k)) occupations for $n_modes k-values."
    ))

    if get(measurements, RESULT_MOMENTUM_DISTRIBUTION, nothing) === nothing ||
       get(measurements, RESULT_K_VALUES, nothing) === nothing
        n_steps_total = size(measurements[RESULT_ENERGY], 1)
        measurements[RESULT_MOMENTUM_DISTRIBUTION] = fill(NaN, n_steps_total, n_modes)
        measurements[RESULT_K_VALUES] = collect(k_values)
    end

    if length(measurements[RESULT_K_VALUES]) != n_modes
        throw(DimensionMismatch(
            "Stored k-grid has $(length(measurements[RESULT_K_VALUES])) entries, " *
            "but the momentum measurement returned $n_modes entries."
        ))
    end
    if measurements[RESULT_K_VALUES] != k_values
        throw(DimensionMismatch(
            "Stored k-grid values differ from the current momentum measurement."
        ))
    end
    if size(measurements[RESULT_MOMENTUM_DISTRIBUTION], 2) != n_modes
        throw(DimensionMismatch(
            "Stored momentum distribution has $(size(measurements[RESULT_MOMENTUM_DISTRIBUTION], 2)) mode columns, " *
            "but the momentum measurement returned $n_modes entries."
        ))
    end

    return nothing
end

"""
    perform_measurements_ed!(measurements, step::Int, problem::CoolingProblem{EDBackend},
                            state::Union{EDStateVector, EDDensityMatrix}, is_monte_carlo::Bool,
                            ham_params, bath_info=nothing)

Shared measurement function for ED backend.
"""
function perform_measurements_ed(measurements, step::Int, state::Union{EDStateVector, EDDensityMatrix},
                                H_sys_mat::AbstractMatrix, ϕ₀::EDStateVector,
                                ham_params, bath_info=nothing)
    N_sys = ham_params.N
    sys_state = _system_state_for_measurement(state, N_sys)
    
    if isa(sys_state, EDStateVector)
        # Monte Carlo: state is a wave function (system only)
        ψ_s = sys_state
        
        # Energy: <ψ|H|ψ>
        measurements[RESULT_ENERGY][step] = expect_ed(H_sys_mat, ψ_s)
        
        # Ground state overlap: |<ϕ₀|ψ>|²
        overlap = abs2(dot(ϕ₀.data, ψ_s.data))
        measurements[RESULT_GROUND_STATE_OVERLAP][step] = overlap
        
        # Purity is always 1 for pure states
        # No bath magnetization for system-only state
    else
        # Density matrix measurements are performed on the reduced system state.
        ρ_sys = sys_state
        
        # Energy
        measurements[RESULT_ENERGY][step] = expect_ed(H_sys_mat, ρ_sys)
        
        # Ground state overlap: <ϕ₀|ρ|ϕ₀>
        measurements[RESULT_GROUND_STATE_OVERLAP][step] = real(ϕ₀.data' * ρ_sys.data * ϕ₀.data)
        
        # Purity
        if haskey(measurements, RESULT_PURITY)
            measurements[RESULT_PURITY][step] = purity_ed(ρ_sys)
        end
        
        if haskey(measurements, RESULT_BATH_MAGNETIZATION) && bath_info !== nothing
            measurements[RESULT_BATH_MAGNETIZATION][step] = Float64(bath_info)
        end
    end
    
    # K-space measurements for ED with periodic/antiperiodic even Ising chains.
    if haskey(measurements, RESULT_MOMENTUM_DISTRIBUTION) && supports_ising_fourier_observables(ham_params)
        gF = _momentum_measurement_gF!(measurements, sys_state, ham_params)
        k_values, tilde_n_k = measure_raw_fourier_occupation_ed(sys_state, ham_params; gF=gF)
        _ensure_momentum_storage!(measurements, k_values, tilde_n_k)
        measurements[RESULT_MOMENTUM_DISTRIBUTION][step, :] .= tilde_n_k
    end

    # Bogoliubov mode-observable measurements ⟨h_k⟩
    # (requires measure_modes=true in run_cooling)
    if haskey(measurements, RESULT_MODE_HK) && supports_ising_fourier_observables(ham_params)
        _record_ising_mode_measurements!(measurements, step, sys_state, ham_params)
    end
end
