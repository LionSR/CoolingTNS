"""
    cooling_evolution_ed_shared.jl

Shared functions for ED backend cooling evolution to follow DRY principles.
"""

# ============================================================================
# Shared ED Backend Functions
# ============================================================================

function _supports_ising_fourier_observables(ham_params)
    return ham_params !== nothing &&
           iseven(ham_params.N) &&
           ham_params.bc in [:periodic, :antiperiodic] &&
           isa(ham_params.model, IsingModel)
end

function _reference_parity_sector(px::Real; atol=0.1, default::Int=1)
    @assert default == 1 || default == -1 "default must be +1 or -1"
    abs(px - 1) <= atol && return 1
    abs(px + 1) <= atol && return -1
    return default
end

function _reference_fermionic_bc(spin_bc::Symbol, px::Real; atol=0.1, default_parity::Int=1)
    return fermionic_bc(
        spin_bc,
        _reference_parity_sector(px; atol=atol, default=default_parity),
    )
end

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

    N_total = 2 * N
    dim_total = 2^N_total
    data = zeros(ComplexF64, dim_total)

    # For each basis state in the combined space
    for sys_idx in 0:(2^N - 1)
        for bath_idx in 0:(2^N - 1)
            # Interleave the bits: sys₁ bath₁ sys₂ bath₂ ...
            combined_idx = 0
            for i in 0:(N-1)
                sys_bit = (sys_idx >> i) & 1
                bath_bit = (bath_idx >> i) & 1
                combined_idx |= (sys_bit << (2*i))      # System at odd positions (0, 2, 4...)
                combined_idx |= (bath_bit << (2*i + 1)) # Bath at even positions (1, 3, 5...)
            end
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

    N_total = 2 * N
    dim_total = 2^N_total
    data = zeros(ComplexF64, dim_total, dim_total)

    # Interleave density matrices
    for sys_i in 0:(2^N - 1), sys_j in 0:(2^N - 1)
        for bath_i in 0:(2^N - 1), bath_j in 0:(2^N - 1)
            # Interleave indices
            combined_i = 0
            combined_j = 0
            for k in 0:(N-1)
                combined_i |= ((sys_i >> k) & 1) << (2*k)
                combined_i |= ((bath_i >> k) & 1) << (2*k + 1)
                combined_j |= ((sys_j >> k) & 1) << (2*k)
                combined_j |= ((bath_j >> k) & 1) << (2*k + 1)
            end
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

    n_steps = Int(ceil(te / tau))
    dt = te / n_steps
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
    # Bath qubits are at even positions in alternating layout
    bath_qubits = [2*i for i in 1:N_bath]
    
    # Measure bath qubits and collapse
    ψ_sys, bath_outcomes = measure_ed!(state, bath_qubits)
    
    return ψ_sys, bath_outcomes
end

"""
    _momentum_measurement_gF!(measurements, state, ϕ₀, ham_params) -> Int

Return and cache the fermionic boundary sector used for ED momentum
diagnostics. If the current system state has definite ``P_x`` parity, the
momentum grid is fixed from that state on the first call. If the state has no
unique parity sector, the grid falls back to the ground-state sector, matching
the mode-diagnostic reference convention. Later calls reuse the cached grid so
all cooling steps share the same momentum axis.
"""
function _momentum_measurement_gF!(measurements, state::Union{EDStateVector, EDDensityMatrix},
                                   ϕ₀::EDStateVector, ham_params)
    if haskey(measurements, "momentum_gF")
        get!(measurements, "momentum_gF_source", "precomputed")
        return measurements["momentum_gF"]
    end

    N = ham_params.N
    px = measure_state_parity(state, N)
    parity = round(Int, px)

    if abs(px - parity) <= 0.1 && abs(parity) == 1
        gF = fermionic_bc(ham_params.bc, parity)
        measurements["momentum_gF_source"] = "state"
    else
        # A mixed-parity state has no unique fermionic boundary condition.
        # Use the ground-state sector as a fixed reference grid for diagnostics,
        # following the convention used for mode-resolved h_k measurements.
        px0 = measure_state_parity(ϕ₀, N)
        gF = _reference_fermionic_bc(ham_params.bc, px0)
        measurements["momentum_gF_source"] = "ground_state"
    end

    measurements["momentum_gF"] = gF
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
    if ρ.n_qubits == 2 * N_sys
        return trace_out_bath_ed(ρ, N_sys)
    elseif ρ.n_qubits == N_sys
        return ρ
    else
        throw(DimensionMismatch(
            "ED density-matrix measurements expected $N_sys or $(2 * N_sys) qubits, " *
            "got $(ρ.n_qubits)"
        ))
    end
end

"""
    perform_measurements_ed!(measurements, step::Int, problem::CoolingProblem{EDBackend},
                            state::Union{EDStateVector, EDDensityMatrix}, is_monte_carlo::Bool,
                            ham_params, bath_info=nothing)

Shared measurement function for ED backend.
"""
function perform_measurements_ed(measurements, step::Int, state::Union{EDStateVector, EDDensityMatrix},
                                H_sys_mat::AbstractMatrix, ϕ₀::EDStateVector,
                                ham_params, _bath_info=nothing)
    N_sys = ham_params.N
    sys_state = _system_state_for_measurement(state, N_sys)
    
    if isa(sys_state, EDStateVector)
        # Monte Carlo: state is a wave function (system only)
        ψ_s = sys_state
        
        # Energy: <ψ|H|ψ>
        measurements["E_list"][step] = expect_ed(H_sys_mat, ψ_s)
        
        # Ground state overlap: |<ϕ₀|ψ>|²
        overlap = abs2(dot(ϕ₀.data, ψ_s.data))
        measurements["GS_overlap_list"][step] = overlap
        
        # Purity is always 1 for pure states
        # No bath magnetization for system-only state
    else
        # Density matrix: may be full system+bath or system only
        ρ_total = state
        ρ_sys = sys_state
        
        # Energy
        measurements["E_list"][step] = expect_ed(H_sys_mat, ρ_sys)
        
        # Ground state overlap: <ϕ₀|ρ|ϕ₀>
        measurements["GS_overlap_list"][step] = real(ϕ₀.data' * ρ_sys.data * ϕ₀.data)
        
        # Purity
        if haskey(measurements, "purity_list")
            measurements["purity_list"][step] = purity_ed(ρ_sys)
        end
        
        # Bath magnetization (only if we have full state and not first step)
        if step > 1 && ρ_total.n_qubits == 2*N_sys && haskey(measurements, "bath_mag_list")
            N_bath = N_sys
            ρ_bath = trace_out_system_ed(ρ_total, N_sys)
            # Compute magnetization
            mag = 0.0
            for i in 1:N_bath
                Z_i = pauli_z(i, N_bath)
                mag += expect_ed(Z_i, ρ_bath)
            end
            measurements["bath_mag_list"][step] = mag / N_bath
        end
    end
    
    # K-space measurements for ED with periodic/antiperiodic BC (only for Ising model)
    if haskey(measurements, "momentum_dist") && _supports_ising_fourier_observables(ham_params)
        gF = _momentum_measurement_gF!(measurements, sys_state, ϕ₀, ham_params)
        k_values, n_k = measure_momentum_distribution_ed_clean(sys_state, ham_params; gF=gF)
        if step == 1
            measurements["k_values"][:] = k_values
        end
        measurements["momentum_dist"][step, :] .= n_k
    end

    # Mode energy measurements ⟨h_k⟩ (requires measure_modes=true in run_cooling)
    if haskey(measurements, "mode_hk") && _supports_ising_fourier_observables(ham_params)
        # Use the stored ground-state gF for consistent sector choice.
        # For pure states, measure_all_mode_energies auto-detects gF from parity.
        # For density matrices (mixed states), parity may not be ±1, so we use
        # the gF determined from the ground state's parity sector.
        gF_kwarg = if haskey(measurements, "mode_gF")
            measurements["mode_gF"]
        else
            nothing
        end

        k_indices, hk_values, εk_values = measure_all_mode_energies(sys_state, ham_params; gF=gF_kwarg)
        n_modes = length(k_indices)

        # Allocate arrays on first call (now we know the number of modes)
        if measurements["mode_hk"] === nothing
            n_steps_total = size(measurements["E_list"], 1)
            measurements["mode_hk"] = fill(NaN, n_steps_total, n_modes)
            measurements["mode_k_indices"] = k_indices
            measurements["mode_ek_values"] = εk_values
        end

        measurements["mode_hk"][step, :] .= hk_values
    end
end
