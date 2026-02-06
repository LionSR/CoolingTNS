"""
    cooling_evolution_ed_shared.jl

Shared functions for ED backend cooling evolution to follow DRY principles.
"""

# ============================================================================
# Shared ED Backend Functions
# ============================================================================

"""
    prepare_combined_state_ed(state::EDStateVector, N_bath::Int)

Prepare system+bath state vector for ED backend.
"""
function prepare_combined_state_ed(state::EDStateVector, N_bath::Int)
    ψ_bath = zero_state_ed(N_bath)
    return kron_states_ed(state, ψ_bath)
end

"""
    prepare_combined_state_ed(state::EDDensityMatrix, N_bath::Int)

Prepare system+bath density matrix for ED backend.
"""
function prepare_combined_state_ed(state::EDDensityMatrix, N_bath::Int)
    ρ_bath = state_to_density_ed(zero_state_ed(N_bath))
    return kron_density_ed(state, ρ_bath)
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
    perform_measurements_ed!(measurements, step::Int, problem::CoolingProblem{EDBackend},
                            state::Union{EDStateVector, EDDensityMatrix}, is_monte_carlo::Bool,
                            ham_params, bath_info=nothing)

Shared measurement function for ED backend.
"""
function perform_measurements_ed(measurements, step::Int, state::Union{EDStateVector, EDDensityMatrix},
                                H_sys_mat::AbstractMatrix, ϕ₀::EDStateVector,
                                ham_params, _bath_info=nothing)
    N_sys = ham_params.N
    
    if isa(state, EDStateVector)
        # Monte Carlo: state is a wave function (system only)
        ψ_s = state
        
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
        
        # Get system density matrix
        if ρ_total.n_qubits == 2*N_sys
            # Full system+bath state - trace out bath
            ρ_sys = trace_out_bath_ed(ρ_total, N_sys)
        else
            # Just system state
            ρ_sys = ρ_total
        end
        
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
    if haskey(measurements, "momentum_dist") && ham_params.bc in [:periodic, :antiperiodic] && isa(ham_params.model, IsingModel)
        if isa(state, EDStateVector)
            # For pure states
            k_values, n_k = measure_momentum_distribution_ed(state, ham_params)
            if step == 1
                measurements["k_values"][:] = k_values
            end
            measurements["momentum_dist"][step, :] .= n_k
        else
            # For density matrices, we need to get the system state
            if ρ_total.n_qubits == 2*N_sys
                ρ_sys = trace_out_bath_ed(ρ_total, N_sys)
            else
                ρ_sys = ρ_total
            end
            
            # Measure momentum distribution from density matrix
            k_values, n_k = measure_momentum_distribution_ed(ρ_sys, ham_params)
            if step == 1
                measurements["k_values"][:] = k_values
            end
            measurements["momentum_dist"][step, :] .= n_k
        end
    end
end