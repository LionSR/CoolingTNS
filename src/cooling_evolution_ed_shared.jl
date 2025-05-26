"""
    cooling_evolution_ed_shared.jl

Shared functions for ED backend cooling evolution to follow DRY principles.
"""

# ============================================================================
# Shared ED Backend Functions
# ============================================================================

"""
    prepare_combined_state_ed(state::Union{EDStateVector, EDDensityMatrix}, N_bath::Int)

Shared function to prepare system+bath state for ED backend.
Works for both state vectors and density matrices.
"""
function prepare_combined_state_ed(state::Union{EDStateVector, EDDensityMatrix}, N_bath::Int)
    if isa(state, EDStateVector)
        # For state vector: append bath in ground state |00...0⟩
        ψ_bath = zero_state_ed(N_bath)
        return kron_states_ed(state, ψ_bath)
    else
        # For density matrix: append bath in ground state
        ρ_bath = state_to_density_ed(zero_state_ed(N_bath))
        return kron_density_ed(state, ρ_bath)
    end
end

"""
    evolve_cooling_step_ed(H::AbstractMatrix, state::Union{EDStateVector, EDDensityMatrix}, 
                          te::Float64, tau::Union{Float64, Nothing}=nothing)

Shared function to evolve ED states for both continuous and Trotter evolution.
"""
function evolve_cooling_step_ed(H::AbstractMatrix, state::Union{EDStateVector, EDDensityMatrix}, 
                               te::Float64, tau::Union{Float64, Nothing}=nothing)
    if tau === nothing
        # Continuous evolution - single step
        return evolve_ed(H, state, te)
    else
        # Trotter evolution - multiple small steps
        n_steps = Int(ceil(te / tau))
        dt = te / n_steps  # Adjust dt to exactly cover te
        
        evolved = state
        for _ in 1:n_steps
            evolved = evolve_ed(H, evolved, dt)
        end
        return evolved
    end
end

"""
    process_bath_ed_monte_carlo(state::EDStateVector, N_sys::Int, N_bath::Int)

Shared function to measure and collapse bath for Monte Carlo methods.
Returns (system_state, bath_outcomes).
"""
function process_bath_ed_monte_carlo(state::EDStateVector, N_sys::Int, N_bath::Int)
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
function perform_measurements_ed!(measurements, step::Int, problem::CoolingProblem{EDBackend},
                                 state::Union{EDStateVector, EDDensityMatrix}, is_monte_carlo::Bool,
                                 ham_params, bath_info=nothing)
    H_sys_mat = problem.H_sys
    ϕ₀ = problem.ϕ₀
    N_sys = ham_params.N
    
    if is_monte_carlo
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
end