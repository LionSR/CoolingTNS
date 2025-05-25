"""
    cooling_evolution_ed_clean.jl

ED backend cooling evolution implementations without Yao dependencies.
"""

include("ed_backend.jl")

# ============================================================================
# ED Backend Specific Implementations for Cooling Evolution
# ============================================================================

# --- Exact Diagonalization + Density Matrix + Continuous Evolution ---

function prepare_combined_state(problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend,DensityMatrix,ContinuousEvolution})
    # Get N from problem parameters 
    N_sys = problem.extra.ham_params.N
    N_bath = N_sys
    
    # Get system density matrix by tracing out previous bath (if any)
    if isa(state.state, EDDensityMatrix) && state.state.n_qubits == 2*N_sys
        # We have a full system+bath state, trace out bath
        ρ_sys = trace_out_bath_ed(state.state, N_sys)
    elseif isa(state.state, Matrix)
        # Legacy matrix format
        if size(state.state, 1) == 2^(2*N_sys)
            ρ_sys_data = tr_bath(state.state, N_sys, N_bath)
            ρ_sys = EDDensityMatrix(real(ρ_sys_data), N_sys)
        else
            ρ_sys = EDDensityMatrix(state.state, N_sys)
        end
    else
        # Already system-only EDDensityMatrix
        ρ_sys = state.state
    end
    
    # Fresh bath in ground state |000...⟩
    ρ_bath = state_to_density_ed(zero_state_ed(N_bath))
    
    return kron_density_ed(ρ_sys, ρ_bath)
end

function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ρ_total::EDDensityMatrix, te::Float64,
                           sim_params::UnifiedSimulationParameters{DensityMatrix,ContinuousEvolution},
                           ham_params)
    # Get sparse Hamiltonian matrix
    H_sparse = problem.H_sys_bath
    if !isa(H_sparse, AbstractMatrix)
        error("Expected sparse matrix Hamiltonian for ED backend")
    end
    
    # Time evolution
    return evolve_ed(H_sparse, ρ_total, te)
end

function apply_noise(ρ::EDDensityMatrix, problem::CoolingProblem{EDBackend}, pe::Float64)
    return apply_depolarizing_ed(ρ, pe)
end

function process_bath_and_update(problem::CoolingProblem{EDBackend}, ρ_evolved::EDDensityMatrix,
                               state::QuantumState{EDBackend,DensityMatrix,ContinuousEvolution},
                               sim_params)
    # For density matrix, we keep the full state and trace out bath during measurements
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ρ_evolved), nothing
end

function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{EDBackend},
                                     state::QuantumState{EDBackend,DensityMatrix,ContinuousEvolution},
                                     ham_params, bath_info=nothing)
    ρ_total = state.state
    N_sys = ham_params.N
    N_bath = N_sys
    
    # Get system density matrix
    if ρ_total.n_qubits == 2*N_sys
        # Full system+bath state - trace out bath
        ρ_sys = trace_out_bath_ed(ρ_total, N_sys)
    else
        # Just system state
        ρ_sys = ρ_total
    end
    
    # Energy
    H_sys_mat = problem.H_sys
    measurements["E_list"][step] = expect_ed(H_sys_mat, ρ_sys)
    
    # Ground state overlap: <ϕ₀|ρ|ϕ₀>
    ϕ₀ = problem.ϕ₀
    measurements["GS_overlap_list"][step] = real(ϕ₀.data' * ρ_sys.data * ϕ₀.data)
    
    # Purity
    measurements["purity_list"][step] = purity_ed(ρ_sys)
    
    # Bath magnetization (only if we have full state and not first step)
    if step > 1 && ρ_total.n_qubits == 2*N_sys
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

# --- Exact Diagonalization + Monte Carlo Wavefunction ---

function prepare_combined_state(problem::CoolingProblem{EDBackend}, state::QuantumState{EDBackend,MonteCarloWavefunction,ContinuousEvolution})
    # Get N from ham_params stored in problem.extra
    N_sys = problem.extra.ham_params.N
    N_bath = N_sys
    
    # Fresh bath in ground state |000...⟩
    ψ_bath = zero_state_ed(N_bath)
    
    # Handle both EDStateVector and legacy formats
    if isa(state.state, EDStateVector)
        return kron_states_ed(state.state, ψ_bath)
    else
        # Legacy: convert to EDStateVector first
        ψ_sys = EDStateVector(vec(state.state), N_sys)
        return kron_states_ed(ψ_sys, ψ_bath)
    end
end

function evolve_cooling_step(problem::CoolingProblem{EDBackend}, ψ_total::EDStateVector, te::Float64,
                           sim_params::UnifiedSimulationParameters{MonteCarloWavefunction,ContinuousEvolution},
                           ham_params)
    # Time evolution 
    H = problem.H_sys_bath
    return evolve_ed(H, ψ_total, te)
end

function apply_noise(ψ::EDStateVector, problem::CoolingProblem{EDBackend}, pe::Float64)
    # Apply local depolarizing noise to system qubits
    N_sys = problem.extra.ham_params.N
    N_total = ψ.n_qubits
    
    # System qubits in alternating layout: 1, 3, 5, ...
    sys_qubits = collect(1:2:2*N_sys-1)
    
    return apply_depolarizing_ed(ψ, pe, sys_qubits)
end

function process_bath_and_update(problem::CoolingProblem{EDBackend}, ψ_evolved::EDStateVector,
                               state::QuantumState{EDBackend,MonteCarloWavefunction,ContinuousEvolution},
                               sim_params)
    # Get N from ham_params stored in problem.extra
    N_sys = problem.extra.ham_params.N
    N_bath = N_sys
    
    # Bath qubits are at even positions: 2, 4, 6, ...
    bath_qubits = collect(2:2:2*N_bath)
    
    # Measure bath qubits
    ψ_sys, bath_samples = measure_ed!(ψ_evolved, bath_qubits)
    
    # Calculate bath magnetization
    bath_mag = 2 * sum(bath_samples) / N_bath - 1.0
    
    return QuantumState(state.backend, state.sim_method, state.evolution_method, ψ_sys), bath_mag
end

function perform_backend_measurements!(measurements, step::Int, problem::CoolingProblem{EDBackend},
                                     state::QuantumState{EDBackend,MonteCarloWavefunction,ContinuousEvolution},
                                     ham_params, bath_info=nothing)
    ψ_sys = state.state
    H_sys = problem.H_sys
    ϕ₀ = problem.ϕ₀
    
    measurements["E_list"][step] = expect_ed(H_sys, ψ_sys)
    measurements["GS_overlap_list"][step] = overlap_ed(ψ_sys, ϕ₀)
    
    # Bath magnetization from measurement
    if bath_info !== nothing && haskey(measurements, "bath_mag_list")
        measurements["bath_mag_list"][step] = bath_info
    end
end

# ============================================================================
# Utility Functions (keep for backward compatibility)
# ============================================================================

"""Trace out bath degrees of freedom (legacy matrix format)"""
function tr_bath(ρ::Matrix, N_sys::Int, N_bath::Int)
    dim_sys = 2^N_sys
    dim_bath = 2^N_bath
    ρ_sys = zeros(Float64, dim_sys, dim_sys)
    
    for i in 1:dim_sys, j in 1:dim_sys
        for k in 1:dim_bath
            idx_i = (i-1)*dim_bath + k
            idx_j = (j-1)*dim_bath + k
            ρ_sys[i,j] += real(ρ[idx_i, idx_j])
        end
    end
    
    return ρ_sys
end

"""Trace out system degrees of freedom (legacy matrix format)"""
function tr_sys(ρ::Matrix, N_sys::Int, N_bath::Int)
    dim_sys = 2^N_sys
    dim_bath = 2^N_bath
    ρ_bath = zeros(Float64, dim_bath, dim_bath)
    
    for i in 1:dim_bath, j in 1:dim_bath
        for k in 1:dim_sys
            idx_i = (k-1)*dim_bath + i
            idx_j = (k-1)*dim_bath + j
            ρ_bath[i,j] += real(ρ[idx_i, idx_j])
        end
    end
    
    return ρ_bath
end

"""Compute bath magnetization from density matrix (legacy format)"""
function compute_bath_magnetization(ρ_bath::Matrix, N_bath::Int)
    mag = 0.0
    dim = 2^N_bath
    
    for i in 1:dim
        # Count number of 1s in binary representation
        n_ones = count_ones(i-1)
        mag += real(ρ_bath[i,i]) * (2*n_ones/N_bath - 1)
    end
    
    return mag
end