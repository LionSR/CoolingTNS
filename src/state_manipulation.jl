"""
    state_manipulation.jl

Dispatched functions for state manipulation operations like appending bath,
sampling, and tracing out subsystems.
"""

using ITensors
using ITensorMPS
using Yao
using Yao: measure!
using LinearAlgebra

# ============================================================================
# Append Bath States
# ============================================================================

"""
    append_bath(backend::CoolingBackend, system_state, bath_state)

Append fresh bath qubits to system state using dispatch.
"""
function append_bath(backend::CoolingBackend, system_state, bath_state)
    error("append_bath not implemented for backend=$(typeof(backend))")
end

# --- Tensor Network Backend ---
function append_bath(::TNBackend, ψ_s::MPS, sites::Vector{<:Index})
    # Use existing appendzeros_MPS function
    return appendzeros_MPS(ψ_s, sites)
end

function append_bath(::TNBackend, ρ_s::MPO, sites::Vector{<:Index})
    # Use existing appendzeros_MPO function
    return appendzeros_MPO(ρ_s, sites)
end

# --- ED Backend ---
function append_bath(::EDBackend, ψ_s::ArrayReg, N_bath::Int)
    # Fresh bath in ground state |000...⟩
    bath_config = 0  # All bits set to 0
    ψ_bath = product_state(N_bath, bath_config)
    return kron(ψ_s, ψ_bath)
end

function append_bath(::EDBackend, ρ_s::Matrix, N_bath::Int)
    # Fresh bath density matrix in ground state |000...⟩⟨000...|
    bath_config = 0  # All bits set to 0
    ψ_bath = product_state(N_bath, bath_config)
    ρ_bath = projector(ψ_bath)
    return kron(ρ_s, ρ_bath)
end

# ============================================================================
# Sample/Trace Out Bath
# ============================================================================

"""
    process_bath(backend::CoolingBackend, sim_method::SimulationMethod, combined_state, N_sys::Int, N_bath::Int)

Process bath degrees of freedom based on simulation method:
- Monte Carlo: Sample bath and collapse state
- Density Matrix: Trace out bath
Returns (system_state, bath_info) where bath_info could be samples or nothing.
"""
function process_bath(backend::CoolingBackend, sim_method::SimulationMethod, combined_state, N_sys::Int, N_bath::Int)
    error("process_bath not implemented for backend=$(typeof(backend)), sim_method=$(typeof(sim_method))")
end

# --- TN + Monte Carlo ---
function process_bath(::TNBackend, ::MonteCarloWavefunction, ψ_sb::MPS, N_sys::Int, N_bath::Int)
    # Use existing sample_bath function
    v_b, ψ_s = sample_bath(ψ_sb)
    return ψ_s, v_b
end

# --- TN + Density Matrix ---
function process_bath(::TNBackend, ::DensityMatrix, ρ_sb::MPO, N_sys::Int, N_bath::Int)
    # Use existing discardBath_MPO function
    ρ_s = discardBath_MPO(ρ_sb)
    ρ_s /= tr(ρ_s)  # Renormalize
    return ρ_s, nothing
end

# --- ED + Monte Carlo ---
function process_bath(::EDBackend, ::MonteCarloWavefunction, ψ_evolved::ArrayReg, N_sys::Int, N_bath::Int)
    # Bath qubits are at positions N_sys+1 to N_sys+N_bath
    bath_qubits = collect((N_sys+1):(N_sys+N_bath))
    
    # Measure bath qubits
    reg_measured = copy(ψ_evolved)
    # Use measure with RemoveMeasured() to get post-measurement state
    measured_results = measure!(RemoveMeasured(), reg_measured, bath_qubits)
    
    # After measurement, reg_measured contains only system qubits
    ψ_sys = reg_measured
    
    # Convert measurement results to integer array (0 or 1)
    # measured_results is a BitStr - convert to integer then to bit array
    measured_int = Int(measured_results)
    bath_samples = [Int((measured_int >> (i-1)) & 1) for i in 1:N_bath]
    
    return ψ_sys, bath_samples
end

# --- ED + Density Matrix ---
function process_bath(::EDBackend, ::DensityMatrix, ρ_total::Matrix, N_sys::Int, N_bath::Int)
    # Just return full state, tracing will be done during measurements
    # This avoids repeated tracing operations
    return ρ_total, nothing
end

# ============================================================================
# Trace Operations
# ============================================================================

"""
    trace_out_bath(backend::CoolingBackend, combined_state, N_sys::Int, N_bath::Int)

Trace out bath degrees of freedom to get system reduced density matrix.
"""
function trace_out_bath(backend::CoolingBackend, combined_state, N_sys::Int, N_bath::Int)
    error("trace_out_bath not implemented for backend=$(typeof(backend))")
end

# --- ED Backend ---
function trace_out_bath(::EDBackend, ρ_total::Matrix, N_sys::Int, N_bath::Int)
    dim_sys = 2^N_sys
    dim_bath = 2^N_bath
    ρ_sys = zeros(ComplexF64, dim_sys, dim_sys)
    
    for i in 1:dim_sys, j in 1:dim_sys
        for k in 1:dim_bath
            idx_i = (i-1)*dim_bath + k
            idx_j = (j-1)*dim_bath + k
            ρ_sys[i,j] += ρ_total[idx_i, idx_j]
        end
    end
    
    return ρ_sys
end

# --- TN Backend ---
function trace_out_bath(::TNBackend, ρ::MPO, N_sys::Int, N_bath::Int)
    # Use existing function
    return discardBath_MPO(ρ)
end

# ============================================================================
# Get Sites Information
# ============================================================================

"""
    get_sites(problem::CoolingProblem, backend::CoolingBackend)

Get site indices from problem, handling different backends.
"""
function get_sites(problem::CoolingProblem, backend::CoolingBackend)
    error("get_sites not implemented for backend=$(typeof(backend))")
end

function get_sites(problem::CoolingProblem, ::TNBackend)
    # For TN backend, sites are stored in extra
    return problem.extra.sites
end

function get_sites(problem::CoolingProblem, ::EDBackend)
    # ED backend doesn't use sites
    return nothing
end

# ============================================================================
# Utility Functions
# ============================================================================

"""Create projector from wavefunction for ED backend"""
function projector(ψ::ArrayReg)
    vec_ψ = ψ.state[:]
    return vec_ψ * vec_ψ'
end

"""Create MPO projector from MPS for TN backend"""
function projector_mpo(ψ::MPS)
    return outer(ψ', ψ)
end