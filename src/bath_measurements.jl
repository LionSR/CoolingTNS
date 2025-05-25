"""
    bath_measurements.jl

Dispatched functions for computing bath-related measurements.
"""

using ITensors
using ITensorMPS
using LinearAlgebra

# ============================================================================
# Bath Magnetization Measurement Dispatch
# ============================================================================

"""
    compute_bath_magnetization(backend::CoolingBackend, state::QuantumState, evolved_state, N_bath::Int)

Compute bath magnetization using dispatch on backend and simulation method.
"""
function compute_bath_magnetization(backend::CoolingBackend, state::QuantumState, evolved_state, N_bath::Int)
    error("compute_bath_magnetization not implemented for backend=$(typeof(backend)), sim_method=$(typeof(state.sim_method))")
end

# --- Tensor Network + Monte Carlo ---
# For MPS Monte Carlo, bath magnetization comes from the sampled bath configuration
function compute_bath_magnetization(::TNBackend, ::QuantumState{TNBackend,MonteCarloWavefunction,E}, 
                                  bath_sample::Vector{Int}, N_bath::Int) where E
    # bath_sample contains 0s and 1s
    return 2 * sum(bath_sample) / N_bath - 1.0
end

# --- ED + Monte Carlo ---  
# For ED Monte Carlo, bath magnetization comes from collapsed measurement
function compute_bath_magnetization(::EDBackend, ::QuantumState{EDBackend,MonteCarloWavefunction,E},
                                  bath_result::Vector{Int}, N_bath::Int) where E
    # bath_result contains measurement outcomes (0 or 1)
    return sum(2 .* bath_result .- 1) / N_bath
end

# --- ED + Density Matrix ---
# For density matrix, compute expectation value from bath reduced density matrix
function compute_bath_magnetization(::EDBackend, ::QuantumState{EDBackend,DensityMatrix,E},
                                  ρ_bath::Matrix, N_bath::Int) where E
    mag = 0.0
    dim = 2^N_bath
    
    for i in 1:dim
        # Count number of 1s in binary representation
        n_ones = count_ones(i-1)
        mag += real(ρ_bath[i,i]) * (2*n_ones/N_bath - 1)
    end
    
    return mag
end

# --- TN + Density Matrix (MPO) ---
# For MPO, we need to compute expectation values differently
function compute_bath_magnetization(::TNBackend, ::QuantumState{TNBackend,DensityMatrix,E},
                                  ρ_bath::MPO, sites_bath::Vector{<:Index}) where E
    # Compute average magnetization of bath sites
    N_bath = length(sites_bath)
    total_mag = 0.0
    
    for i in 1:N_bath
        # Create Sz operator for site i
        sz_op = MPO(sites_bath, [i => "Sz"])
        # Compute expectation value
        mag_i = real(inner(ρ_bath, sz_op))
        total_mag += mag_i
    end
    
    return total_mag / N_bath
end

# ============================================================================
# Bath State Extraction Dispatch
# ============================================================================

"""
    extract_bath_state(backend::CoolingBackend, state::QuantumState, evolved_state, N_sys::Int, N_bath::Int)

Extract bath state or measurement using dispatch.
"""
function extract_bath_state(backend::CoolingBackend, state::QuantumState, evolved_state, N_sys::Int, N_bath::Int)
    error("extract_bath_state not implemented for backend=$(typeof(backend)), sim_method=$(typeof(state.sim_method))")
end

# --- ED + Density Matrix ---
# Trace out system to get bath density matrix
function extract_bath_state(::EDBackend, ::QuantumState{EDBackend,DensityMatrix,E},
                          ρ_total::Matrix, N_sys::Int, N_bath::Int) where E
    return tr_sys(ρ_total, N_sys, N_bath)
end

# --- ED + Monte Carlo ---
# Already handled during measurement collapse - return the measurement result
function extract_bath_state(::EDBackend, ::QuantumState{EDBackend,MonteCarloWavefunction,E},
                          bath_result::Vector{Int}, N_sys::Int, N_bath::Int) where E
    return bath_result
end

# --- TN + Monte Carlo ---
# Already handled during bath sampling - return the sample
function extract_bath_state(::TNBackend, ::QuantumState{TNBackend,MonteCarloWavefunction,E},
                          bath_sample::Vector{Int}, N_sys::Int, N_bath::Int) where E
    return bath_sample
end

# ============================================================================
# Utility Functions
# ============================================================================

"""Trace out system degrees of freedom to get bath density matrix"""
function tr_sys(ρ::Matrix, N_sys::Int, N_bath::Int)
    dim_sys = 2^N_sys
    dim_bath = 2^N_bath
    ρ_bath = zeros(ComplexF64, dim_bath, dim_bath)
    
    for i in 1:dim_bath, j in 1:dim_bath
        for k in 1:dim_sys
            idx_i = (k-1)*dim_bath + i
            idx_j = (k-1)*dim_bath + j
            ρ_bath[i,j] += ρ[idx_i, idx_j]
        end
    end
    
    return ρ_bath
end