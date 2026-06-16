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

function _pauli_z_from_tn_sample(sample::Int)
    sample == 1 && return 1.0
    sample == 2 && return -1.0
    throw(ArgumentError("TN bath samples are ITensor site indices 1 or 2, got $sample"))
end

function _pauli_z_from_ed_bit(bit::Int)
    bit == 0 && return 1.0
    bit == 1 && return -1.0
    throw(ArgumentError("ED bath measurement bits must be 0 or 1, got $bit"))
end

function _single_site_mpo(sites::Vector{<:Index}, op_name::String, site::Int)
    terms = OpSum()
    terms += 1.0, op_name, site
    return MPO(terms, sites)
end

# --- Tensor Network + Monte Carlo ---
# For MPS Monte Carlo, bath magnetization comes from the sampled bath configuration
function compute_bath_magnetization(::TNBackend, ::QuantumState{TNBackend,MonteCarloWavefunction,E}, 
                                  bath_sample::Vector{Int}, N_bath::Int) where E
    # `sample_bath` returns ITensor site indices: 1 = Up (Z=+1), 2 = Dn (Z=-1).
    return sum(_pauli_z_from_tn_sample.(bath_sample)) / N_bath
end

# --- ED + Monte Carlo ---  
# For ED Monte Carlo, bath magnetization comes from collapsed measurement
function compute_bath_magnetization(::EDBackend, ::QuantumState{EDBackend,MonteCarloWavefunction,E},
                                  bath_result::Vector{Int}, N_bath::Int) where E
    # `measure_ed!` returns computational bits: 0 = Up (Z=+1), 1 = Dn (Z=-1).
    return sum(_pauli_z_from_ed_bit.(bath_result)) / N_bath
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
        mag += real(ρ_bath[i,i]) * (1 - 2*n_ones/N_bath)
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
        z_op = _single_site_mpo(sites_bath, "Z", i)
        # Compute expectation value
        mag_i = real(inner(ρ_bath, z_op))
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
    N_sys == N_bath || throw(ArgumentError(
        "ED bath extraction expects the interleaved one-bath-per-system layout; " *
        "got N_sys=$N_sys and N_bath=$N_bath."
    ))
    ρ_total = EDDensityMatrix(Matrix{ComplexF64}(ρ), N_sys + N_bath)
    return trace_out_system_ed(ρ_total, N_sys).data
end
