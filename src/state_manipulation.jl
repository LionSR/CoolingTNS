"""
    state_manipulation.jl

Dispatched functions for state manipulation operations like appending bath,
sampling, and tracing out subsystems.
"""

using ITensors
using ITensorMPS
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
# Clean ED backend implementation
function append_bath(::EDBackend, ψ_s::EDStateVector, N_bath::Int)
    # Fresh bath in ground state |000...⟩
    ψ_bath = zero_state_ed(N_bath)
    return kron_states_ed(ψ_s, ψ_bath)
end

function append_bath(::EDBackend, ρ_s::EDDensityMatrix, N_bath::Int)
    # Fresh bath density matrix in ground state |000...⟩⟨000...|
    ρ_bath = state_to_density_ed(zero_state_ed(N_bath))
    return kron_density_ed(ρ_s, ρ_bath)
end



function append_bath(::EDBackend, ρ_s::Matrix, N_bath::Int)
    # Convert to EDDensityMatrix
    N_sys = Int(log2(size(ρ_s, 1)))
    ρ_s_ed = EDDensityMatrix(ρ_s, N_sys)
    ρ_combined = append_bath(EDBackend(), ρ_s_ed, N_bath)
    return ρ_combined.data  # Return as Matrix for backward compatibility
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
# Clean ED backend implementation
function process_bath(::EDBackend, ::MonteCarloWavefunction, ψ_evolved::EDStateVector, N_sys::Int, N_bath::Int)
    # For alternating layout: bath qubits are at even positions 2, 4, 6, ...
    bath_qubits = collect(2:2:2*N_bath)
    
    # Measure bath qubits and get collapsed state
    ψ_sys, bath_samples = measure_ed!(ψ_evolved, bath_qubits)
    
    return ψ_sys, bath_samples
end

# --- ED + Density Matrix ---
function process_bath(::EDBackend, ::DensityMatrix, ρ_total::EDDensityMatrix, N_sys::Int, N_bath::Int)
    # Just return full state, tracing will be done during measurements
    # This avoids repeated tracing operations
    return ρ_total, nothing
end

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
function trace_out_bath(::EDBackend, ρ_total::EDDensityMatrix, N_sys::Int, N_bath::Int)
    return trace_out_bath_ed(ρ_total, N_sys)
end

# Matrix support for backward compatibility
function trace_out_bath(::EDBackend, ρ_total::Matrix, N_sys::Int, N_bath::Int)
    # For alternating layout, need to trace out even-indexed qubits
    # Convert to EDDensityMatrix first
    N_total = Int(log2(size(ρ_total, 1)))
    ρ_ed = EDDensityMatrix(real(ρ_total), N_total)
    ρ_sys = trace_out_bath_ed(ρ_ed, N_sys)
    return ρ_sys.data
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

# Clean ED backend projector
"""Create projector from EDStateVector"""
function projector(ψ::EDStateVector)
    return state_to_density_ed(ψ).data
end


"""Create MPO projector from MPS for TN backend"""
function projector_mpo(ψ::MPS)
    return outer(ψ', ψ)
end