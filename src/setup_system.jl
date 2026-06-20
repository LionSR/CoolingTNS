"""
    setup_system.jl

Dispatch-based system setup for different Hamiltonian models and backends.
"""

using ITensors
using ITensorMPS
using KrylovKit
using LinearAlgebra
using SparseArrays

"""
    setup_system(ham_params::HamiltonianParameters, backend::CoolingBackend)

Generic interface for system setup using dispatch on HamiltonianParameters and backend.
"""
function setup_system(ham_params::HamiltonianParameters, backend::CoolingBackend)
    error("setup_system not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

# Removed duplicate method - use the one at line 52 instead

# ============================================================================
# Tensor Network Backend System Setup
# ============================================================================

"""
    setup_system(ham_params::HamiltonianParameters, backend::TNBackend, sites)

Setup system for tensor network backends using ITensors and DMRG.
"""
function setup_system(ham_params::HamiltonianParameters, backend::TNBackend, sites::Vector{<:Index})
    # Build system Hamiltonian using dispatch
    H_sys = construct_system_hamiltonian(ham_params, backend, sites)
    
    # Find ground state and gap using dispatch
    e₀, ϕ₀, gap = find_ground_state(H_sys, backend, sites)
    # Delta is positive. Downstream system-bath setup chooses the bath Pauli
    # with get_bath_operator(coupling), and the prepared bath state is its
    # eigenvalue -1 state.
    Δ_dmrg = gap

    return H_sys, Δ_dmrg, e₀, ϕ₀
end

# For backward compatibility with TN-only calls that pass sites instead of backend
function setup_system(ham_params::HamiltonianParameters{M}, sites::Vector{<:Index}) where M<:HamiltonianModel
    return setup_system(ham_params, TNBackend(), sites)
end

# ============================================================================
# Exact Diagonalization Backend System Setup  
# ============================================================================

"""
    setup_system(ham_params::HamiltonianParameters, backend::EDBackend)

Setup system for exact diagonalization backend using dense matrices.
"""
function setup_system(ham_params::HamiltonianParameters, backend::EDBackend)
    # Build system Hamiltonian using dispatch
    H_sys = construct_system_hamiltonian(ham_params, backend, ham_params.N)
    
    # Find ground state and gap using dispatch
    e₀, ϕ₀, gap = find_ground_state(H_sys, backend)
    # Delta is positive. Downstream system-bath setup chooses the bath Pauli
    # with get_bath_operator(coupling), and the prepared bath state is its
    # eigenvalue -1 state.
    Δ_ed = gap

    return H_sys, Δ_ed, e₀, ϕ₀
end

# Ground state computation now handled by ground_state.jl
