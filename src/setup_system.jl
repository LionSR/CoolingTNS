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
    
    # Find ground state and gap using dispatch.
    # The returned gap is the (positive) bath detuning Δ. Downstream system-bath
    # setup chooses the bath Pauli with get_bath_operator(coupling), and the
    # prepared bath state is its eigenvalue -1 state.
    e₀, ϕ₀, gap = find_ground_state(H_sys, backend, sites)

    return H_sys, gap, e₀, ϕ₀
end

# ============================================================================
# Exact Diagonalization Backend System Setup  
# ============================================================================

"""
    setup_system(ham_params::HamiltonianParameters, backend::EDBackend)

Setup system for exact diagonalization backend using sparse Pauli matrices.
"""
function setup_system(ham_params::HamiltonianParameters, backend::EDBackend)
    # Build system Hamiltonian using dispatch
    H_sys = construct_system_hamiltonian(ham_params, backend, ham_params.N)

    # Find ground state and gap using dispatch.
    # The returned gap is the (positive) bath detuning Δ. Downstream system-bath
    # setup chooses the bath Pauli with get_bath_operator(coupling), and the
    # prepared bath state is its eigenvalue -1 state.
    e₀, ϕ₀, gap = find_ground_state(H_sys, backend)

    return H_sys, gap, e₀, ϕ₀
end
