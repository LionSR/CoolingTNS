"""
    ground_state.jl

Ground state computation using multiple dispatch on backend type.
"""

using ITensors
using ITensorMPS
using KrylovKit
using LinearAlgebra
using SparseArrays


# ============================================================================
# Ground State Computation Interface
# ============================================================================

"""
    find_ground_state(H_sys, backend::CoolingBackend, additional_args...)

Generic interface for ground state computation using dispatch on backend type.
"""
function find_ground_state(H_sys, backend::CoolingBackend, additional_args...)
    error("find_ground_state not implemented for backend $(typeof(backend))")
end

# ============================================================================
# Tensor Network Ground State (DMRG)
# ============================================================================

"""
    find_ground_state(H_sys, backend::TNBackend, sites)

Find ground state and energy gap using DMRG for tensor network backends.
"""
function find_ground_state(H_sys::MPO, backend::TNBackend, sites::Vector{<:Index})
    # Find ground state using DMRG
    ψ₀ = randomMPS(sites, linkdims=10)
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, 10, 20, 100, 100, 200)
    setcutoff!(sweeps, 1E-10)
    
    e₀, ϕ₀ = dmrg(H_sys, ψ₀, sweeps; outputlevel=0)
    
    # Compute gap for resonant cooling
    excited_sweeps = Sweeps(3)
    setmaxdim!(excited_sweeps, 100)
    setcutoff!(excited_sweeps, 1E-10)
    
    e₁, _ = dmrg(H_sys, [ϕ₀], ψ₀, excited_sweeps; outputlevel=0, weight=20.0)
    gap = e₁ - e₀
    
    return e₀, ϕ₀, gap
end

# ============================================================================
# Exact Diagonalization Ground State
# ============================================================================

"""
    find_ground_state(H_sys, backend::EDBackend)

Find ground state and energy gap using exact diagonalization for ED backend.
"""
function find_ground_state(H_sys, backend::EDBackend)
    # Use our clean ED backend function
    return ground_state_ed(H_sys)
end

"""
    find_ground_state(H_sys::SparseMatrixCSC, backend::EDBackend)

Find ground state energy, state, and gap for ED backend with sparse matrix.
"""
function find_ground_state(H_sys::SparseMatrixCSC, backend::EDBackend)
    E0, ψ0, gap = ground_state_ed(H_sys)
    return E0, ψ0, gap
end

"""
    find_ground_state(H_sys::Matrix, backend::EDBackend)

Find ground state for dense matrix (convert to sparse first).
"""
function find_ground_state(H_sys::Matrix, backend::EDBackend)
    return find_ground_state(sparse(H_sys), backend)
end