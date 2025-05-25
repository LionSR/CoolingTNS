"""
    ground_state_dispatch.jl

Ground state computation using multiple dispatch on backend type.
"""

using ITensors
using KrylovKit
using LinearAlgebra
using Yao
include("parameter_types.jl")

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
function find_ground_state(H_sys, backend::TNBackend, sites)
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
    H_sys_mat = mat(H_sys)
    
    # Find ground state
    vals, vecs, info = eigsolve(H_sys_mat, 1, :SR; krylovdim=min(30, size(H_sys_mat, 1)))
    e₀ = real(vals[1])
    ϕ₀_vec = vecs[1]
    ϕ₀ = ArrayReg(normalize!(Complex.(ϕ₀_vec)))
    
    # Find first excited state for gap computation
    vals2, _, _ = eigsolve(H_sys_mat, 2, :SR; krylovdim=min(30, size(H_sys_mat, 1)))
    e₁ = real(vals2[2])
    gap = e₁ - e₀
    
    return e₀, ϕ₀, gap
end

# ============================================================================
# Convenience Functions (Backward Compatibility)
# ============================================================================

"""
    find_ground_state_dmrg(H_sys, sites)

Legacy wrapper for TN backend ground state computation.
"""
function find_ground_state_dmrg(H_sys, sites)
    return find_ground_state(H_sys, TNBackend(), sites)
end

"""
    find_ground_state_ed(H_sys)

Legacy wrapper for ED backend ground state computation.
"""
function find_ground_state_ed(H_sys)
    return find_ground_state(H_sys, EDBackend())
end