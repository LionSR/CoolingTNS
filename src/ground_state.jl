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

This is the low-budget end of the same DMRG protocol as
[`compute_excitation_gaps`](@ref)`(ham_params, ::TNBackend)`: one excited state
at `maxdim = 100`, restarted from the *same* `random_mps` draw `ψ₀` used for the
ground state, just to set the single-Δ default detuning.
`compute_excitation_gaps` runs the higher-budget variant -- `maxdim_excited = 200`
and a second `random_mps` draw for the excited search, then warm-starting each
level from the previous eigenvector -- because it builds an `R`-level ladder for
multi-frequency cooling, where an error in level `n` mis-places a whole bath
detuning. Both divergences are deliberate, not drift; note the differing
`random_mps` draw counts mean the two routines also consume the RNG differently.
"""
function find_ground_state(H_sys::MPO, backend::TNBackend, sites::Vector{<:Index})
    # Find ground state using DMRG
    ψ₀ = random_mps(sites, linkdims=10)
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
    find_ground_state(H_sys::AbstractMatrix, backend::EDBackend)

Find ground state and energy gap using exact diagonalization for ED backend.
Handles both sparse and dense matrices.
"""
function find_ground_state(H_sys::AbstractMatrix, ::EDBackend)
    return ground_state_ed(H_sys)
end
