"""
    setup_system.jl

Dispatch-based system setup for different Hamiltonian models and backends.
"""

using ITensors
using ITensorMPS
using KrylovKit
using LinearAlgebra
using Yao
# parameter_types.jl already included by parent
# The core files are included by parent

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
    Δ_dmrg = -gap  # Resonant cooling
    
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
    Δ_ed = -gap  # Resonant cooling
    
    return H_sys, Δ_ed, e₀, ϕ₀
end

# Ground state computation now handled by ground_state.jl

# ============================================================================
# Backend-Agnostic Interface (Auto-detects from parameters)
# ============================================================================

"""
    setup_system_auto(ham_params::HamiltonianParameters; backend=nothing, sites=nothing)

Automatically choose between ED and TN backends based on system size and availability.
"""
function setup_system_auto(ham_params::HamiltonianParameters; backend=nothing, sites=nothing)
    if backend !== nothing
        if backend isa EDBackend
            return setup_system(ham_params, backend)
        elseif backend isa TNBackend && sites !== nothing
            return setup_system(ham_params, backend, sites)
        else
            error("TNBackend requires sites to be provided")
        end
    else
        # Auto-detect: use ED for small systems, TN for larger ones
        if ham_params.N <= 10 && sites === nothing
            return setup_system(ham_params, EDBackend())
        elseif sites !== nothing
            return setup_system(ham_params, TNBackend(), sites)
        else
            error("For N > 12, sites must be provided for tensor network backend")
        end
    end
end