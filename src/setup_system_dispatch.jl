"""
    setup_system_dispatch.jl

Dispatch-based system setup for different Hamiltonian models and backends.
"""

using ITensors
using KrylovKit
using LinearAlgebra
using Yao
include("parameter_types.jl")
include("system_hamiltonian_dispatch.jl")
include("ground_state_dispatch.jl")

"""
    setup_system(N, ham_params::HamiltonianParameters, backend::CoolingBackend)

Generic interface for system setup using dispatch on HamiltonianParameters and backend.
"""
function setup_system(N, ham_params::HamiltonianParameters, backend::CoolingBackend)
    error("setup_system not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

# For backward compatibility with TN-only calls that pass sites instead of backend
function setup_system(N, ham_params::HamiltonianParameters, sites::Vector)
    return setup_system(N, ham_params, TNBackend(), sites)
end

# ============================================================================
# Tensor Network Backend System Setup
# ============================================================================

"""
    setup_system(N, ham_params::HamiltonianParameters, backend::TNBackend, sites)

Setup system for tensor network backends using ITensors and DMRG.
"""
function setup_system(N, ham_params::HamiltonianParameters, backend::TNBackend, sites::Vector)
    # Build system Hamiltonian using dispatch
    H_sys = construct_system_hamiltonian(ham_params, backend, sites)
    
    # Find ground state and gap using dispatch
    e₀, ϕ₀, gap = find_ground_state(H_sys, backend, sites)
    Δ_dmrg = -gap  # Resonant cooling
    
    return H_sys, Δ_dmrg, e₀, ϕ₀
end

# Backward compatibility - sites passed directly
function setup_system(N, ham_params::HamiltonianParameters{M}, sites::Vector) where M<:HamiltonianModel
    return setup_system(N, ham_params, TNBackend(), sites)
end

# ============================================================================
# Exact Diagonalization Backend System Setup  
# ============================================================================

"""
    setup_system(N, ham_params::HamiltonianParameters, backend::EDBackend)

Setup system for exact diagonalization backend using dense matrices.
"""
function setup_system(N, ham_params::HamiltonianParameters, backend::EDBackend)
    # Build system Hamiltonian using dispatch
    H_sys = construct_system_hamiltonian(ham_params, backend, N)
    
    # Find ground state and gap using dispatch
    e₀, ϕ₀, gap = find_ground_state(H_sys, backend)
    Δ_ed = -gap  # Resonant cooling
    
    return H_sys, Δ_ed, e₀, ϕ₀
end

# Ground state computation now handled by ground_state_dispatch.jl

# ============================================================================
# Backend-Agnostic Interface (Auto-detects from parameters)
# ============================================================================

"""
    setup_system_auto(N, ham_params::HamiltonianParameters; backend=nothing, sites=nothing)

Automatically choose between ED and TN backends based on system size and availability.
"""
function setup_system_auto(N, ham_params::HamiltonianParameters; backend=nothing, sites=nothing)
    if backend !== nothing
        if backend isa EDBackend
            return setup_system(N, ham_params, backend)
        elseif backend isa TNBackend && sites !== nothing
            return setup_system(N, ham_params, backend, sites)
        else
            error("TNBackend requires sites to be provided")
        end
    else
        # Auto-detect: use ED for small systems, TN for larger ones
        if N <= 12 && sites === nothing
            return setup_system(N, ham_params, EDBackend())
        elseif sites !== nothing
            return setup_system(N, ham_params, TNBackend(), sites)
        else
            error("For N > 12, sites must be provided for tensor network backend")
        end
    end
end