"""
    system_hamiltonian_dispatch.jl

System-only Hamiltonian construction using multiple dispatch on HamiltonianModel and backend.
"""

using ITensors
using Yao
# parameter_types.jl already included by parent

# ============================================================================
# System Hamiltonian Construction Interface
# ============================================================================

"""
    construct_system_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites_or_N)

Generic interface for constructing system Hamiltonians using multiple dispatch on both model and backend.
"""
function construct_system_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites_or_N)
    error("construct_system_hamiltonian not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

# ============================================================================
# Tensor Network (ITensors) Implementations
# ============================================================================

function construct_system_hamiltonian(ham_params::HamiltonianParameters{IsingModel}, backend::TNBackend, sites::Vector{<:Index})
    J, h = ham_params.params.J, ham_params.params.h
    N = ham_params.N
    
    terms = OpSum()
    for i in 1:N-1
        terms += J, "Z", i, "Z", i+1
    end
    for i in 1:N
        terms += h, "X", i  
    end
    
    return MPO(terms, sites)
end

function construct_system_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, backend::TNBackend, sites::Vector{<:Index})
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    N = ham_params.N
    
    terms = OpSum()
    for i in 1:N-1
        terms += J, "Z", i, "Z", i+1
    end
    for i in 1:N
        terms += hx, "X", i
        terms += hz, "Z", i
    end
    
    return MPO(terms, sites)
end

function construct_system_hamiltonian(ham_params::HamiltonianParameters{RydbergModel}, backend::TNBackend, sites::Vector{<:Index})
    Ω, Δ, V = ham_params.params.Ω, ham_params.params.Δ, ham_params.params.V
    N = ham_params.N
    
    terms = OpSum()
    # Rabi coupling: Ω/2 * (σ^+ + σ^-)  
    for i in 1:N
        terms += Ω/2, "S+", i
        terms += Ω/2, "S-", i
        terms += -Δ, "ProjUp", i  # Detuning term
    end
    
    # Van der Waals interaction: V/r^6
    for i in 1:N-1, j in i+1:N
        r_ij = abs(j - i)
        terms += V/r_ij^6, "ProjUp", i, "ProjUp", j
    end
    
    return MPO(terms, sites)
end

# ============================================================================
# Exact Diagonalization (Dense Matrix) Implementations
# ============================================================================

function construct_system_hamiltonian(ham_params::HamiltonianParameters{IsingModel}, backend::EDBackend, ::Int)
    J, h = ham_params.params.J, ham_params.params.h
    N = ham_params.N
    
    # Transverse field Ising model using Yao.jl
    H_sys = sum([
        map(i -> J * put(N, (i, i+1)=>kron(Z, Z)), 1:N-1)...,
        map(i -> h * put(N, i=>X), 1:N)...
    ])
    
    return H_sys
end

function construct_system_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, backend::EDBackend, ::Int)
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    N = ham_params.N
    
    # Non-integrable Ising model using Yao.jl
    H_sys = sum([
        # ZZ interactions
        map(i -> J * put(N, (i, i+1)=>kron(Z, Z)), 1:N-1)...,
        # X field
        map(i -> hx * put(N, i=>X), 1:N)...,
        # Z field  
        map(i -> hz * put(N, i=>Z), 1:N)...
    ])
    
    return H_sys
end