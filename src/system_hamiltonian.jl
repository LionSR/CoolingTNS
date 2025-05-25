"""
    system_hamiltonian.jl

System-only Hamiltonian construction using multiple dispatch on HamiltonianModel and backend.
"""

using ITensors
using LinearAlgebra
using SparseArrays

# ============================================================================
# System Hamiltonian Construction Interface
# ============================================================================

# Generic fallback method for construct_system_hamiltonian
# Specific implementations have their own docstrings
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
    
    # Transverse field Ising model using clean ED backend
    H_sys = spzeros(Float64, 2^N, 2^N)
    
    # ZZ interactions
    for i in 1:N-1
        H_sys += J * pauli_zz(i, i+1, N)
    end
    
    # X field
    for i in 1:N
        H_sys += h * pauli_x(i, N)
    end
    
    return H_sys
end

function construct_system_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, backend::EDBackend, ::Int)
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    N = ham_params.N
    
    # Non-integrable Ising model using clean ED backend
    H_sys = spzeros(Float64, 2^N, 2^N)
    
    # ZZ interactions
    for i in 1:N-1
        H_sys += J * pauli_zz(i, i+1, N)
    end
    
    # X field
    for i in 1:N
        H_sys += hx * pauli_x(i, N)
    end
    
    # Z field  
    for i in 1:N
        H_sys += hz * pauli_z(i, N)
    end
    
    return H_sys
end

function construct_system_hamiltonian(ham_params::HamiltonianParameters{RydbergModel}, backend::EDBackend, ::Int)
    Ω, Δ, V = ham_params.params.Ω, ham_params.params.Δ, ham_params.params.V
    N = ham_params.N
    
    # Rydberg model using clean ED backend
    H_sys = spzeros(Float64, 2^N, 2^N)
    
    # Rabi coupling: Ω * X
    for i in 1:N
        H_sys += Ω * pauli_x(i, N)
    end
    
    # Detuning: -Δ * (I + Z)/2 = -Δ/2 * I - Δ/2 * Z
    # We only include the Z part since constant energy shifts don't matter
    for i in 1:N
        H_sys += -Δ/2 * pauli_z(i, N)
    end
    
    # Van der Waals interaction: V/r^6 * n_i * n_j where n = (I + Z)/2
    # This becomes V/4r^6 * (I + Z_i + Z_j + Z_i*Z_j)
    # Again, we only keep the non-constant terms
    for i in 1:N-1
        for j in i+1:N
            r_ij = abs(j - i)
            V_ij = V / r_ij^6
            # Z_i*Z_j term
            H_sys += V_ij/4 * pauli_zz(i, j, N)
            # Single Z terms
            H_sys += V_ij/4 * pauli_z(i, N)
            H_sys += V_ij/4 * pauli_z(j, N)
        end
    end
    
    return H_sys
end