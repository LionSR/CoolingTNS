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

function construct_system_hamiltonian(ham_params::HamiltonianParameters{IsingModel}, ::TNBackend, sites::Vector{<:Index})
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

function construct_system_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, ::TNBackend, sites::Vector{<:Index})
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

function construct_system_hamiltonian(ham_params::HamiltonianParameters{RydbergModel}, ::TNBackend, sites::Vector{<:Index})
    Ω, Δ, V = ham_params.params.Ω, ham_params.params.Δ, ham_params.params.V
    N = ham_params.N

    terms = OpSum()
    for i in 1:N
        terms += Ω/2, "S+", i
        terms += Ω/2, "S-", i
        terms += -Δ, "ProjUp", i
    end
    for i in 1:N-1, j in i+1:N
        terms += V / (j - i)^6, "ProjUp", i, "ProjUp", j
    end
    return MPO(terms, sites)
end

# ============================================================================
# Exact Diagonalization (Dense Matrix) Implementations
# ============================================================================

"""Add nearest-neighbor ZZ interactions with boundary condition handling."""
function add_zz_chain_ed!(H::SparseMatrixCSC, J::Float64, N::Int, bc::Symbol)
    for i in 1:N-1
        H .+= J * pauli_zz(i, i+1, N)
    end
    bc == :periodic && (H .+= J * pauli_zz(N, 1, N))
    bc == :antiperiodic && (H .-= J * pauli_zz(N, 1, N))
    return H
end

function construct_system_hamiltonian(ham_params::HamiltonianParameters{IsingModel}, ::EDBackend, ::Int)
    J, h = ham_params.params.J, ham_params.params.h
    N, bc = ham_params.N, ham_params.bc

    H_sys = spzeros(Float64, 2^N, 2^N)
    add_zz_chain_ed!(H_sys, J, N, bc)

    for i in 1:N
        H_sys .+= h * pauli_x(i, N)
    end
    return H_sys
end

function construct_system_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, ::EDBackend, ::Int)
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    N, bc = ham_params.N, ham_params.bc

    H_sys = spzeros(Float64, 2^N, 2^N)
    add_zz_chain_ed!(H_sys, J, N, bc)

    for i in 1:N
        H_sys .+= hx * pauli_x(i, N)
        H_sys .+= hz * pauli_z(i, N)
    end
    return H_sys
end

function construct_system_hamiltonian(ham_params::HamiltonianParameters{RydbergModel}, ::EDBackend, ::Int)
    Ω, Δ, V = ham_params.params.Ω, ham_params.params.Δ, ham_params.params.V
    N = ham_params.N

    H_sys = spzeros(Float64, 2^N, 2^N)

    # Single-site terms: Rabi coupling (Ω*X) and detuning (-Δ/2*Z)
    for i in 1:N
        H_sys += Ω * pauli_x(i, N) - (Δ/2) * pauli_z(i, N)
    end

    # Van der Waals interaction: V/r^6 * n_i * n_j where n = (I + Z)/2
    # Expands to V/4r^6 * (Z_i*Z_j + Z_i + Z_j + const), keeping non-constant terms
    for i in 1:N-1, j in i+1:N
        V_ij = V / (j - i)^6
        H_sys += (V_ij/4) * (pauli_zz(i, j, N) + pauli_z(i, N) + pauli_z(j, N))
    end

    return H_sys
end