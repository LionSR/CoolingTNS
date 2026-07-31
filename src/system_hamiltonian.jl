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

"""
    IsingFamilyParameters

Hamiltonian parameters that are a `J`-weighted `ZZ` chain plus purely
single-site system fields: `IsingModel` and `NiIsingModel`. Only the
single-site part differs, so each construction path (TN `OpSum`, ED sparse
matrix, and the interleaved Trotter circuit in `trotter.jl`) writes the shared
`ZZ` skeleton once and reaches the fields by dispatch.
"""
const IsingFamilyParameters =
    Union{HamiltonianParameters{IsingModel}, HamiltonianParameters{NiIsingModel}}

"""
    rydberg_rabi_x_coefficient(Ω)

Return the coefficient multiplying `σˣ` in the Rydberg Hamiltonian. The input
`Ω` is the Rabi frequency, so the Hamiltonian term is `(Ω/2) σˣ`.

For the tensor-network backend, `S+ + S- = σˣ`, hence the same coefficient is
used for the two ladder-operator terms.
"""
rydberg_rabi_x_coefficient(Ω) = Ω / 2

"""
    rydberg_interaction_coefficient(V, i, j)

Return the van der Waals coefficient multiplying `n_i n_j` for Rydberg sites
`i < j`, namely `V / |i-j|^6`.
"""
function rydberg_interaction_coefficient(V, i::Int, j::Int)
    i == j && throw(ArgumentError("Rydberg interaction sites must be distinct."))
    return V / abs(j - i)^6
end

"""
    rydberg_number_identity_shift(N, Δ, V)

Return the coefficient of the identity operator in the Rydberg Hamiltonian after
expanding the number operators as `n = (I + σᶻ)/2`.

The tensor-network construction uses `ProjUp` number operators directly. The ED
construction uses Pauli `σᶻ` operators, so this scalar is added once in ED to
make the absolute Hamiltonian, and hence its energy expectation values, agree
with the tensor-network convention.
"""
function rydberg_number_identity_shift(N::Int, Δ, V)
    interaction_shift = sum(
        (rydberg_interaction_coefficient(V, i, j) / 4 for i in 1:N-1 for j in i+1:N);
        init=0.0,
    )
    return -N * Δ / 2 + interaction_shift
end

function require_open_rydberg_boundary(ham_params::HamiltonianParameters{RydbergModel})
    ham_params.bc == :open && return nothing
    throw(ArgumentError(
        "Rydberg Hamiltonian construction currently supports only :open boundary " *
        "conditions; got $(ham_params.bc). A periodic Rydberg model needs an " *
        "explicit distance convention."
    ))
end

# ============================================================================
# Tensor Network (ITensors) Implementations
# ============================================================================

"""
    append_system_terms_tn(terms::OpSum, ham_params, site_of) -> OpSum

Append the tensor-network system Hamiltonian terms to `terms`.

The function `site_of(i)` gives the site number at which the system spin `i`
is embedded.  For an isolated system this is `i`; for the interleaved
system-bath chain it is `2i-1`.  This keeps the model-dependent Hamiltonian
terms in one place.
"""
function append_system_terms_tn(terms::OpSum, ham_params::HamiltonianParameters, site_of)
    error("TN system terms not implemented for model $(typeof(ham_params.model))")
end

"""
    spin_boundary_bond_sign(bc::Symbol) -> Int

Return the coefficient sign of the closing Ising spin bond `Z_N Z_1`.
The values are `0` for an open chain, `+1` for a periodic chain, and `-1`
for an antiperiodic chain.
"""
function spin_boundary_bond_sign(bc::Symbol)
    bc == :open && return 0
    bc == :periodic && return 1
    bc == :antiperiodic && return -1
    throw(ArgumentError("Unsupported boundary condition: $bc"))
end

"""Append Ising ZZ chain terms with the requested spin boundary condition."""
function append_zz_chain_terms_tn(terms::OpSum, J::Real, N::Int, bc::Symbol, site_of)
    for i in 1:N-1
        terms += J, "Z", site_of(i), "Z", site_of(i+1)
    end
    boundary_sign = spin_boundary_bond_sign(bc)
    if boundary_sign != 0
        terms += boundary_sign * J, "Z", site_of(N), "Z", site_of(1)
    end
    return terms
end

"""Append the Ising transverse field `h X_i`."""
function append_single_site_fields_tn(terms::OpSum,
        ham_params::HamiltonianParameters{IsingModel}, site_of)
    h = ham_params.params.h
    for i in 1:ham_params.N
        terms += h, "X", site_of(i)
    end
    return terms
end

"""Append the non-integrable Ising fields `hx X_i + hz Z_i`."""
function append_single_site_fields_tn(terms::OpSum,
        ham_params::HamiltonianParameters{NiIsingModel}, site_of)
    hx, hz = ham_params.params.hx, ham_params.params.hz
    for i in 1:ham_params.N
        terms += hx, "X", site_of(i)
        terms += hz, "Z", site_of(i)
    end
    return terms
end

"""
Append the Ising-family system terms: the shared `ZZ` chain, then the
model-dependent single-site fields from [`append_single_site_fields_tn`](@ref).
"""
function append_system_terms_tn(terms::OpSum, ham_params::IsingFamilyParameters, site_of)
    terms = append_zz_chain_terms_tn(
        terms, ham_params.params.J, ham_params.N, ham_params.bc, site_of
    )
    return append_single_site_fields_tn(terms, ham_params, site_of)
end

function append_system_terms_tn(
    terms::OpSum,
    ham_params::HamiltonianParameters{RydbergModel},
    site_of,
)
    require_open_rydberg_boundary(ham_params)

    Ω, Δ, V = ham_params.params.Ω, ham_params.params.Δ, ham_params.params.V
    Ωx = rydberg_rabi_x_coefficient(Ω)
    N = ham_params.N

    for i in 1:N
        terms += Ωx, "S+", site_of(i)
        terms += Ωx, "S-", site_of(i)
        terms += -Δ, "ProjUp", site_of(i)
    end
    for i in 1:N-1, j in i+1:N
        terms += rydberg_interaction_coefficient(V, i, j),
            "ProjUp", site_of(i), "ProjUp", site_of(j)
    end
    return terms
end

function construct_system_hamiltonian(
    ham_params::HamiltonianParameters,
    ::TNBackend,
    sites::Vector{<:Index},
)
    terms = append_system_terms_tn(OpSum(), ham_params, identity)
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
    boundary_sign = spin_boundary_bond_sign(bc)
    boundary_sign != 0 && (H .+= boundary_sign * J * pauli_zz(N, 1, N))
    return H
end

"""Add the Ising transverse field `h X_i`."""
function add_single_site_fields_ed!(H::SparseMatrixCSC, ham_params::HamiltonianParameters{IsingModel})
    N, h = ham_params.N, ham_params.params.h
    for i in 1:N
        H .+= h * pauli_x(i, N)
    end
    return H
end

"""Add the non-integrable Ising fields `hx X_i + hz Z_i`."""
function add_single_site_fields_ed!(H::SparseMatrixCSC, ham_params::HamiltonianParameters{NiIsingModel})
    N = ham_params.N
    hx, hz = ham_params.params.hx, ham_params.params.hz
    for i in 1:N
        H .+= hx * pauli_x(i, N)
        H .+= hz * pauli_z(i, N)
    end
    return H
end

"""
Build the Ising-family ED system Hamiltonian: the shared `ZZ` chain, then the
model-dependent single-site fields from [`add_single_site_fields_ed!`](@ref).
"""
function construct_system_hamiltonian(ham_params::IsingFamilyParameters, ::EDBackend, ::Int)
    N = ham_params.N
    H_sys = spzeros(Float64, 2^N, 2^N)
    add_zz_chain_ed!(H_sys, ham_params.params.J, N, ham_params.bc)
    return add_single_site_fields_ed!(H_sys, ham_params)
end

function construct_system_hamiltonian(ham_params::HamiltonianParameters{RydbergModel}, ::EDBackend, ::Int)
    require_open_rydberg_boundary(ham_params)

    Ω, Δ, V = ham_params.params.Ω, ham_params.params.Δ, ham_params.params.V
    Ωx = rydberg_rabi_x_coefficient(Ω)
    N = ham_params.N

    H_sys = spzeros(Float64, 2^N, 2^N)

    # Single-site terms: (Ω/2) X - Δ n with n = (I + Z)/2.
    for i in 1:N
        H_sys .+= Ωx * pauli_x(i, N) - (Δ/2) * pauli_z(i, N)
    end

    # Van der Waals interaction: V/r^6 * n_i * n_j where n = (I + Z)/2
    # Expands to V/4r^6 * (Z_i*Z_j + Z_i + Z_j + I).
    for i in 1:N-1, j in i+1:N
        V_ij = rydberg_interaction_coefficient(V, i, j)
        H_sys .+= (V_ij/4) * (pauli_zz(i, j, N) + pauli_z(i, N) + pauli_z(j, N))
    end

    identity_shift = rydberg_number_identity_shift(N, Δ, V)
    if identity_shift != 0
        H_sys .+= spdiagm(0 => fill(identity_shift, 2^N))
    end

    return H_sys
end
