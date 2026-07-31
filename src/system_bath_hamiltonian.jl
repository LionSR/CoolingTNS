"""
    system_bath_hamiltonian.jl

System+bath Hamiltonian construction using multiple dispatch on HamiltonianModel and backend.
"""

using ITensors
using LinearAlgebra
using SparseArrays
using KrylovKit


# ============================================================================
# System-Bath Hamiltonian Construction Interface  
# ============================================================================

"""
    construct_system_bath_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites, coupling_params)

Generic interface for constructing full system+bath Hamiltonians with double dispatch.
"""
function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites, coupling_params)
    error("construct_system_bath_hamiltonian not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

# ============================================================================
# Tensor Network System-Bath Hamiltonians
# ============================================================================

# Note: OpSum uses immutable operations (+=), so bath/coupling terms are inlined in each constructor

function construct_system_bath_hamiltonian(
    ham_params::HamiltonianParameters,
    ::TNBackend,
    sites::Vector{<:Index},
    coupling_params::CouplingParameters,
)
    N = ham_params.N
    g, bath_detuning, coupling =
        coupling_params.g, coupling_params.delta, coupling_params.coupling
    bath_op = get_bath_operator(coupling)

    terms = append_system_terms_tn(OpSum(), ham_params, interleaved_system_site)
    for i in 1:N
        sys_site = interleaved_system_site(i)
        bath_site = interleaved_bath_site(i)
        terms += bath_detuning/2, bath_op, bath_site
        for (sys_op, bath_coupling_op) in coupling_operator_terms(coupling)
            terms += g, sys_op, sys_site, bath_coupling_op, bath_site
        end
    end

    return MPO(terms, sites)
end

# ============================================================================
# ED System-Bath Hamiltonians  
# ============================================================================

# ED System-Bath Hamiltonian Implementation
function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters, 
                                         backend::EDBackend, nbits::Int, coupling_params::CouplingParameters)
    N = ham_params.N
    N_total = nbits
    
    # Get system Hamiltonian on N qubits
    H_sys = construct_system_hamiltonian(ham_params, backend, N)

    # Seed the full Hamiltonian with H_sys ⊗ I_bath in the interleaved layout.
    # The element type is ComplexF64 because mixed couplings involving Y require
    # the standard complex Pauli Y to be Hermitian.
    H_sb = embed_system_hamiltonian_ed(H_sys, N, N_total)

    # Add bath terms (at resonance with system gap if not specified).
    # Δ > 0, so the bath ground state is the eigenvalue -1 state of bath_op.
    Δ = @something coupling_params.delta compute_gap_ed(H_sys)
    
    # Bath qubits are at positions: 2, 4, 6, ..., 2N.
    # get_bath_operator is the source of truth for the one-Pauli and mixed
    # symmetric bath-field conventions.
    coupling_type = coupling_params.coupling
    bath_op_func = ED_HAMILTONIAN_PAULI_MAP[get_bath_operator(coupling_type)]

    for i in 1:N
        bath_idx = interleaved_bath_site(i)
        H_sb += (Δ/2) * bath_op_func(bath_idx, N_total)
    end
    
    # Add coupling terms (coupling_type was bound above and is unchanged)
    g = coupling_params.g

    for i in 1:N
        sys_idx = interleaved_system_site(i)
        bath_idx = interleaved_bath_site(i)
        
        H_sb += construct_coupling_term_ed(sys_idx, bath_idx, N_total, coupling_type, g)
    end
    
    return H_sb
end

# ============================================================================
# ED Helper Functions
# ============================================================================

"""
    embed_system_hamiltonian_ed(H_sys, N, N_total) -> SparseMatrixCSC{ComplexF64,Int}

Embed `H_sys ⊗ I_bath` into the full interleaved system-bath basis, once per
bath basis state.

Built from a `findnz` triplet list rather than scalar `setindex!` into a sparse
matrix: each such assignment is O(nnz) for CSC, so the old form was quadratic in
the stored entries. `map_system_bath_to_full_basis_ed` is injective in
`(system state, bath state)`, so no two triplets collide and `sparse` has
nothing to sum — the assembled values are those of `H_sys` verbatim.

`V` is widened to `ComplexF64` even though `findnz(H_sys)` yields `Float64`:
mixed-Y couplings added by the caller need the Hermitian complex Pauli Y.
"""
function embed_system_hamiltonian_ed(H_sys, N, N_total)
    sys_rows, sys_cols, sys_vals = findnz(sparse(H_sys))
    # Matching the old `val != 0` guard keeps explicitly-stored zeros out of the
    # pattern, so `hash(H_sb)` — and therefore the `EVOLUTION_EIG_CACHE` key —
    # is unchanged.
    stored = findall(!iszero, sys_vals)

    n_bath_states = 2^N
    n_terms = length(stored) * n_bath_states
    rows = Vector{Int}(undef, n_terms)
    cols = Vector{Int}(undef, n_terms)
    vals = Vector{ComplexF64}(undef, n_terms)

    t = 0
    for bath_state in 0:(n_bath_states - 1)
        for k in stored
            t += 1
            rows[t] = map_system_bath_to_full_basis_ed(sys_rows[k] - 1, bath_state, N) + 1
            cols[t] = map_system_bath_to_full_basis_ed(sys_cols[k] - 1, bath_state, N) + 1
            vals[t] = sys_vals[k]
        end
    end

    return sparse(rows, cols, vals, 2^N_total, 2^N_total)
end

"""
    map_system_bath_to_full_basis_ed(sys_state::Int, bath_state::Int, N::Int) -> Int

Map system and bath basis states to the full interleaved basis.
"""
function map_system_bath_to_full_basis_ed(sys_state::Int, bath_state::Int, N::Int)
    return interleaved_basis_state(sys_state, bath_state, N)
end

# Load order: unlike most of this file, the right-hand side is evaluated at
# *load* time, and it evaluates function bindings rather than calling them --
# `pauli_x`/`pauli_z` from `ed_backend.jl` and `pauli_y_complex` from
# `ed_backend_complex_jw.jl`. Both must already be included when
# `system_bath_hamiltonian.jl` is, so this file cannot be moved above them in
# `CoolingTNS.jl`'s include list.
const ED_HAMILTONIAN_PAULI_MAP = Dict(
    "X" => pauli_x,
    "Y" => pauli_y_complex,
    "Z" => pauli_z,
)

"""
    construct_coupling_term_ed(sys_idx::Int, bath_idx::Int, N_total::Int, coupling_type::String, g::Float64)

Construct coupling term between system and bath qubits.
For symmetric couplings (XX, YY, ZZ): g * A⊗A
For mixed couplings (XY, XZ, YZ): g * (A⊗B + B⊗A)
"""
function construct_coupling_term_ed(sys_idx::Int, bath_idx::Int, N_total::Int, coupling_type::String, g::Float64)
    terms = coupling_operator_terms(coupling_type)
    result = spzeros(ComplexF64, 2^N_total, 2^N_total)

    for (sys_label, bath_label) in terms
        sys_op = ED_HAMILTONIAN_PAULI_MAP[sys_label](sys_idx, N_total)
        bath_op = ED_HAMILTONIAN_PAULI_MAP[bath_label](bath_idx, N_total)
        result += g * sys_op * bath_op
    end

    return result
end

"""
    compute_gap_ed(H::AbstractMatrix) -> Float64

Compute energy gap between ground and first excited state.
"""
function compute_gap_ed(H::AbstractMatrix)
    vals, _, _ = eigsolve(H, 2, :SR; krylovdim=min(30, size(H, 1)))
    E0 = real(vals[1])
    E1 = real(vals[2])
    return abs(E1 - E0)  # Return absolute value for positive frequency
end
