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

"""
    get_bath_operator(coupling::String) -> String

Return the bath Hamiltonian operator that doesn't commute with the coupling.
- XX, XY, XZ coupling → bath feels Z
- ZZ, YZ coupling → bath feels X
- YY coupling → bath feels Z
"""
function get_bath_operator(coupling::String)
    # Bath operator must NOT commute with coupling for energy transfer
    if coupling in ["ZZ", "YZ"]
        return "X"  # Z coupling → bath feels X
    else
        return "Z"  # X or Y coupling → bath feels Z
    end
end

function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters{IsingModel},
                                         ::TNBackend, sites::Vector{<:Index}, coupling_params::CouplingParameters)
    J, h = ham_params.params.J, ham_params.params.h
    N = ham_params.N
    g, Δ, coupling = coupling_params.g, coupling_params.delta, coupling_params.coupling
    bath_op = get_bath_operator(coupling)

    terms = OpSum()
    for i in 1:N-1
        terms += J, "Z", 2i-1, "Z", 2(i+1)-1
    end
    for i in 1:N
        terms += h, "X", 2i-1
        terms += Δ/2, bath_op, 2i             # Bath site - operator depends on coupling
        for (sys_op, bath_coupling_op) in coupling_operator_terms(coupling)
            terms += g, sys_op, 2i-1, bath_coupling_op, 2i
        end
    end

    return MPO(terms, sites)
end

function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel},
                                         ::TNBackend, sites::Vector{<:Index}, coupling_params::CouplingParameters)
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    N = ham_params.N
    g, Δ, coupling = coupling_params.g, coupling_params.delta, coupling_params.coupling
    bath_op = get_bath_operator(coupling)

    terms = OpSum()
    for i in 1:N-1
        terms += J, "Z", 2i-1, "Z", 2(i+1)-1
    end
    for i in 1:N
        terms += hx, "X", 2i-1
        terms += hz, "Z", 2i-1
        terms += Δ/2, bath_op, 2i             # Bath site - operator depends on coupling
        for (sys_op, bath_coupling_op) in coupling_operator_terms(coupling)
            terms += g, sys_op, 2i-1, bath_coupling_op, 2i
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
    N_total = nbits  # Should be 2 * N
    
    # Get system Hamiltonian on N qubits
    H_sys = construct_system_hamiltonian(ham_params, backend, N)
    
    # Initialize full Hamiltonian. Mixed couplings involving Y require the
    # standard complex Pauli Y to be Hermitian.
    H_sb = spzeros(ComplexF64, 2^N_total, 2^N_total)
    
    # Add system Hamiltonian terms with alternating layout mapping
    add_system_hamiltonian_ed!(H_sb, H_sys, N, N_total)
    
    # Add bath terms (at resonance with system gap if not specified)
    # Δ > 0 so bath ground state is eigenvalue -1 (|↓⟩ for Z, |−⟩ for X)
    Δ = coupling_params.delta !== nothing ? coupling_params.delta : compute_gap_ed(H_sys)
    
    # Bath qubits are at positions: 2, 4, 6, ..., 2N
    # Bath operator depends on coupling type (must not commute with coupling)
    coupling_type = coupling_params.coupling
    bath_op_func = (coupling_type in ["ZZ", "YZ"]) ? pauli_x : pauli_z

    for i in 1:N
        bath_idx = 2*i
        H_sb += (Δ/2) * bath_op_func(bath_idx, N_total)
    end
    
    # Add coupling terms
    g = coupling_params.g
    coupling_type = coupling_params.coupling
    
    for i in 1:N
        sys_idx = 2*i - 1  # System qubit i
        bath_idx = 2*i     # Corresponding bath qubit
        
        H_sb += construct_coupling_term_ed(sys_idx, bath_idx, N_total, coupling_type, g)
    end
    
    return H_sb
end



"""
    construct_zero_coupling_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites)

Create Hamiltonian with zero coupling for Trotter evolution using double dispatch.
"""
function construct_zero_coupling_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites)
    error("construct_zero_coupling_hamiltonian not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

function construct_zero_coupling_hamiltonian(ham_params::HamiltonianParameters, backend::TNBackend, sites::Vector{<:Index})
    zero_coupling_params = BasicCouplingParameters("XX", 0.0, 1, 0.0, 0.0)  # coupling, g, steps, te, delta
    return construct_system_bath_hamiltonian(ham_params, backend, sites, zero_coupling_params)
end

# ============================================================================
# ED Helper Functions
# ============================================================================

"""
    add_system_hamiltonian_ed!(H_sb, H_sys, N, N_total)

Add system Hamiltonian terms to the full system+bath Hamiltonian.
Embeds H_sys ⊗ I_bath by looping over all bath basis states.
"""
function add_system_hamiltonian_ed!(H_sb, H_sys, N, N_total)
    N_bath = N
    for bath_state in 0:(2^N_bath - 1)
        for i in 1:2^N, j in 1:2^N
            val = H_sys[i,j]
            if val != 0
                full_i = map_system_bath_to_full_basis_ed(i-1, bath_state, N)
                full_j = map_system_bath_to_full_basis_ed(j-1, bath_state, N)
                H_sb[full_i+1, full_j+1] += val
            end
        end
    end
end

"""
    map_system_to_full_basis_ed(sys_state::Int, N::Int) -> Int

Map a system basis state to the full system+bath basis (bath bits set to 0).
System qubits are at odd positions: 1, 3, 5, ...
"""
function map_system_to_full_basis_ed(sys_state::Int, N::Int)
    full_state = 0
    for i in 0:(N-1)
        if (sys_state >> i) & 1 == 1
            # System qubit i is at position 2*i in the full space (0-indexed)
            full_state |= (1 << (2*i))
        end
    end
    return full_state
end

"""
    map_system_bath_to_full_basis_ed(sys_state::Int, bath_state::Int, N::Int) -> Int

Map system and bath basis states to the full interleaved basis.
System qubit i → position 2i (0-indexed), bath qubit i → position 2i+1 (0-indexed).
"""
function map_system_bath_to_full_basis_ed(sys_state::Int, bath_state::Int, N::Int)
    full_state = 0
    for i in 0:(N-1)
        if (sys_state >> i) & 1 == 1
            full_state |= (1 << (2*i))       # System qubit at position 2i
        end
        if (bath_state >> i) & 1 == 1
            full_state |= (1 << (2*i + 1))   # Bath qubit at position 2i+1
        end
    end
    return full_state
end

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
