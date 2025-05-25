"""
    system_bath_hamiltonian.jl

System+bath Hamiltonian construction using multiple dispatch on HamiltonianModel and backend.
"""

using ITensors
using LinearAlgebra
using SparseArrays
using KrylovKit

if !@isdefined(construct_coupling_term)
    include("system_hamiltonian.jl")  # For pauli operators
end


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

function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters{IsingModel}, 
                                         backend::TNBackend, sites::Vector{<:Index}, coupling_params::CouplingParameters)
    J, h = ham_params.params.J, ham_params.params.h
    g, Δ, coupling = coupling_params.g, coupling_params.delta, coupling_params.coupling
    
    N = ham_params.N
    # Use site indices (integers) instead of Index objects
    sys_sites = 1:2:2N-1
    bath_sites = 2:2:2N
    
    op1, op2 = parse_coupling(coupling)
    
    terms = OpSum()
    
    # System Hamiltonian
    for i in 1:N-1
        # sys_sites[i] is the i-th system site, sys_sites[i+1] is the next system site
        terms += J, "Z", 2i-1, "Z", 2(i+1)-1  # Direct calculation: site 1,3,5,7...
    end
    for i in 1:N
        terms += h, "X", 2i-1  # System sites: 1,3,5,7...
    end
    
    # Bath Hamiltonians  
    for i in 1:N
        terms += Δ/2, "Z", 2i  # Bath sites: 2,4,6,8... (positive for cooling)
    end
    
    # System-Bath coupling
    for i in 1:N
        terms += g, op1, 2i-1, op2, 2i  # System site coupled to adjacent bath site
    end
    
    return MPO(terms, sites)
end

function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, 
                                         backend::TNBackend, sites::Vector{<:Index}, coupling_params::CouplingParameters)
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    g, Δ, coupling = coupling_params.g, coupling_params.delta, coupling_params.coupling
    
    N = ham_params.N
    # Use site indices (integers) instead of Index objects
    sys_sites = 1:2:2N-1
    bath_sites = 2:2:2N
    
    op1, op2 = parse_coupling(coupling)
    
    terms = OpSum()
    
    # System Hamiltonian
    for i in 1:N-1
        # sys_sites[i] is the i-th system site, sys_sites[i+1] is the next system site
        terms += J, "Z", 2i-1, "Z", 2(i+1)-1  # Direct calculation: site 1,3,5,7...
    end
    for i in 1:N
        terms += hx, "X", 2i-1  # System sites: 1,3,5,7...
        terms += hz, "Z", 2i-1
    end
    
    # Bath Hamiltonians
    for i in 1:N
        terms += Δ/2, "Z", 2i  # Bath sites: 2,4,6,8...
    end
    
    # System-Bath coupling
    for i in 1:N
        terms += g, op1, 2i-1, op2, 2i  # System site coupled to adjacent bath site
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
    
    # Initialize full Hamiltonian
    H_sb = spzeros(Float64, 2^N_total, 2^N_total)
    
    # Add system Hamiltonian terms with alternating layout mapping
    add_system_hamiltonian_ed!(H_sb, H_sys, N, N_total)
    
    # Add bath terms (at resonance with system gap if not specified)
    Δ = coupling_params.delta !== nothing ? coupling_params.delta : -compute_gap_ed(H_sys)
    
    # Bath qubits are at positions: 2, 4, 6, ..., 2N
    # Copy TN backend exactly: use Δ/2 for bath energy
    for i in 1:N
        bath_idx = 2*i
        H_sb += (Δ/2) * pauli_z(bath_idx, N_total)
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
Maps system indices to alternating qubit layout.
"""
function add_system_hamiltonian_ed!(H_sb, H_sys, N, N_total)
    # H_sys acts on N qubits in standard ordering
    # We need to map it to alternating layout in N_total qubits
    
    for i in 1:2^N, j in 1:2^N
        if H_sys[i,j] != 0
            # Map system basis states to full space
            full_i = map_system_to_full_basis_ed(i-1, N)
            full_j = map_system_to_full_basis_ed(j-1, N)
            H_sb[full_i+1, full_j+1] = H_sys[i,j]
        end
    end
end

"""
    map_system_to_full_basis_ed(sys_state::Int, N::Int) -> Int

Map a system basis state to the full system+bath basis.
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
    construct_coupling_term_ed(sys_idx::Int, bath_idx::Int, N_total::Int, coupling_type::String, g::Float64)

Construct coupling term between system and bath qubits.
"""
function construct_coupling_term_ed(sys_idx::Int, bath_idx::Int, N_total::Int, coupling_type::String, g::Float64)
    if coupling_type == "XX"
        return g * pauli_x(sys_idx, N_total) * pauli_x(bath_idx, N_total)
        
    elseif coupling_type == "YY"
        return g * pauli_y(sys_idx, N_total) * pauli_y(bath_idx, N_total)
        
    elseif coupling_type == "ZZ"
        return g * pauli_z(sys_idx, N_total) * pauli_z(bath_idx, N_total)
        
    elseif coupling_type == "XY"
        # XY = X⊗Y + Y⊗X
        return g * (pauli_x(sys_idx, N_total) * pauli_y(bath_idx, N_total) +
                   pauli_y(sys_idx, N_total) * pauli_x(bath_idx, N_total))
        
    elseif coupling_type == "XZ"
        # XZ = X⊗Z + Z⊗X
        return g * (pauli_x(sys_idx, N_total) * pauli_z(bath_idx, N_total) +
                   pauli_z(sys_idx, N_total) * pauli_x(bath_idx, N_total))
        
    elseif coupling_type == "YZ"
        # YZ = Y⊗Z + Z⊗Y
        return g * (pauli_y(sys_idx, N_total) * pauli_z(bath_idx, N_total) +
                   pauli_z(sys_idx, N_total) * pauli_y(bath_idx, N_total))
        
    else
        error("Unknown coupling type: $coupling_type")
    end
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