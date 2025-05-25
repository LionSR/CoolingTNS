"""
    system_bath_hamiltonian_ed_clean.jl

System-bath Hamiltonian construction for ED backend without Yao dependencies.
"""

include("ed_backend.jl")
include("system_hamiltonian_ed.jl")

# ============================================================================
# System-Bath Hamiltonian Construction for ED Backend
# ============================================================================

"""
    construct_system_bath_hamiltonian(ham_params::HamiltonianParameters, coupling_params::CouplingParameters, 
                                     backend::EDBackend, ::Int)

Construct system+bath Hamiltonian for ED backend with alternating qubit layout.
Layout: |s₁ b₁ s₂ b₂ ... sₙ bₙ⟩
"""
function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters, coupling_params::CouplingParameters, 
                                         backend::EDBackend, ::Int)
    N = ham_params.N
    N_total = 2 * N  # System + bath
    
    # Get system Hamiltonian on N qubits
    H_sys = construct_system_hamiltonian(ham_params, backend, N)
    
    # Initialize full Hamiltonian
    H_sb = spzeros(Float64, 2^N_total, 2^N_total)
    
    # Add system Hamiltonian terms
    # Need to map system indices to alternating pattern
    # System qubits are at positions: 1, 3, 5, ..., 2N-1
    add_system_hamiltonian!(H_sb, H_sys, N, N_total)
    
    # Add bath terms (at resonance with system gap if not specified)
    if coupling_params.delta !== nothing
        gap = coupling_params.delta
    else
        gap = compute_gap_ed(H_sys)
    end
    
    # Bath qubits are at positions: 2, 4, 6, ..., 2N
    # Use negative gap for cooling (bath should have lower energy)
    for i in 1:N
        bath_idx = 2*i
        H_sb += (-gap) * pauli_z(bath_idx, N_total)
    end
    
    # Add coupling terms
    g = coupling_params.g
    coupling_type = coupling_params.coupling
    
    for i in 1:N
        sys_idx = 2*i - 1  # System qubit i
        bath_idx = 2*i     # Corresponding bath qubit
        
        H_sb += construct_coupling_term(sys_idx, bath_idx, N_total, coupling_type, g)
    end
    
    return H_sb
end

"""
    add_system_hamiltonian!(H_sb, H_sys, N, N_total)

Add system Hamiltonian terms to the full system+bath Hamiltonian.
Maps system indices to alternating qubit layout.
"""
function add_system_hamiltonian!(H_sb, H_sys, N, N_total)
    # H_sys acts on N qubits in standard ordering
    # We need to map it to alternating layout in N_total qubits
    
    for i in 1:2^N, j in 1:2^N
        if H_sys[i,j] != 0
            # Map system basis states to full space
            full_i = map_system_to_full_basis(i-1, N)
            full_j = map_system_to_full_basis(j-1, N)
            H_sb[full_i+1, full_j+1] = H_sys[i,j]
        end
    end
end

"""
    map_system_to_full_basis(sys_state::Int, N::Int) -> Int

Map a system basis state to the full system+bath basis.
System qubits are at odd positions: 1, 3, 5, ...
"""
function map_system_to_full_basis(sys_state::Int, N::Int)
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
    construct_coupling_term(sys_idx::Int, bath_idx::Int, N_total::Int, coupling_type::String, g::Float64)

Construct coupling term between system and bath qubits.
"""
function construct_coupling_term(sys_idx::Int, bath_idx::Int, N_total::Int, coupling_type::String, g::Float64)
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
    return E1 - E0
end