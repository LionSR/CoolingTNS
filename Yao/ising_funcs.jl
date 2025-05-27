using Yao
using KrylovKit
using LinearAlgebra
using Printf

# Build the transverse field Ising Hamiltonian using Yao's native methods
function build_ising_hamiltonian(N::Int, θ::Real, bc::Symbol)
    """
    H = (cos θ)/2 * Σ σ_x^i σ_x^{i+1} + (sin θ)/2 * Σ σ_z^i
    bc: :open, :periodic, or :antiperiodic
    """
    # Interaction terms using Yao's sum function
    if bc == :periodic
        # For PBC, include the wrap-around term
        interaction = sum(i -> (cos(θ)/2) * kron(N, i=>X, mod1(i+1, N)=>X), 1:N)
    elseif bc == :antiperiodic
        # For APBC, nearest neighbors plus negative wrap-around
        interaction = sum(i -> (cos(θ)/2) * kron(N, i=>X, (i+1)=>X), 1:N-1)
        interaction += -(cos(θ)/2) * kron(N, N=>X, 1=>X)
    else  # :open
        # For OBC, only nearest neighbors
        interaction = sum(i -> (cos(θ)/2) * kron(N, i=>X, (i+1)=>X), 1:N-1)
    end
    
    # Field terms
    field = sum(i -> (sin(θ)/2) * put(N, i=>Z), 1:N)
    
    return interaction + field
end


# Let's find ground state in each parity sector separately
function get_parity_sector(H_mat, P_mat, parity)
    dim = size(H_mat, 1)
    # Projector for parity eigenvalue = +1 (even) or -1 (odd)
    proj = (I(dim) + parity * P_mat) / 2
    # Find indices of states in this sector
    indices = findall(x -> abs(x) > 1e-10, diag(proj))
    # Extract submatrix
    H_sector = H_mat[indices, indices]
    return H_sector, indices
end

function epsilon_k(k, θ)
    return sqrt(1 + sin(2*θ) * cos(k))
end

# Ground state energy is sum of all negative mode energies
function compute_fermionic_gs_energy(N::Int, θ::Real, bc_type::Symbol)
    E_gs = 0.0
    all_energies = Float64[]
    
    if bc_type == :pbc_even  # Fermionic APBC
        # k values are half-integers: (2j+1)π/N for j = -N/2, ..., N/2-1
        for j in -N÷2:(N÷2-1)
            k = (2j + 1) * π / N
            eps_k = epsilon_k(k, θ)
            push!(all_energies, eps_k)
            E_gs -= eps_k  # Each mode contributes -eps_k to ground state
        end
    elseif bc_type == :pbc_odd  # Fermionic PBC  
        # k values: 2πj/N for j = -N/2+1, ..., N/2
        for j in (-N÷2+1):N÷2
            k = 2π * j / N
            eps_k = epsilon_k(k, θ)
            push!(all_energies, eps_k)
            E_gs -= eps_k
        end
    end
    
    return E_gs, sort(all_energies)
end


function find_gap_modes(N::Int, θ::Real, bc_type::Symbol)
    min_energy = Inf
    gap_modes = []
    
    if bc_type == :pbc_even  # APBC
        for j in -N÷2:(N÷2-1)
            k = (2j + 1) * π / N
            ε_k = sqrt(1 + sin(2θ) * cos(k))
            if ε_k < min_energy + 1e-10
                if abs(ε_k - min_energy) < 1e-10
                    push!(gap_modes, (j=j, k=k, ε=ε_k))
                else
                    min_energy = ε_k
                    gap_modes = [(j=j, k=k, ε=ε_k)]
                end
            end
        end
    elseif bc_type == :pbc_odd  # PBC
        for j in (-N÷2+1):N÷2
            k = 2π * j / N
            ε_k = sqrt(1 + sin(2θ) * cos(k))
            if ε_k < min_energy + 1e-10
                if abs(ε_k - min_energy) < 1e-10
                    push!(gap_modes, (j=j, k=k, ε=ε_k))
                else
                    min_energy = ε_k
                    gap_modes = [(j=j, k=k, ε=ε_k)]
                end
            end
        end
    end
    
    return gap_modes
end