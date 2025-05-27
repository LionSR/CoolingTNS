using Yao
using KrylovKit
using LinearAlgebra
using Printf

# Build the transverse field Ising Hamiltonian (same as before)
function build_ising_hamiltonian(N::Int, θ::Real, bc::Symbol)
    """
    H = (cos θ)/2 * Σ σ_x^i σ_x^{i+1} + (sin θ)/2 * Σ σ_z^i
    From Eq. \ref{eq:spin_hamiltonian} in MapToSpin.tex
    """
    if bc == :periodic
        interaction = sum(i -> (cos(θ)/2) * kron(N, i=>X, mod1(i+1, N)=>X), 1:N)
    else
        interaction = sum(i -> (cos(θ)/2) * kron(N, i=>X, (i+1)=>X), 1:N-1)
    end
    
    field = sum(i -> (sin(θ)/2) * kron(N, i=>Z), 1:N)
    
    return interaction + field
end

# Get the even parity sector
function get_even_parity_sector(H_mat::Matrix, P_mat::Matrix)
    dim = size(H_mat, 1)
    # Projector for even parity (P = +1)
    projector = (I(dim) + P_mat) / 2
    
    # Find states in even parity sector
    indices = findall(x -> abs(x) > 1e-10, diag(projector))
    
    # Extract the reduced Hamiltonian
    H_even = H_mat[indices, indices]
    
    return H_even, indices
end

# Compute fermionic ground state energy for APBC
function compute_fermionic_gs_apbc(N::Int, θ::Real; verbose=false)
    """
    For even parity spin sector → fermionic APBC
    From Eq. \ref{eq:diagonalized_hamiltonian} in MapToSpin.tex
    """
    E_gs = 0.0
    mode_details = []
    
    # APBC: k values are half-integers (2j+1)π/N for j = -N/2, ..., N/2-1
    for j in -N÷2:(N÷2-1)
        k = (2j + 1) * π / N
        # Mode energy from Eq. \ref{eq:mode_energy}
        ε_k = sqrt(1 + sin(2θ) * cos(k))
        
        # Ground state: all modes have n_k = 0, contributing -ε_k each
        E_gs -= ε_k
        
        push!(mode_details, (j=j, k=k, ε=ε_k, n=0, contribution=-ε_k))
    end
    
    # Add constant term from Eq. \ref{eq:transformed_hamiltonian}
    const_offset = sin(θ) * N / 2
    E_gs_with_offset = E_gs + const_offset
    
    if verbose
        println("\nFermionic APBC ground state calculation:")
        println("  Number of modes: $(length(mode_details))")
        println("  Mode contributions: Σ(-ε_k) = $E_gs")
        println("  Constant offset: $const_offset")
        println("  Total energy: $E_gs_with_offset")
        
        println("\nDetailed mode breakdown:")
        println("  j\tk/π\t\tε_k\t\tn_k\tcontribution")
        for (i, mode) in enumerate(mode_details)
            if i <= 3 || i >= length(mode_details)-2
                @printf("  %d\t%.3f\t\t%.6f\t%d\t%.6f\n", 
                        mode.j, mode.k/π, mode.ε, mode.n, mode.contribution)
            elseif i == 4
                println("  ...")
            end
        end
    end
    
    return E_gs_with_offset, E_gs, const_offset, mode_details
end

# Main investigation
function investigate_pbc_even(N::Int, θ::Real)
    println("="^60)
    println("INVESTIGATING PBC SPIN → EVEN PARITY → FERMIONIC APBC")
    println("="^60)
    println("System: N=$N, θ=$(θ/π)π")
    println("sin(θ) = $(sin(θ)), cos(θ) = $(cos(θ))")
    
    # Build spin Hamiltonian with PBC
    H_spin = build_ising_hamiltonian(N, θ, :periodic)
    H_mat = Matrix(mat(ComplexF64, H_spin))  # Convert to dense matrix
    
    # Parity operator
    P = kron(N, [i=>Z for i in 1:N]...)
    P_mat = Matrix(mat(ComplexF64, P))  # Convert to dense matrix
    
    # Get even parity sector
    H_even, indices = get_even_parity_sector(H_mat, P_mat)
    println("\nEven parity sector dimension: $(size(H_even, 1))")
    
    # Compute ED ground state
    vals, vecs = eigsolve(Hermitian(H_even), min(2, size(H_even,1)), :SR; 
                          krylovdim=min(30, size(H_even,1)-1))
    E0_ed = real(vals[1])
    E1_ed = length(vals) > 1 ? real(vals[2]) : NaN
    gap_ed = isnan(E1_ed) ? NaN : E1_ed - E0_ed
    
    println("\nED results (even parity):")
    println("  Ground state energy: $E0_ed")
    if !isnan(gap_ed)
        println("  First excited energy: $E1_ed")
        println("  Energy gap: $gap_ed")
    end
    
    # Compute analytical fermionic result
    E_analytical, E_fermionic, const_offset, modes = compute_fermionic_gs_apbc(N, θ; verbose=true)
    
    # Compare
    discrepancy = abs(E0_ed - E_analytical)
    println("\n" * "="^40)
    println("COMPARISON")
    println("="^40)
    println("ED ground state energy:      $E0_ed")
    println("Analytical (with offset):    $E_analytical")
    println("Discrepancy:                 $discrepancy")
    println("Discrepancy/N:               $(discrepancy/N)")
    
    # Check if gap matches
    if !isnan(gap_ed)
        gap_analytical = 2 * minimum(m.ε for m in modes)
        gap_match = abs(gap_ed - gap_analytical) < 1e-6
        println("\nGap comparison:")
        println("  ED gap:         $gap_ed")
        println("  Analytical gap: $gap_analytical")
        println("  Match: $(gap_match ? "✓" : "✗ (diff: $(abs(gap_ed - gap_analytical)))")")
    end
    
    # Try different constant interpretations
    println("\nExploring the discrepancy:")
    println("  If we had E_fermionic = $E_fermionic")
    println("  And offset = $const_offset")
    println("  What additional constant would we need?")
    println("  Required: $(E0_ed - E_fermionic - const_offset)")
    
    return (E_ed=E0_ed, E_analytical=E_analytical, discrepancy=discrepancy, 
            gap_ed=gap_ed, modes=modes)
end

# Run investigation for multiple cases
println("Testing N=10, θ=π/3 case:")
result = investigate_pbc_even(10, π/3)

println("\n\n" * "="^60)
println("TESTING OTHER PARAMETERS")
println("="^60)

# Test different N
println("\nVarying N with θ=π/3:")
for N in [6, 8, 10, 12]
    result = investigate_pbc_even(N, π/3)
    println("\nN=$N: Discrepancy = $(result.discrepancy), Discrepancy/N = $(result.discrepancy/N)")
end

# Test different θ
println("\n\nVarying θ with N=10:")
for θ_frac in [1/6, 1/4, 1/3, 1/2]
    θ = θ_frac * π
    result = investigate_pbc_even(10, θ)
    println("\nθ=$(θ_frac)π: Discrepancy = $(result.discrepancy)")
end