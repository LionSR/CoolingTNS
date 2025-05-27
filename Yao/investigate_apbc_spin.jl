using Yao
using KrylovKit
using LinearAlgebra
using Printf

# Build Ising Hamiltonian with APBC
function build_ising_hamiltonian_apbc(N::Int, θ::Real)
    """
    H = (cos θ)/2 * Σ σ_x^i σ_x^{i+1} + (sin θ)/2 * Σ σ_z^i
    with anti-periodic boundary conditions on the spin chain
    """
    # Regular nearest-neighbor interactions
    interaction = sum(i -> (cos(θ)/2) * kron(N, i=>X, (i+1)=>X), 1:N-1)
    
    # APBC: Add negative sign for the wrap-around term
    interaction += -(cos(θ)/2) * kron(N, N=>X, 1=>X)
    
    # Field terms (same as PBC/OBC)
    field = sum(i -> (sin(θ)/2) * kron(N, i=>Z), 1:N)
    
    return interaction + field
end

# Build parity operator (same for any BC)
function build_parity_operator(N::Int)
    P = reduce(kron, [X for _ in 1:N])
    return P
end

# Get parity sector
function get_parity_sector(H_mat::Matrix, P_mat::Matrix, parity::Int)
    dim = size(H_mat, 1)
    # Projector for parity eigenvalue = +1 (even) or -1 (odd)
    proj = (I(dim) + parity * P_mat) / 2
    # Find indices of states in this sector
    indices = findall(x -> abs(x) > 1e-10, diag(proj))
    # Extract submatrix
    H_sector = H_mat[indices, indices]
    return H_sector, indices
end

# Analytical ground state energy for fermionic PBC
function analytical_gs_energy_pbc(N::Int, θ::Real)
    """
    For APBC spin + even parity → fermionic PBC
    PBC: k = 2πm/N, m = 0, 1, ..., N-1
    """
    E_gs = 0.0
    mode_details = []
    
    for m in 0:N-1
        k = 2*m*π/N
        ε_k = sqrt(1 + sin(2*θ)*cos(k))
        E_gs -= ε_k  # Each mode contributes -ε_k to ground state
        push!(mode_details, (m=m, k=k, ε=ε_k))
    end
    
    # Add constant offset from JW transformation
    const_offset = sin(θ) * N / 2
    
    return E_gs + const_offset, mode_details
end

# Main investigation function
function investigate_apbc_spin(N::Int, θ::Real)
    println("="^60)
    println("INVESTIGATING APBC SPIN → EVEN PARITY → FERMIONIC PBC")
    println("="^60)
    println("System: N=$N, θ=$(θ/π)π")
    println("sin(θ) = $(sin(θ)), cos(θ) = $(cos(θ))")
    
    # Build APBC Hamiltonian
    H_apbc = build_ising_hamiltonian_apbc(N, θ)
    H_mat = Matrix(mat(ComplexF64, H_apbc))  # Convert to dense
    
    # Build parity operator
    P = build_parity_operator(N)
    P_mat = Matrix(mat(ComplexF64, P))  # Convert to dense
    
    # Get even parity sector
    H_even, _ = get_parity_sector(H_mat, P_mat, 1)
    println("\nEven parity sector dimension: $(size(H_even, 1))")
    
    # Diagonalize
    vals_even, _, _ = eigsolve(Hermitian(H_even), min(2, size(H_even,1)), :SR; 
                              krylovdim=min(30, size(H_even,1)-1))
    
    E_ed = real(vals_even[1])
    gap_ed = length(vals_even) > 1 ? real(vals_even[2] - vals_even[1]) : NaN
    
    println("\nED results (even parity):")
    println("  Ground state energy: $E_ed")
    if !isnan(gap_ed)
        println("  First excited energy: $(real(vals_even[2]))")
        println("  Energy gap: $gap_ed")
    end
    
    # Analytical calculation
    E_analytical, modes = analytical_gs_energy_pbc(N, θ)
    
    println("\nFermionic PBC ground state calculation:")
    println("  Number of modes: $(length(modes))")
    
    # Find minimum mode energy for gap
    ε_min = minimum(m.ε for m in modes)
    gap_analytical = 2 * ε_min
    
    # Special modes
    ε_0 = sqrt(1 + sin(2*θ))  # k=0 mode
    ε_π = sqrt(1 - sin(2*θ))  # k=π mode
    
    println("\nSpecial modes in PBC:")
    println("  k=0: ε_0 = $ε_0")
    println("  k=π: ε_π = $ε_π")
    println("  Minimum ε_k = $ε_min")
    
    # Comparison
    discrepancy = E_ed - E_analytical
    
    println("\n" * "="^40)
    println("COMPARISON")
    println("="^40)
    println("ED ground state energy:      $E_ed")
    println("Analytical (with offset):    $E_analytical")
    println("Discrepancy:                 $discrepancy")
    println("Discrepancy/N:               $(discrepancy/N)")
    
    if !isnan(gap_ed)
        println("\nGap comparison:")
        println("  ED gap:         $gap_ed")
        println("  Analytical gap: $gap_analytical")
        println("  Match: $(abs(gap_ed - gap_analytical) < 1e-6 ? "✓" : "✗")")
    end
    
    return (N=N, θ=θ, E_ed=E_ed, E_analytical=E_analytical, 
            discrepancy=discrepancy, gap_ed=gap_ed, gap_analytical=gap_analytical,
            modes=modes)
end

# Test multiple cases
println("Testing APBC spin case with even parity projection")
println()

# Test 1: N=10, θ=π/3 (same as PBC case for comparison)
result1 = investigate_apbc_spin(10, π/3)

# Test 2: Different N values
println("\n\n" * "="^60)
println("TESTING DIFFERENT SYSTEM SIZES")
println("="^60)
println("\nFixed θ=π/3, varying N:")
println("N\tDiscrepancy\tDisc/N")
println("-"^30)

for N in [4, 6, 8, 10, 12]
    saved_stdout = stdout
    redirect_stdout(devnull)
    result = investigate_apbc_spin(N, π/3)
    redirect_stdout(saved_stdout)
    
    @printf("%d\t%.6f\t%.6f\n", N, result.discrepancy, result.discrepancy/N)
end

# Test 3: Different θ values
println("\n\n" * "="^60)
println("TESTING DIFFERENT θ VALUES")
println("="^60)
println("\nFixed N=10, varying θ:")
println("θ/π\tDiscrepancy\tDisc/N\t\tGap match")
println("-"^50)

for θ_frac in [0.1, 0.2, 0.3, 0.4, 0.5]
    θ = θ_frac * π
    
    saved_stdout = stdout
    redirect_stdout(devnull)
    result = investigate_apbc_spin(10, θ)
    redirect_stdout(saved_stdout)
    
    gap_match = abs(result.gap_ed - result.gap_analytical) < 1e-6
    
    @printf("%.1f\t%.6f\t%.6f\t%s\n", 
            θ_frac, result.discrepancy, result.discrepancy/result.N, 
            gap_match ? "✓" : "✗")
end

# Compare with PBC results
println("\n\n" * "="^60)
println("COMPARISON: PBC vs APBC SPIN CHAINS")
println("="^60)
println("\nBoth with even parity projection, N=10, θ=π/3:")
println()

# Results comparison
println("From previous investigations:")

println("\nAPBC spin + even parity → fermionic PBC:")
println("  Discrepancy/N = $(result1.discrepancy/result1.N)")

println("\nKey difference:")
println("  Fermionic PBC includes k=0 and k=π modes")
println("  Fermionic APBC has k = (2m+1)π/N modes only")