using Yao
using KrylovKit
using LinearAlgebra
using Printf

# Include necessary functions
include("ising.jl")

# Quick summary of PBC spin case findings
function summarize_pbc_findings()
    println("="^70)
    println("SUMMARY: PBC SPIN CASE INVESTIGATION")
    println("="^70)
    
    println("\nKey Finding: For PBC spin model with even parity projection")
    println("(which maps to fermionic APBC), there is a systematic")
    println("discrepancy between ED and analytical ground state energies.")
    
    # Test case: N=10, θ=π/3
    N = 10
    θ = π/3
    
    println("\nExample: N=$N, θ=π/3")
    println("-"^40)
    
    # Build and solve
    H_pbc = build_ising_hamiltonian(N, θ, :periodic)
    P = kron(N, [i=>Z for i in 1:N]...)
    P_mat = mat(ComplexF64, P)
    H_pbc_mat = mat(ComplexF64, H_pbc)
    
    H_even, _ = get_parity_sector(H_pbc_mat, P_mat, 1)
    vals_even, _ = eigsolve(Hermitian(H_even), 2, :SR; krylovdim=min(30, size(H_even,1)-1))
    
    E_ed = real(vals_even[1])
    gap_ed = real(vals_even[2] - vals_even[1])
    
    # Analytical
    E_analytical_fermionic, energies = compute_fermionic_gs_energy(N, θ, :pbc_even)
    E_analytical = E_analytical_fermionic + sin(θ) * N / 2
    gap_analytical = 2 * minimum(energies)
    
    discrepancy = E_ed - E_analytical
    
    println("ED ground state energy: $E_ed")
    println("Analytical energy: $E_analytical")
    println("Discrepancy: $discrepancy")
    println("Discrepancy/N: $(discrepancy/N)")
    
    println("\nGap comparison:")
    println("ED gap: $gap_ed")
    println("Analytical gap: $gap_analytical")
    println("Gap match: $(abs(gap_ed - gap_analytical) < 1e-6 ? "✓" : "✗")")
    
    # Pattern across N
    println("\n\nPattern: Discrepancy/N is constant for fixed θ")
    println("-"^40)
    println("N\tDiscrepancy/N")
    println("-"^20)
    
    for N_test in [4, 6, 8, 10, 12]
        result = test_single_case(N_test, θ; verbose=false)
        @printf("%d\t%.6f\n", N_test, result.discrepancy_per_mode)
    end
    
    # Pattern across θ
    println("\n\nPattern: Discrepancy vanishes as θ→π/2")
    println("-"^40)
    println("θ/π\tcos(θ)\t\tDiscrepancy/N")
    println("-"^35)
    
    N = 10
    for θ_frac in [0.1, 0.2, 0.3, 0.4, 0.49]
        result = test_single_case(N, θ_frac * π; verbose=false)
        @printf("%.1f\t%.4f\t\t%.6f\n", θ_frac, cos(θ_frac * π), result.discrepancy_per_mode)
    end
    
    println("\n\nPhysical Interpretation:")
    println("-"^40)
    println("1. The discrepancy is NOT a simple constant like ε_π")
    println("2. It scales with interaction strength cos(θ)")
    println("3. It vanishes at θ=π/2 (pure transverse field, no XX)")
    println("4. Likely related to Bogoliubov transformation details")
    println("5. All gaps match perfectly - mode energies are correct")
    
    println("\n\nConclusion:")
    println("-"^40)
    println("The PBC spin model with even parity projection shows a")
    println("systematic ground state energy offset that:")
    println("- Is proportional to system size N")
    println("- Depends on interaction strength cos(θ)")
    println("- Does NOT affect energy gaps")
    println("- Suggests a missing normalization constant in the")
    println("  fermionic representation")
end

# Run the summary
summarize_pbc_findings()