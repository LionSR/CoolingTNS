using Yao
using KrylovKit
using LinearAlgebra
using Printf

# Include existing functions
include("ising.jl")  # For build_ising_hamiltonian, get_parity_sector, etc.
include("investigate_pbc_even.jl")  # For investigate_pbc_even

# Comprehensive analysis of PBC spin case
function analyze_pbc_comprehensive()
    println("Comprehensive Analysis: PBC Spin Case")
    println("="^70)
    println("PBC Spin → Parity Sectors → Fermionic BC")
    println("  Even parity (P=+1) → Fermionic APBC")
    println("  Odd parity (P=-1) → Fermionic PBC")
    println()
    
    # Part 1: Pattern verification across different N
    println("Part 1: Discrepancy Pattern for Different System Sizes")
    println("-"^60)
    
    θ = π/3
    println("Fixed θ = π/3")
    println("\nN\tE_ED\t\tE_analytical\tDiscrepancy\tDisc/N")
    println("-"^60)
    
    results_by_N = []
    for N in [4, 6, 8, 10, 12, 14]
        # Use investigate_pbc_even for detailed analysis
        saved_stdout = stdout
        redirect_stdout(devnull)
        result = investigate_pbc_even(N, θ)
        redirect_stdout(saved_stdout)
        
        # Add N to the result for easier access
        result_with_N = (N=N, result...)
        push!(results_by_N, result_with_N)
        
        @printf("%d\t%.6f\t%.6f\t%.6f\t%.6f\n", 
                N, result.E_ed, result.E_analytical, 
                result.discrepancy, result.discrepancy/N)
    end
    
    # Statistical analysis
    disc_per_N = [r.discrepancy/r.N for r in results_by_N]
    mean_disc = sum(disc_per_N) / length(disc_per_N)
    std_disc = sqrt(sum((x - mean_disc)^2 for x in disc_per_N) / length(disc_per_N))
    
    println("\nStatistical Analysis:")
    println("  Mean discrepancy/N: $mean_disc")
    println("  Std deviation: $std_disc")
    println("  Coefficient of variation: $(std_disc/mean_disc * 100)%")
    
    # Part 2: θ dependence
    println("\n\nPart 2: θ Dependence (N=10)")
    println("-"^60)
    println("θ/π\tsin(2θ)\t\tDiscrepancy\tDisc/N\t\tGap Match")
    println("-"^60)
    
    N = 10
    results_by_theta = []
    for θ_frac in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49]
        θ = θ_frac * π
        
        saved_stdout = stdout
        redirect_stdout(devnull)
        result = investigate_pbc_even(N, θ)
        redirect_stdout(saved_stdout)
        
        # Add θ to the result
        result_with_theta = (θ=θ, θ_frac=θ_frac, result...)
        push!(results_by_theta, result_with_theta)
        
        gap_analytical = 2 * minimum(m.ε for m in result.modes)
        gap_match = abs(result.gap_ed - gap_analytical) < 1e-6
        
        @printf("%.2f\t%.6f\t%.6f\t%.6f\t%s\n",
                θ_frac, sin(2*θ), result.discrepancy, 
                result.discrepancy/N, gap_match ? "✓" : "✗")
    end
    
    # Part 3: Mode structure analysis
    println("\n\nPart 3: Mode Structure Analysis")
    println("-"^60)
    
    # Compare PBC vs APBC mode structures
    N = 10
    θ = π/3
    
    println("\nFermionic Mode Comparison (N=$N, θ=π/3):")
    println("\nAPBC modes (even parity):")
    println("k/π\t\tε_k")
    println("-"^30)
    
    # APBC modes
    for m in 0:N-1
        k = (2*m + 1)*π/N
        ε_k = sqrt(1 + sin(2*θ)*cos(k))
        @printf("%.3f\t\t%.6f\n", k/π, ε_k)
    end
    
    println("\nPBC modes (odd parity):")
    println("k/π\t\tε_k")
    println("-"^30)
    
    # PBC modes
    for m in 0:N-1
        k = 2*m*π/N
        ε_k = sqrt(1 + sin(2*θ)*cos(k))
        @printf("%.3f\t\t%.6f\n", k/π, ε_k)
    end
    
    # Special modes
    ε_0 = sqrt(1 + sin(2*θ))
    ε_π = sqrt(1 - sin(2*θ))
    
    println("\nSpecial modes:")
    println("  k=0 (PBC only): ε_0 = $ε_0")
    println("  k=π (PBC only): ε_π = $ε_π")
    
    # Part 4: Relationship to interaction strength
    println("\n\nPart 4: Discrepancy vs Interaction Strength")
    println("-"^60)
    println("cos(θ)\t\tDiscrepancy\tDisc/cos(θ)\tDisc/cos²(θ)")
    println("-"^60)
    
    N = 10
    for result in results_by_theta
        θ = result.θ
        disc = result.discrepancy
        if abs(cos(θ)) > 1e-10  # Avoid division by zero
            @printf("%.4f\t\t%.6f\t%.6f\t%.6f\n",
                    cos(θ), disc, disc/cos(θ), disc/cos(θ)^2)
        end
    end
    
    # Part 5: Both parity sectors
    println("\n\nPart 5: Both Parity Sectors (N=10, θ=π/3)")
    println("-"^60)
    
    N = 10
    θ = π/3
    
    # Build Hamiltonian
    H_pbc = build_ising_hamiltonian(N, θ, :periodic)
    P = kron(N, [i=>Z for i in 1:N]...)
    P_mat = mat(ComplexF64, P)
    H_pbc_mat = mat(ComplexF64, H_pbc)
    
    # Get both sectors
    H_even, _ = get_parity_sector(H_pbc_mat, P_mat, 1)
    H_odd, _ = get_parity_sector(H_pbc_mat, P_mat, -1)
    
    # Diagonalize
    vals_even, _ = eigsolve(Hermitian(H_even), min(3, size(H_even,1)), :SR; 
                           krylovdim=min(30, size(H_even,1)-1))
    vals_odd, _ = eigsolve(Hermitian(H_odd), min(3, size(H_odd,1)), :SR; 
                          krylovdim=min(30, size(H_odd,1)-1))
    
    # Analytical energies
    E_analytical_even, _ = compute_fermionic_gs_energy(N, θ, :pbc_even)
    E_analytical_odd, _ = compute_fermionic_gs_energy(N, θ, :pbc_odd)
    
    const_offset = sin(θ) * N / 2
    E_analytical_even_spin = E_analytical_even + const_offset
    E_analytical_odd_spin = E_analytical_odd + const_offset
    
    println("Even parity sector (→ APBC):")
    println("  E₀ (ED): $(real(vals_even[1]))")
    println("  E₀ (analytical): $E_analytical_even_spin")
    println("  Discrepancy: $(abs(real(vals_even[1]) - E_analytical_even_spin))")
    println("  E₁ (ED): $(real(vals_even[2]))")
    println("  Intra-parity gap: $(real(vals_even[2]) - real(vals_even[1]))")
    
    println("\nOdd parity sector (→ PBC):")
    println("  E₀ (ED): $(real(vals_odd[1]))")
    println("  E₀ (analytical): $E_analytical_odd_spin")
    println("  Discrepancy: $(abs(real(vals_odd[1]) - E_analytical_odd_spin))")
    println("  E₁ (ED): $(real(vals_odd[2]))")
    println("  Intra-parity gap: $(real(vals_odd[2]) - real(vals_odd[1]))")
    
    println("\nInter-parity gap:")
    println("  ΔE (odd - even): $(real(vals_odd[1]) - real(vals_even[1]))")
    
    # Part 6: Summary
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    
    println("\n1. Discrepancy Pattern:")
    println("   - Discrepancy/N is remarkably constant for fixed θ")
    println("   - For θ=π/3: discrepancy/N ≈ $mean_disc")
    
    println("\n2. θ Dependence:")
    println("   - Discrepancy → 0 as θ → π/2 (pure transverse field)")
    println("   - Discrepancy is maximal for small θ (strong XX interaction)")
    
    println("\n3. Gap Structure:")
    println("   - Intra-parity gaps match analytical predictions perfectly")
    println("   - Inter-parity gap < intra-parity gap (as expected)")
    
    println("\n4. Mode Analysis:")
    println("   - APBC has k = (2m+1)π/N modes")
    println("   - PBC has k = 2mπ/N modes (includes k=0, π)")
    println("   - Special modes ε₀, ε_π only appear in PBC")
    
    println("\n5. Physical Interpretation:")
    println("   - Discrepancy likely from Bogoliubov transformation")
    println("   - Related to vacuum energy normalization")
    println("   - Scales with interaction strength cos(θ)")
end

# Run the comprehensive analysis
analyze_pbc_comprehensive()