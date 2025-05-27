using Yao
using KrylovKit
using LinearAlgebra
using Printf

include("investigate_pbc_even.jl")

# Analyze the discrepancy pattern
function analyze_discrepancy_pattern()
    println("="^60)
    println("DISCREPANCY PATTERN ANALYSIS")
    println("="^60)
    
    # Test 1: Fixed θ, varying N
    θ = π/3
    println("\nTest 1: Fixed θ=$(θ/π)π, varying N")
    println("N\tE_ED\t\tE_analytical\tDiscrepancy\tDisc/N")
    println("-"^60)
    
    discrepancies = []
    for N in [4, 6, 8, 10, 12]
        result = investigate_pbc_even(N, θ)
        push!(discrepancies, (N=N, disc=result.discrepancy, disc_per_N=result.discrepancy/N))
        @printf("%d\t%.6f\t%.6f\t%.6f\t%.6f\n", 
                N, result.E_ed, result.E_analytical, result.discrepancy, result.discrepancy/N)
    end
    
    # Check if discrepancy/N is constant
    disc_per_N_values = [d.disc_per_N for d in discrepancies]
    mean_disc_per_N = sum(disc_per_N_values) / length(disc_per_N_values)
    std_disc_per_N = sqrt(sum((x - mean_disc_per_N)^2 for x in disc_per_N_values) / length(disc_per_N_values))
    
    println("\nAnalysis:")
    println("  Mean discrepancy/N: $mean_disc_per_N")
    println("  Std dev: $std_disc_per_N")
    println("  Relative variation: $(std_disc_per_N/mean_disc_per_N * 100)%")
    
    # Test 2: Fixed N, varying θ
    N = 10
    println("\n\nTest 2: Fixed N=$N, varying θ")
    println("θ/π\tsin(θ)\t\tcos(θ)\t\tDiscrepancy\tε_π")
    println("-"^70)
    
    for θ_frac in [0.1, 0.2, 0.3, 0.4, 0.5]
        θ = θ_frac * π
        result = investigate_pbc_even(N, θ)
        ε_π = abs(sin(θ) - cos(θ))
        @printf("%.1f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                θ_frac, sin(θ), cos(θ), result.discrepancy, ε_π)
    end
    
    # Test 3: Check specific hypothesis
    println("\n\nTest 3: Checking if discrepancy relates to missing modes")
    
    # For APBC, we have N modes with k = (2j+1)π/N
    # For PBC, we would have k = 2πj/N including k=0 and k=π
    # The k=π mode has special energy ε_π = |sin(θ) - cos(θ)|
    
    θ = π/3
    N = 10
    result = investigate_pbc_even(N, θ)
    ε_π = abs(sin(θ) - cos(θ))
    
    println("\nFor N=$N, θ=$(θ/π)π:")
    println("  Discrepancy: $(result.discrepancy)")
    println("  ε_π = |sin(θ) - cos(θ)| = $ε_π")
    println("  Ratio: $(result.discrepancy/ε_π)")
    
    # What if the discrepancy is related to vacuum energy normalization?
    println("\n\nTest 4: Vacuum energy hypothesis")
    println("  In the fermionic picture, each mode contributes -ε_k to ground state")
    println("  Total modes in APBC: $N")
    println("  What if there's a systematic error of ~0.037 per mode?")
    
    # Check if the error could be from normal ordering
    println("\n  Checking BdG Hamiltonian structure:")
    println("  H_k = [[w_k, ir_k], [-ir_k, -w_k]] with eigenvalues ±ε_k")
    println("  Ground state: fill all negative energy states")
    println("  Could there be a zero-point energy contribution we're missing?")
end

# Run the analysis
analyze_discrepancy_pattern()