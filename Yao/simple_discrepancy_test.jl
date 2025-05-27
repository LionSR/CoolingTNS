using Yao
using KrylovKit
using LinearAlgebra
using Printf

include("investigate_pbc_even.jl")

# Simple test to verify the pattern
println("Testing discrepancy pattern for PBC even parity → APBC")
println("="^60)

# Test with θ=π/3
θ = π/3
println("θ = π/3, sin(θ) = $(sin(θ)), cos(θ) = $(cos(θ))")
println("\nN\tDiscrepancy\tDisc/N\t\tGap matches?")
println("-"^50)

for N in [4, 6, 8, 10, 12]
    # Suppress verbose output
    saved_stdout = stdout
    redirect_stdout(devnull)
    
    result = investigate_pbc_even(N, θ)
    
    redirect_stdout(saved_stdout)
    
    # Check gap
    gap_analytical = 2 * minimum(m.ε for m in result.modes)
    gap_matches = abs(result.gap_ed - gap_analytical) < 1e-6
    
    @printf("%d\t%.8f\t%.8f\t%s\n", 
            N, result.discrepancy, result.discrepancy/N, gap_matches ? "✓" : "✗")
end

println("\nObservation: Discrepancy/N is remarkably constant!")

# Check at θ=π/2 (should be zero)
println("\nSpecial case θ=π/2 (pure transverse field):")
saved_stdout = stdout
redirect_stdout(devnull)
result_half = investigate_pbc_even(10, π/2)
redirect_stdout(saved_stdout)

println("N=10, θ=π/2: Discrepancy = $(result_half.discrepancy)")
println("This confirms discrepancy → 0 when cos(θ) → 0 (no XX interaction)")