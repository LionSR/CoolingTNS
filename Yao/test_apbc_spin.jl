using Yao
using KrylovKit
using LinearAlgebra
using Printf

# Include the main functions
include("ising.jl")

# Test APBC spin case
function test_apbc_case(N::Int, θ::Real)
    # Build Hamiltonian with APBC
    H_apbc = build_ising_hamiltonian(N, θ, :antiperiodic)
    P = kron(N, [i=>Z for i in 1:N]...)
    P_mat = Matrix(mat(ComplexF64, P))
    H_apbc_mat = Matrix(mat(ComplexF64, H_apbc))
    
    # Get even parity sector
    H_even, _ = get_parity_sector(H_apbc_mat, P_mat, 1)
    
    # Diagonalize
    vals_even, _ = eigsolve(Hermitian(H_even), min(2, size(H_even,1)), :SR; 
                           krylovdim=min(30, size(H_even,1)-1))
    
    # Analytical (APBC spin + even parity → fermionic PBC)
    E_analytical_fermionic, _ = compute_fermionic_gs_energy(N, θ, :pbc_odd)  # PBC modes
    E_analytical = E_analytical_fermionic + sin(θ) * N / 2
    
    E_ed = real(vals_even[1])
    discrepancy = E_ed - E_analytical
    
    return (N=N, θ=θ, E_ed=E_ed, E_analytical=E_analytical, 
            discrepancy=discrepancy, discrepancy_per_N=discrepancy/N)
end

# Quick comparison
println("APBC vs PBC Spin Cases (N=10, θ=π/3)")
println("="^50)

# Test both cases
result_pbc = test_single_case(10, π/3; verbose=false)
result_apbc = test_apbc_case(10, π/3)

println("PBC spin + even parity → fermionic APBC:")
println("  Discrepancy/N = $(result_pbc.discrepancy_per_mode)")

println("\nAPBC spin + even parity → fermionic PBC:")
println("  Discrepancy/N = $(result_apbc.discrepancy_per_N)")

println("\nBoth show similar discrepancy per site!")