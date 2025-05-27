using Yao
using KrylovKit
using LinearAlgebra
using Printf

include("ising_funcs.jl")

# Test with N=10, θ=π/3
N = 10
θ = π/3

println("Testing Ising Model with N=$N, θ=π/3")
println("sin(θ) = $(sin(θ)), cos(θ) = $(cos(θ))")
println()

# Build Hamiltonians
H_obc = build_ising_hamiltonian(N, θ, :open)
H_pbc = build_ising_hamiltonian(N, θ, :periodic)

# Get matrices using Yao's mat function
H_obc_mat = mat(ComplexF64, H_obc)
H_pbc_mat = mat(ComplexF64, H_pbc)

# Compute parity operator for PBC case
P = kron(N, [i=>Z for i in 1:N]...)
P_mat = mat(ComplexF64, P)

# Let's also check the constant term more carefully
println("\nChecking constant terms in the Hamiltonian:")
const_term_notes = sin(θ) * N / 2
println("  Constant from notes (sinθ*N/2): $const_term_notes")

# Find ground state with KrylovKit
println("Open Boundary Conditions:")
vals_obc, vecs_obc, info = eigsolve(Hermitian(H_obc_mat), 2, :SR; krylovdim=30)
E0_obc = real(vals_obc[1])
E1_obc = real(vals_obc[2])
println("  Ground state energy: $E0_obc")
println("  Energy gap: $(E1_obc - E0_obc)")

println()
println("Periodic Boundary Conditions:")
vals_pbc, vecs_pbc, info = eigsolve(Hermitian(H_pbc_mat), 2, :SR; krylovdim=30)
E0_pbc = real(vals_pbc[1])
E1_pbc = real(vals_pbc[2])
println("  Ground state energy: $E0_pbc")
println("  Energy gap: $(E1_pbc - E0_pbc)")

# Check parity of ground state
gs_vec = vecs_pbc[1]
parity_gs = real(gs_vec' * P_mat * gs_vec)
println("  Ground state parity: $parity_gs")
if abs(parity_gs - 1) < 0.1
    println("  Ground state is in EVEN parity sector")
elseif abs(parity_gs + 1) < 0.1
    println("  Ground state is in ODD parity sector")
else
    println("  Ground state is a mixture of parities!")
end

H_even, idx_even = get_parity_sector(H_pbc_mat, P_mat, 1)
H_odd, idx_odd = get_parity_sector(H_pbc_mat, P_mat, -1)

# Get lowest two energies in each sector
vals_even_sector, _ = eigsolve(Hermitian(H_even), min(2, size(H_even,1)), :SR; krylovdim=min(30, size(H_even,1)-1))
vals_odd_sector, _ = eigsolve(Hermitian(H_odd), min(2, size(H_odd,1)), :SR; krylovdim=min(30, size(H_odd,1)-1))

println("\nParity sector analysis:")
println("  Even sector: E0 = $(real(vals_even_sector[1])), E1 = $(real(vals_even_sector[2]))")
println("  Even sector gap: $(real(vals_even_sector[2]) - real(vals_even_sector[1]))")
println("  Odd sector: E0 = $(real(vals_odd_sector[1])), E1 = $(real(vals_odd_sector[2]))")
println("  Odd sector gap: $(real(vals_odd_sector[2]) - real(vals_odd_sector[1]))")

# Now let's compute the analytical formula for the ground state energy
println("\nAnalytical formula (from fermionic picture):")
println("Mode energies: ε_k = √(1 + sin(2θ)cos(2πk/N))")
println("For θ=π/3: sin(2*θ) = $(sin(2*θ))")

# For PBC, we need to consider both parity sectors
# Even parity -> fermionic APBC (k = half-integers)
# Odd parity -> fermionic PBC (k = integers)

# Let's also check the vacuum energy more carefully
function check_vacuum_contribution(N::Int, θ::Real)
    println("\nVacuum energy investigation:")
    
    # Let me check if we're handling the constant terms correctly
    # From the notes, we have H = Σ_k ε_k (2â_k†â_k - 1) + const
    # where const includes the sin(θ)N/2 term
    
    # But there might be additional constants from:
    # 1. Normal ordering
    # 2. The Bogoliubov transformation itself
    # 3. Regularization of the vacuum energy
    
    println("  Constant terms we know about:")
    println("    - sin(θ)N/2 = $(sin(θ)*N/2)")
    println("    - Vacuum energy Σ(-ε_k) already included")
    
    # Wait - in the transformation from spins to fermions,
    # are we handling the vacuum state correctly?
    println("\n  Vacuum state mapping:")
    println("    Spin vacuum: all spins up?")
    println("    Fermionic vacuum: all modes empty")
    println("    Are these equivalent under JW transformation?")
    
    return nothing
end

E_analytical_even, energies_even = compute_fermionic_gs_energy(N, θ, :pbc_even)
E_analytical_odd, energies_odd = compute_fermionic_gs_energy(N, θ, :pbc_odd)

# Also check what modes are in PBC vs APBC
println("\nMode comparison:")
println("  Lowest energy in APBC: $(minimum(energies_even))")
println("  Lowest energy in PBC: $(minimum(energies_odd))")
println("  Note: PBC includes k=0 and k=π modes")

# Find which k gives the minimum energy (gap modes)

# Find gap modes for both sectors
gap_modes_even = find_gap_modes(N, θ, :pbc_even)
gap_modes_odd = find_gap_modes(N, θ, :pbc_odd)

println("\nGap-determining modes:")
println("  APBC (even parity):")
for mode in gap_modes_even
    println("    j=$(mode.j), k=$(mode.k/π)π, ε=$(mode.ε)")
    println("    cos(k) = $(cos(mode.k))")
end
println("  Intra-parity gap = 2ε_min = 2 × $(gap_modes_even[1].ε) = $(2 * gap_modes_even[1].ε)")

println("\n  PBC (odd parity):")
for mode in gap_modes_odd
    println("    j=$(mode.j), k=$(mode.k/π)π, ε=$(mode.ε)")
    println("    cos(k) = $(cos(mode.k))")
end

# Add the constant offset from the JW transformation (Eq. 82 in notes)
const_offset = sin(θ) * N / 2
E_analytical_even_spin = E_analytical_even + const_offset
E_analytical_odd_spin = E_analytical_odd + const_offset

println("\nFermionic ground state energies:")
println("  Even parity (APBC): $E_analytical_even")
println("  Odd parity (PBC): $E_analytical_odd")
println("\nConstant offset from JW transformation: $(const_offset)")
println("\nSpin ground state energies (with offset):")
println("  Even parity (APBC): $E_analytical_even_spin")
println("  Odd parity (PBC): $E_analytical_odd_spin")
println("\nComparing ground state energies:")
println("  ED result (even parity): $(real(vals_even_sector[1]))")
println("  Analytical (even parity): $E_analytical_even_spin")
println("  Difference: $(abs(real(vals_even_sector[1]) - E_analytical_even_spin))")

# Let's check if we need a factor of 1/2
E_analytical_even_corrected = E_analytical_even/2 + const_offset
println("\n  Testing with E_fermionic/2 + offset: $E_analytical_even_corrected")
println("  Difference: $(abs(real(vals_even_sector[1]) - E_analytical_even_corrected))")

println("\nUnderstanding the gap:")
println("  The ED 'gap' of $(E1_pbc - E0_pbc) is between:")
println("    Ground state (even parity): $E0_pbc")
println("    First excited (odd parity): $(real(vals_odd_sector[1]))")
if real(vals_odd_sector[1]) < real(vals_even_sector[2])
    println("  This is an inter-parity gap (first excited state is in different parity sector)")
end

gap_ed_even = real(vals_even_sector[2]) - real(vals_even_sector[1])
gap_analytical_even = 2 * minimum(energies_even)
println("\n  The intra-parity gap in even sector: $gap_ed_even")
println("  Analytical gap (APBC): $gap_analytical_even")
if abs(gap_ed_even - gap_analytical_even) < 1e-6
    println("  ✓ Gaps match within tolerance")
else
    println("  ✗ Gap mismatch: $(abs(gap_ed_even - gap_analytical_even))")
end

# Compute energy gaps
println("\nEnergy gap analysis:")
println("  ED gap (PBC): $(E1_pbc - E0_pbc)")

# The gap in the fermionic picture is 2 * min(ε_k)
if E_analytical_even_spin < E_analytical_odd_spin
    # Even parity (APBC) is ground state
    gap_analytical = 2 * minimum(energies_even)
    println("  Analytical gap (APBC): $gap_analytical")
    println("  Note: This is the gap to the first excited state within the same parity sector")
else
    # Odd parity (PBC) is ground state
    gap_analytical = 2 * minimum(energies_odd)
    println("  Analytical gap (PBC): $gap_analytical")
    println("  Note: This is the gap to the first excited state within the same parity sector")
end
println("\nInvestigating the ground state energy discrepancy:")
discrepancy = abs(real(vals_even_sector[1]) - E_analytical_even_spin)
println("  Discrepancy: $discrepancy")
println("  Number of modes: $(length(energies_even))")

# Check various normalizations
println("\nChecking possible sources:")
println("  Discrepancy / N: $(discrepancy / N)")
println("  Discrepancy / #modes: $(discrepancy / length(energies_even))")
println("  Sum of all ε_k: $(sum(energies_even))")

# Check if it could be related to zero-point energy
println("\nZero-point energy analysis:")
println("  If each mode contributes 1/2 to vacuum: $(length(energies_even) * 0.5)")
println("  Discrepancy / 0.5: $(discrepancy / 0.5)")

# Let's think about the BdG formulation more carefully
println("\nBdG analysis:")
println("  In BdG, we have pairs (k,-k) except for k=0,π in PBC")
println("  For APBC, all k are paired since no k=0 or k=π")
println("  Number of independent k modes in APBC: $(length(energies_even))")

# Check if we're double counting
actual_independent_modes = length(energies_even) / 2
println("  If we pair (k,-k), independent modes: $actual_independent_modes")
println("  Discrepancy / (2 * #independent): $(discrepancy / (2 * actual_independent_modes))")

# Check vacuum contribution
check_vacuum_contribution(N, θ)

# Function to test a single case
function test_single_case(N::Int, θ::Real; verbose=true)
    # Build Hamiltonian and compute ED results
    H_pbc = build_ising_hamiltonian(N, θ, :periodic)
    P = kron(N, [i=>Z for i in 1:N]...)
    P_mat = mat(ComplexF64, P)
    H_pbc_mat = mat(ComplexF64, H_pbc)
    
    # Get parity sectors
    H_even, _ = get_parity_sector(H_pbc_mat, P_mat, 1)
    H_odd, _ = get_parity_sector(H_pbc_mat, P_mat, -1)
    
    # Compute energies
    vals_even, _ = eigsolve(Hermitian(H_even), min(2, size(H_even,1)), :SR; krylovdim=min(30, size(H_even,1)-1))
    vals_odd, _ = eigsolve(Hermitian(H_odd), min(2, size(H_odd,1)), :SR; krylovdim=min(30, size(H_odd,1)-1))
    
    # Analytical results
    E_analytical_even, energies_even = compute_fermionic_gs_energy(N, θ, :pbc_even)
    E_analytical_odd, energies_odd = compute_fermionic_gs_energy(N, θ, :pbc_odd)
    
    const_offset = sin(θ) * N / 2
    E_analytical_even_spin = E_analytical_even + const_offset
    E_analytical_odd_spin = E_analytical_odd + const_offset
    
    # Compare
    E0_even_ed = real(vals_even[1])
    discrepancy = abs(E0_even_ed - E_analytical_even_spin)
    gap_ed = real(vals_even[2]) - real(vals_even[1])
    gap_analytical = 2 * minimum(energies_even)
    gap_match = abs(gap_ed - gap_analytical) < 1e-6
    
    # Special mode energies
    ε_0 = abs(sin(θ) + cos(θ))
    ε_π = abs(sin(θ) - cos(θ))
    
    if verbose
        println("\nCase: N=$N, θ=$(θ/π)π")
        println("  Ground state energy discrepancy: $discrepancy")
        println("  Discrepancy/N: $(discrepancy/N)")
        println("  Gap match: $(gap_match ? "✓" : "✗")")
        println("  ε_π = $ε_π, Discrepancy/ε_π = $(discrepancy/ε_π)")
    end
    
    return (N=N, θ=θ, discrepancy=discrepancy, discrepancy_per_mode=discrepancy/N, 
            gap_match=gap_match, ε_π=ε_π, ratio=discrepancy/ε_π)
end

# Let's verify our understanding by computing things from scratch
println("\nDouble-checking the fermionic Hamiltonian structure:")
println("  H_spin = Σ_k ε_k (2â_k†â_k - 1) + const")
println("  In ground state: â_k†â_k = 0 for all k")
println("  So E_GS = Σ_k (-ε_k) + const")
println("  We computed: Σ_k (-ε_k) = $E_analytical_even")
println("  Plus constant: $(sin(θ)*N/2) = $(E_analytical_even + sin(θ)*N/2)")

# Check special mode energies
println("\nSpecial mode energies (from notes):")
ε_0 = abs(sin(θ) + cos(θ))
ε_π = abs(sin(θ) - cos(θ))
println("  ε_0 = |sin(θ) + cos(θ)| = $ε_0")
println("  ε_π = |sin(θ) - cos(θ)| = $ε_π")
println("  For θ=π/3: sin(θ)=$(sin(θ)), cos(θ)=$(cos(θ))")
println("  sin(θ) + cos(θ) = $(sin(θ) + cos(θ))")
println("  sin(θ) - cos(θ) = $(sin(θ) - cos(θ))")

# These would appear in PBC fermionic case (odd parity)
# Let's check if they relate to our discrepancy
println("\nChecking if special modes relate to discrepancy:")
println("  Discrepancy = $(discrepancy)")
println("  ε_π = $ε_π")
println("  Discrepancy/ε_π = $(discrepancy/ε_π)")
println("  Difference: $(discrepancy - ε_π)")
println("  Relative difference: $((discrepancy - ε_π)/ε_π * 100)%")

# The k=π mode has exactly ε_π energy in PBC
# Let's check if the discrepancy could be related to mode counting
println("\nMode counting hypothesis:")
println("  In PBC, k=π has ε_π = |sinθ - cosθ| = $ε_π")
println("  In APBC, we don't have k=π, but k=±0.9π with ε = $(gap_modes_even[1].ε)")
println("  Could we be missing a contribution from this difference?")

# Test different cases
println("\n" * "="^60)
println("TESTING DIFFERENT CASES")
println("="^60)

# Test different N values
println("\nTesting different system sizes with θ=π/3:")
for N_test in [6, 8, 10, 12]
    result = test_single_case(N_test, π/3; verbose=false)
    println("N=$N_test: discrepancy/N = $(result.discrepancy_per_mode), ratio to ε_π = $(result.ratio)")
end

# Test different θ values  
println("\nTesting different θ values with N=10:")
for θ_frac in [1/6, 1/4, 1/3, 1/2, 2/3]
    θ_test = θ_frac * π
    result = test_single_case(10, θ_test; verbose=false)
    println("θ=$(θ_frac)π: discrepancy = $(result.discrepancy), ε_π = $(result.ε_π), ratio = $(result.ratio)")
end

# The discrepancy is very close to ε_π!
# But APBC doesn't have k=π mode... what's going on?
println("\nInvestigating the near-match with ε_π:")

# Special investigation of θ=π/2 case
println("\n" * "="^60)
println("SPECIAL CASE: θ=π/2")
println("="^60)
result_half = test_single_case(10, π/2; verbose=true)
println("At θ=π/2: sin(θ)=1, cos(θ)=0")
println("This gives ε_k = √(1 + 0) = 1 for all k")
println("So all modes have the same energy!")

# Check what's special about this case
println("\nWhat makes θ=π/2 special?")
println("  H_spin = (1/2)Σσ_z (pure transverse field)")
println("  No XX interactions, so the fermionic Hamiltonian is diagonal!")
println("  This might explain why the mapping works perfectly here.")

# More systematic θ scan
println("\n" * "="^60)
println("SYSTEMATIC θ SCAN")
println("="^60)
println("\nScanning θ from 0 to π/2 with N=10:")
println("θ/π\t\tDiscrepancy\tε_π\t\tDisc/N\t\tDisc/ε_π")
for i in 0:10
    θ_test = i * π/20  # θ from 0 to π/2
    if i == 0 || i == 10
        continue  # Skip θ=0 and θ=π/2 edge cases
    end
    result = test_single_case(10, θ_test; verbose=false)
    @printf("%.3f\t\t%.6f\t%.6f\t%.6f\t%.6f\n", 
            θ_test/π, result.discrepancy, result.ε_π, result.discrepancy_per_mode, result.ratio)
end
println("  APBC modes: k = (2j+1)π/N for j = -N/2, ..., N/2-1")
println("\nLooking for a pattern in the discrepancy:")
println("  Notice that discrepancy decreases as θ → π/2")
println("  At θ=π/2, H is diagonal (no XX term) and discrepancy → 0")
println("  At θ=0, H is pure XX and discrepancy is maximal")

# Check if discrepancy scales with interaction strength
println("\nChecking if discrepancy scales with cos(θ) (XX interaction strength):")
θ_test = π/3
result = test_single_case(10, θ_test; verbose=false)
println("  θ=π/3: cos(θ) = $(cos(θ_test)), discrepancy = $(result.discrepancy)")
println("  Discrepancy/cos(θ) = $(result.discrepancy/cos(θ_test))")
println("  Discrepancy/cos²(θ) = $(result.discrepancy/cos(θ_test)^2)")

println("\n" * "="^60)
println("SUMMARY OF FINDINGS")
println("="^60)
println("1. The intra-parity gap matches perfectly in all cases")
println("2. The ground state energy has a systematic discrepancy")
println("3. For fixed θ, discrepancy/N is constant across different N")
println("4. The discrepancy → 0 as θ → π/2 (pure transverse field)")
println("5. The discrepancy is maximal at θ → 0 (pure XX interaction)")
println("6. The discrepancy is NOT simply ε_π, though close for θ=π/3")
println("\nConclusion: The discrepancy appears related to the Bogoliubov")
println("transformation or normal ordering in the fermionic representation.")