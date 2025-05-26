#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays
using ITensors
using ITensorMPS

# Include necessary files
include("src/CoolingTNS.jl")
using .CoolingTNS

# Test with small system to compare backends exactly
N = 2  # 2 system qubits, 2 bath qubits, 4 total
println("=== Comparing TN and ED Backends ===")
println("System: N=$N ($(2*N) total qubits)")
println("Method: Density Matrix + Continuous Evolution")

# Common parameters
base_args = [
    "--N", "$N",
    "--problem", "Ising",
    "--coupling", "XX",
    "--g", "0.1",
    "--te", "0.5",
    "--steps", "1",
    "--sim_method", "density_matrix",
    "--evolution_method", "continuous"
]

# Setup ED backend
println("\n--- ED Backend Setup ---")
args_ed = vcat(base_args, ["--backend", "ED"])
parsed_ed = CoolingTNS.parse_commandline(args_ed)

# Setup parameters manually to inspect
ham_params = CoolingTNS.HamiltonianParameters(
    CoolingTNS.IsingModel(), 
    N, 
    (J=1.0, h=2.0)
)

# First check system Hamiltonian
println("\nSystem Hamiltonians:")

# ED system Hamiltonian
H_sys_ed = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
println("ED System Hamiltonian:")
display(Matrix(H_sys_ed))

# TN system Hamiltonian (convert to matrix for comparison)
sites_sys = siteinds("S=1/2", N)
H_sys_tn_mpo = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.TNBackend(), sites_sys)
# Convert MPO to matrix
H_sys_tn_matrix = zeros(2^N, 2^N)
for i in 1:2^N, j in 1:2^N
    # Create basis states
    ψ_i = productMPS(sites_sys, [((i-1) >> (k-1)) & 1 == 0 ? "Up" : "Dn" for k in 1:N])
    ψ_j = productMPS(sites_sys, [((j-1) >> (k-1)) & 1 == 0 ? "Up" : "Dn" for k in 1:N])
    H_sys_tn_matrix[i,j] = real(inner(ψ_i', H_sys_tn_mpo, ψ_j))
end
println("\nTN System Hamiltonian (as matrix):")
display(H_sys_tn_matrix)

println("\nDifference between ED and TN system Hamiltonians:")
display(Matrix(H_sys_ed) - H_sys_tn_matrix)
println("Max difference: $(maximum(abs.(Matrix(H_sys_ed) - H_sys_tn_matrix)))")

# Check eigenvalues
vals_ed = eigvals(Matrix(H_sys_ed))
vals_tn = eigvals(H_sys_tn_matrix)
println("\nSystem eigenvalues:")
println("ED: $vals_ed")
println("TN: $vals_tn")

# Find ground state and gap
e0_ed, ψ0_ed, gap_ed = CoolingTNS.find_ground_state(H_sys_ed, CoolingTNS.EDBackend())
println("\nED ground state energy: $e0_ed, gap: $gap_ed")

e0_tn, ψ0_tn, gap_tn = CoolingTNS.find_ground_state(H_sys_tn_mpo, CoolingTNS.TNBackend(), sites_sys)
println("TN ground state energy: $e0_tn, gap: $gap_tn")

# Setup coupling parameters with resonant cooling
Δ = -gap_ed  # Should be negative for cooling
coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.1, 1, 0.5, Δ)

println("\nResonant frequency Δ = $Δ")
println("Bath energy per qubit: $(Δ/2)")

# Now check system+bath Hamiltonians
println("\n\n=== System+Bath Hamiltonians ===")

# ED system+bath
H_sb_ed = CoolingTNS.construct_system_bath_hamiltonian(ham_params, CoolingTNS.EDBackend(), 2*N, coupling_params)
println("ED System+Bath Hamiltonian size: $(size(H_sb_ed))")
println("Is real: $(isreal(H_sb_ed))")
println("Is symmetric: $(issymmetric(H_sb_ed))")

# TN system+bath
sites_full = siteinds("S=1/2", 2*N)
H_sb_tn = CoolingTNS.construct_system_bath_hamiltonian(ham_params, CoolingTNS.TNBackend(), sites_full, coupling_params)

# Convert TN MPO to matrix for comparison (only for small systems!)
H_sb_tn_matrix = zeros(2^(2*N), 2^(2*N))
for i in 1:2^(2*N), j in 1:2^(2*N)
    # Create basis states with alternating system/bath layout
    config_i = [(((i-1) >> (k-1)) & 1 == 0) ? "Up" : "Dn" for k in 1:2*N]
    config_j = [(((j-1) >> (k-1)) & 1 == 0) ? "Up" : "Dn" for k in 1:2*N]
    ψ_i = productMPS(sites_full, config_i)
    ψ_j = productMPS(sites_full, config_j)
    H_sb_tn_matrix[i,j] = real(inner(ψ_i', H_sb_tn, ψ_j))
end

println("\nTN System+Bath as matrix - first 8x8 block:")
display(H_sb_tn_matrix[1:8, 1:8])

println("\nED System+Bath - first 8x8 block:")
display(Matrix(H_sb_ed)[1:8, 1:8])

# Check specific matrix elements to understand the difference
println("\n\nChecking bath energy terms:")
println("Expected bath energy on site 2: $(Δ/2)")
println("Expected bath energy on site 4: $(Δ/2)")

# Check lowest eigenvalues
vals_sb_ed = eigvals(Matrix(H_sb_ed))[1:8]
vals_sb_tn = eigvals(H_sb_tn_matrix)[1:8]
println("\nLowest 8 eigenvalues:")
println("ED: $vals_sb_ed")
println("TN: $vals_sb_tn")
println("Difference: $(vals_sb_ed - vals_sb_tn)")

# Now test evolution
println("\n\n=== Testing Evolution ===")

# Initial state: product state |0000⟩ for both backends
# ED: Create density matrix
ρ_init_ed = zeros(Float64, 4, 4)
ρ_init_ed[1,1] = 1.0  # |00⟩⟨00| for system only
ρ_init_ed_full = kron(ρ_init_ed, zeros(Float64, 4, 4))
ρ_init_ed_full[1,1] = 1.0  # |0000⟩⟨0000|

# Measure initial system energy
ρ_sys_init = zeros(Float64, 4, 4)
ρ_sys_init[1,1] = 1.0
E_init = real(tr(Matrix(H_sys_ed) * ρ_sys_init))
println("Initial system energy: $E_init")

# Evolve with ED backend
ρ_ed = CoolingTNS.EDDensityMatrix(ρ_init_ed_full, 2*N)
ρ_evolved_ed = CoolingTNS.evolve_ed(H_sb_ed, ρ_ed, 0.5)

# Trace out bath to get system density matrix
ρ_sys_final_ed = CoolingTNS.trace_out_bath_ed(ρ_evolved_ed, N)
E_final_ed = CoolingTNS.expect_ed(H_sys_ed, ρ_sys_final_ed)
println("\nED final system energy: $E_final_ed")
println("ED energy change: $(E_final_ed - E_init)")

# For TN, we need to run through the full machinery
# ... (TN comparison would be more complex due to the MPS structure)

if E_final_ed > E_init
    println("\n❌ ED Backend is HEATING!")
else
    println("\n✅ ED Backend is cooling")
end