using ITensors
using ITensorMPS
using LinearAlgebra
using SparseArrays
using KrylovKit

# Include necessary files
include("src/parameter_types.jl")
include("src/cooling_types.jl")
include("src/system_hamiltonian.jl")
include("src/ed_backend.jl")
include("src/ground_state.jl")
include("src/system_bath_hamiltonian.jl")
include("src/setup_system.jl")

# Test system parameters
N = 2  # Very small system for debugging
J = 1.0
h = 2.0

# Create parameters
ham_params = HamiltonianParameters(IsingModel(), N, (J=J, h=h))

# Test both backends
println("=== Testing Ising Model Ground State ===")
println("Parameters: N=$N, J=$J, h=$h")

# ED Backend
println("\n--- ED Backend ---")
H_sys_ed = construct_system_hamiltonian(ham_params, EDBackend(), N)
println("System Hamiltonian (ED):")
println(Matrix(H_sys_ed))

e0_ed, ψ0_ed, gap_ed = find_ground_state(H_sys_ed, EDBackend())
println("\nGround state energy: $e0_ed")
println("Energy gap: $gap_ed")
println("Resonant frequency Δ = -gap = $(-gap_ed)")

# Check all eigenvalues to understand the spectrum
vals, vecs = eigen(Matrix(H_sys_ed))
println("\nFull spectrum:")
for (i, E) in enumerate(vals)
    println("  E$i = $E")
end

# TN Backend
println("\n--- TN Backend ---")
sites_sys = siteinds("S=1/2", N)
H_sys_tn = construct_system_hamiltonian(ham_params, TNBackend(), sites_sys)

e0_tn, ψ0_tn, gap_tn = find_ground_state(H_sys_tn, TNBackend(), sites_sys)
println("\nGround state energy: $e0_tn")
println("Energy gap: $gap_tn")
println("Resonant frequency Δ = -gap = $(-gap_tn)")

# Now test with system+bath
println("\n\n=== Testing System+Bath Hamiltonian ===")
coupling_params = BasicCouplingParameters("XX", 0.1, 1, 1.0, nothing)  # Will use resonant cooling

# ED System+Bath
println("\n--- ED System+Bath ---")
# First get the resonant frequency
H_sys_ed, Δ_ed, e0_ed, ψ0_ed = setup_system(ham_params, EDBackend())
println("Computed resonant frequency Δ = $Δ_ed")

# Update coupling params with computed delta
coupling_params_ed = BasicCouplingParameters("XX", 0.1, 1, 1.0, Δ_ed)
H_sb_ed = construct_system_bath_hamiltonian(ham_params, EDBackend(), 2*N, coupling_params_ed)

println("\nSystem+Bath Hamiltonian size: $(size(H_sb_ed))")
println("Bath energy terms (Δ/2 * Z on bath qubits):")
println("  Bath qubit 1 (position 2): $(Δ_ed/2)")
println("  Bath qubit 2 (position 4): $(Δ_ed/2)")

# Check lowest eigenvalues of full system
vals_sb, _ = eigen(Matrix(H_sb_ed))
println("\nLowest eigenvalues of system+bath:")
for i in 1:min(8, length(vals_sb))
    println("  E$i = $(vals_sb[i])")
end

# TN System+Bath
println("\n--- TN System+Bath ---")
sites_full = siteinds("S=1/2", 2*N)
sites_sys_tn = sites_full[1:2:2*N-1]
H_sys_tn, Δ_tn, e0_tn, ψ0_tn = setup_system(ham_params, TNBackend(), sites_sys_tn)
println("Computed resonant frequency Δ = $Δ_tn")

coupling_params_tn = BasicCouplingParameters("XX", 0.1, 1, 1.0, Δ_tn)
H_sb_tn = construct_system_bath_hamiltonian(ham_params, TNBackend(), sites_full, coupling_params_tn)

println("\nBath energy terms (Δ/2 * Z on bath qubits):")
println("  Bath qubit 1 (position 2): $(Δ_tn/2)")
println("  Bath qubit 2 (position 4): $(Δ_tn/2)")