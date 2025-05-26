using CoolingTNS
using LinearAlgebra

# Parameters
N = 4
J = 1.0
h = 2.0
bc = :periodic

# Create Hamiltonian
ham_params = CoolingTNS.HamiltonianParameters(
    CoolingTNS.IsingModel(),
    N,
    (J=J, h=h),
    bc
)

H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)

# Find ground state
E0, ψ0 = CoolingTNS.find_ground_state(H_sys, CoolingTNS.EDBackend())

println("Ground state analysis:")
println("======================")
println("E0/N = ", E0/N)
println("Ground state vector (first 16 components):")
for i in 1:min(16, length(ψ0.data))
    if abs(ψ0.data[i]) > 1e-10
        # Convert index to binary representation
        config = string(i-1, base=2, pad=N)
        println("  |$config⟩: ", round(ψ0.data[i], digits=4))
    end
end

# Check what state this corresponds to
println("\nState interpretation:")
println("Index 1 (|0000⟩) amplitude: ", round(abs(ψ0.data[1]), digits=4))
println("This is the all-down state in spin language")
println("Or equivalently, the vacuum state in fermion language")

# Verify by computing magnetization
mag_z = 0.0
for i in 1:N
    Z_i = CoolingTNS.pauli_z(i, N)
    global mag_z += real(dot(ψ0.data, Z_i * ψ0.data))
end
println("\nTotal magnetization ⟨Σ Z_i⟩ = ", mag_z)
println("Per-site magnetization ⟨Z⟩ = ", mag_z/N)

# Compute n_k to verify
k_values, n_k = CoolingTNS.measure_momentum_distribution_ed(ψ0, ham_params)
println("\nMomentum distribution n_k:")
for (k, nk) in zip(k_values, n_k)
    println("  k/π = $(round(k/π, digits=3)): n_k = $(round(nk, digits=6))")
end
println("Sum of n_k = ", sum(n_k), " (should equal number of fermions)")

# Let's also check a different initial state for comparison
println("\n\nFor comparison, product state |↑↑↑↑⟩:")
ψ_up = CoolingTNS.zero_state_ed(N)  # This is actually all up
k_values_up, n_k_up = CoolingTNS.measure_momentum_distribution_ed(ψ_up, ham_params)
println("Momentum distribution n_k:")
for (k, nk) in zip(k_values_up, n_k_up)
    println("  k/π = $(round(k/π, digits=3)): n_k = $(round(nk, digits=6))")
end