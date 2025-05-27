using CoolingTNS
using LinearAlgebra

# Clear the JW cache
empty!(CoolingTNS.JW_CACHE)
empty!(CoolingTNS.CORRELATION_OP_CACHE)

# Test the fixed Jordan-Wigner
println("Testing fixed Jordan-Wigner transformation:")
println("==========================================")

# Single qubit test
a, a_dag = CoolingTNS.jordan_wigner_transform(1, 1)

up = [1.0 + 0im, 0.0 + 0im]  # |↑⟩ = vacuum
down = [0.0 + 0im, 1.0 + 0im]  # |↓⟩ = occupied

println("a|↑⟩ = ", a * up, " (should be zero)")
println("a|↓⟩ = ", a * down, " (should be |↑⟩)")
println("a†|↑⟩ = ", a_dag * up, " (should be |↓⟩)")
println("a†|↓⟩ = ", a_dag * down, " (should be zero)")

# Number operator
n_op = a_dag * a
println("\nNumber operator n = a†a:")
display(Matrix(n_op))
println("\nn|↑⟩ = ", n_op * up, " (expect 0)")
println("n|↓⟩ = ", n_op * down, " (expect |↓⟩)")

# Test on 4-qubit states
N = 4
println("\n\n4-qubit tests:")

# All down |↓↓↓↓⟩ = |1111⟩ = 4 fermions
ψ_down = zeros(ComplexF64, 2^N)
ψ_down[16] = 1.0
ψ_down_state = CoolingTNS.EDStateVector(ψ_down, N)

println("\nAll down |↓↓↓↓⟩ (4 fermions):")
for i in 1:N
    a, a_dag = CoolingTNS.jordan_wigner_transform(i, N)
    n_i = real(dot(ψ_down, a_dag * a * ψ_down))
    println("  n_$i = $n_i")
end

# Compute ground state again
println("\n\nTransverse field Ising ground state:")
ham_params = CoolingTNS.HamiltonianParameters(
    CoolingTNS.IsingModel(),
    N,
    (J=1.0, h=2.0),
    :periodic
)
H = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
E0, ψ0 = CoolingTNS.find_ground_state(H, CoolingTNS.EDBackend())

total_n = 0.0
for i in 1:N
    a, a_dag = CoolingTNS.jordan_wigner_transform(i, N)
    n_i = real(dot(ψ0.data, a_dag * a * ψ0.data))
    println("  n_$i = $n_i")
    global total_n += n_i
end
println("Total fermion number = $total_n")

# Now compute momentum distribution
k_values, n_k = CoolingTNS.measure_momentum_distribution_ed(ψ0, ham_params)
println("\nMomentum distribution:")
for (k, nk) in zip(k_values, n_k)
    println("  k/π = $(round(k/π, digits=3)): n_k = $(round(nk, digits=3))")
end