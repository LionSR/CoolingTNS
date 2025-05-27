using CoolingTNS
using LinearAlgebra

# Test Jordan-Wigner on simple states
N = 4

# Test 1: All spins up |↑↑↑↑⟩ = |0000⟩ in computational basis
println("Test 1: All spins up |↑↑↑↑⟩")
ψ_up = zeros(ComplexF64, 2^N)
ψ_up[1] = 1.0  # |0000⟩
ψ_up_state = CoolingTNS.EDStateVector(ψ_up, N)

# In spin language: σ^+ |↑⟩ = 0, σ^- |↑⟩ = |↓⟩
# After JW: a |vacuum⟩ = 0
# So we expect n_k = 0 for all k

# Test 2: One spin flipped |↓↑↑↑⟩ = |1000⟩ 
println("\nTest 2: One spin down |↓↑↑↑⟩")
ψ_one = zeros(ComplexF64, 2^N)
ψ_one[9] = 1.0  # |1000⟩ = 2^3 + 1 = 9
ψ_one_state = CoolingTNS.EDStateVector(ψ_one, N)

# Test 3: All spins down |↓↓↓↓⟩ = |1111⟩
println("\nTest 3: All spins down |↓↓↓↓⟩")
ψ_down = zeros(ComplexF64, 2^N)
ψ_down[16] = 1.0  # |1111⟩
ψ_down_state = CoolingTNS.EDStateVector(ψ_down, N)

# This should have 4 fermions

# Test the Jordan-Wigner operators directly
println("\nTesting Jordan-Wigner operators:")
for i in 1:N
    a, a_dag = CoolingTNS.jordan_wigner_transform(i, N)
    
    # Check on vacuum state (all up)
    result_a = a * ψ_up
    result_adag = a_dag * ψ_up
    
    println("Site $i:")
    println("  a|↑↑↑↑⟩ norm = ", norm(result_a), " (should be 0)")
    println("  a†|↑↑↑↑⟩ norm = ", norm(result_adag), " (should be 1)")
    
    # Check what state a† creates
    if norm(result_adag) > 0.1
        idx = findfirst(x -> abs(x) > 0.1, result_adag)
        config = string(idx-1, base=2, pad=N)
        println("  a†|↑↑↑↑⟩ creates |$config⟩")
    end
end

# Now check n_i = a†_i a_i for different states
println("\n\nOccupation numbers n_i = ⟨a†_i a_i⟩:")
states = [("All up", ψ_up_state), ("One down", ψ_one_state), ("All down", ψ_down_state)]

for (name, ψ) in states
    println("\n$name:")
    for i in 1:N
        a, a_dag = CoolingTNS.jordan_wigner_transform(i, N)
        n_i = real(dot(ψ.data, a_dag * a * ψ.data))
        println("  n_$i = $n_i")
    end
end

# Check the transverse field Ising ground state
println("\n\nTransverse field Ising ground state (J=1, h=2):")
ham_params = CoolingTNS.HamiltonianParameters(
    CoolingTNS.IsingModel(),
    N,
    (J=1.0, h=2.0),
    :periodic
)
H = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
E0, ψ0 = CoolingTNS.find_ground_state(H, CoolingTNS.EDBackend())

for i in 1:N
    a, a_dag = CoolingTNS.jordan_wigner_transform(i, N)
    n_i = real(dot(ψ0.data, a_dag * a * ψ0.data))
    println("  n_$i = $n_i")
end

# Total number of fermions
total_n = 0.0
for i in 1:N
    a, a_dag = CoolingTNS.jordan_wigner_transform(i, N)
    global total_n += real(dot(ψ0.data, a_dag * a * ψ0.data))
end
println("Total fermion number = $total_n")