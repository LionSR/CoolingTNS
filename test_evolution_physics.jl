#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays

# Include necessary files
include("src/parameter_types.jl")
include("src/cooling_types.jl")
include("src/ed_backend.jl")
include("src/system_hamiltonian.jl")
include("src/system_bath_hamiltonian.jl")

# Test simple 2-qubit evolution
println("=== Testing Evolution Physics ===")

# Create a simple 2-qubit system: 1 system qubit + 1 bath qubit
N = 1  # 1 system qubit means 2 total qubits

# System Hamiltonian: Just a field on the system qubit
# H_sys = h * X_1
h = 2.0
H_sys = h * pauli_x(1, 1)
println("System Hamiltonian (1 qubit):")
println(Matrix(H_sys))

# Find ground state of system
vals_sys, vecs_sys = eigen(Matrix(H_sys))
E0_sys = vals_sys[1]
ψ0_sys = vecs_sys[:, 1]
println("\nSystem ground state energy: $E0_sys")
println("System ground state: $ψ0_sys")

# System+bath Hamiltonian
# Layout: qubit 1 = system, qubit 2 = bath
# H_sb = H_sys ⊗ I + I ⊗ (Δ/2 * Z) + g * X_sys ⊗ X_bath

# First, embed system Hamiltonian in 2-qubit space
H_sys_full = kron(Matrix(H_sys), I(2))

# Bath Hamiltonian: Δ/2 * Z on bath qubit
Δ = -3.0  # Negative for cooling
H_bath = kron(I(2), Δ/2 * [1 0; 0 -1])

# Coupling: g * X_sys ⊗ X_bath
g = 0.1
H_coupling = g * kron([0 1; 1 0], [0 1; 1 0])

# Total Hamiltonian
H_total = H_sys_full + H_bath + H_coupling

println("\n\nTotal Hamiltonian (2 qubits):")
println(Matrix(H_total))

# Check eigenvalues
vals_total, vecs_total = eigen(Matrix(H_total))
println("\nEigenvalues of total Hamiltonian:")
for (i, E) in enumerate(vals_total)
    println("  E$i = $E")
end

# Initial state: system in excited state |1⟩, bath in ground state |0⟩
# This gives |10⟩ in the computational basis
ψ_sys_init = [0.0, 1.0]  # Excited state of system
ψ_bath_init = [1.0, 0.0]  # Ground state of bath (|0⟩ has lower energy when Δ < 0)
ψ_init = kron(ψ_sys_init, ψ_bath_init)
println("\n\nInitial state: |ψ⟩ = |1⟩_sys ⊗ |0⟩_bath")
println("Initial state vector: $ψ_init")

# Compute initial energy
E_init = real(ψ_init' * H_total * ψ_init)
println("\nInitial energy: $E_init")
println("Initial system energy: $(real(ψ_sys_init' * Matrix(H_sys) * ψ_sys_init))")

# Time evolution
t = 0.5
println("\n\nTime evolution for t = $t")

# Exact evolution: |ψ(t)⟩ = exp(-i H t) |ψ(0)⟩
U = exp(-im * t * H_total)
ψ_final = U * ψ_init

# Trace out bath to get system state
ρ_total_final = ψ_final * ψ_final'
ρ_sys_final = zeros(ComplexF64, 2, 2)
ρ_sys_final[1,1] = ρ_total_final[1,1] + ρ_total_final[3,3]  # |0⟩⟨0|
ρ_sys_final[1,2] = ρ_total_final[1,2] + ρ_total_final[3,4]  # |0⟩⟨1|
ρ_sys_final[2,1] = ρ_total_final[2,1] + ρ_total_final[4,3]  # |1⟩⟨0|
ρ_sys_final[2,2] = ρ_total_final[2,2] + ρ_total_final[4,4]  # |1⟩⟨1|

# Compute final system energy
E_sys_final = real(tr(H_sys * ρ_sys_final))
println("Final system energy: $E_sys_final")
println("Energy change: $(E_sys_final - real(ψ_sys_init' * Matrix(H_sys) * ψ_sys_init))")

# Check if cooling is happening
if E_sys_final < real(ψ_sys_init' * Matrix(H_sys) * ψ_sys_init)
    println("\n✅ System is cooling!")
else
    println("\n❌ System is heating!")
end

# Now test with density matrix evolution using evolve_ed
println("\n\n=== Testing evolve_ed function ===")

# Create initial density matrix
ρ_init = EDDensityMatrix(ψ_init * ψ_init', 2)
println("Initial density matrix:")
println(ρ_init.data)

# Evolve using evolve_ed
ρ_evolved = evolve_ed(H_total, ρ_init, t)
println("\nEvolved density matrix:")
println(ρ_evolved.data)

# Trace out bath
ρ_sys_ed = zeros(Float64, 2, 2)
ρ_sys_ed[1,1] = ρ_evolved.data[1,1] + ρ_evolved.data[3,3]
ρ_sys_ed[1,2] = ρ_evolved.data[1,2] + ρ_evolved.data[3,4]
ρ_sys_ed[2,1] = ρ_evolved.data[2,1] + ρ_evolved.data[4,3]
ρ_sys_ed[2,2] = ρ_evolved.data[2,2] + ρ_evolved.data[4,4]

E_sys_ed = tr(Matrix(H_sys) * ρ_sys_ed)
println("\nSystem energy after evolve_ed: $E_sys_ed")

if abs(E_sys_ed - E_sys_final) < 1e-10
    println("✅ evolve_ed gives correct result!")
else
    println("❌ evolve_ed gives different result: $(E_sys_ed) vs $(E_sys_final)")
end