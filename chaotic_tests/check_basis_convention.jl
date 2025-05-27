using CoolingTNS
using LinearAlgebra

# Check basis convention
println("Basis convention check:")
println("======================")

# In computational basis: |0⟩ = spin up, |1⟩ = spin down
# σ^z|0⟩ = +|0⟩, σ^z|1⟩ = -|1⟩

# Check Pauli matrices
σx = CoolingTNS.pauli_x(1, 1)
σy = CoolingTNS.pauli_y(1, 1) 
σz = CoolingTNS.pauli_z(1, 1)

println("Pauli X:")
display(Matrix(σx))
println("\nPauli Y:")
display(Matrix(σy))
println("\nPauli Z:")
display(Matrix(σz))

# Check what the Jordan-Wigner transformation gives us
println("\n\nJordan-Wigner for single qubit:")
a, a_dag = CoolingTNS.jordan_wigner_transform(1, 1)

println("\nAnnihilation operator a:")
display(Matrix(a))
println("\nCreation operator a†:")
display(Matrix(a_dag))

# The standard JW transformation should be:
# a = (σ^x + i·σ^y)/2 = σ^+
# a† = (σ^x - i·σ^y)/2 = σ^-

# Check what these do to basis states
up = [1.0 + 0im, 0.0 + 0im]  # |0⟩ = |↑⟩
down = [0.0 + 0im, 1.0 + 0im]  # |1⟩ = |↓⟩

println("\n\nAction on basis states:")
println("a|↑⟩ = ", a * up)
println("Should map |↑⟩ → |↓⟩")
println("a|↓⟩ = ", a * down)
println("Should be zero")

println("\na†|↑⟩ = ", a_dag * up)
println("Should be zero")
println("a†|↓⟩ = ", a_dag * down)
println("Should map |↓⟩ → |↑⟩")

# It looks like our JW maps are backwards!
# Let's check the number operator
n_op = a_dag * a
println("\n\nNumber operator n = a†a:")
display(Matrix(n_op))

println("\nn|↑⟩ = ", n_op * up, " (expect 0)")
println("n|↓⟩ = ", n_op * down, " (expect 1)")

# The issue seems to be that our fermion operators are defined for spin-up fermions
# In the standard mapping: |↑⟩ = vacuum (no fermion), |↓⟩ = occupied (one fermion)