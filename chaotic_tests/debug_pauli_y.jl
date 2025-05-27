using LinearAlgebra

# Standard complex Pauli matrices
σx = [0 1; 1 0]
σy_complex = [0 -im; im 0]
σz = [1 0; 0 -1]

# Our real representation
σy_real = [0 -1; 1 0]

println("Standard complex Pauli Y:")
display(σy_complex)

println("\n\nOur real Pauli Y:")
display(σy_real)

# Standard operators
σ_plus = (σx + im*σy_complex)/2
σ_minus = (σx - im*σy_complex)/2

println("\n\nStandard σ^+:")
display(σ_plus)
println("\nStandard σ^-:")
display(σ_minus)

# With our real Y, we need to be careful
# If Y_real = -i*Y_complex, then:
# σ^+ = (X + iY)/2 = (X - i*(-i*Y_real))/2 = (X - Y_real)/2
a_our = (σx - σy_real)/2
a_dag_our = (σx + σy_real)/2

println("\n\nOur a (should match σ^+):")
display(a_our)
println("\nOur a† (should match σ^-):")
display(a_dag_our)

# Test on basis states
up = [1, 0]
down = [0, 1]

println("\n\nStandard convention test:")
println("σ^+|↑⟩ = ", σ_plus * up, " (should be 0)")
println("σ^+|↓⟩ = ", σ_plus * down, " (should be |↑⟩)")
println("σ^-|↑⟩ = ", σ_minus * up, " (should be |↓⟩)")
println("σ^-|↓⟩ = ", σ_minus * down, " (should be 0)")

println("\n\nOur operators (real representation):")
println("a|↑⟩ = ", a_our * up, " (should be 0)")
println("a|↓⟩ = ", a_our * down, " (should be |↑⟩)")
println("a†|↑⟩ = ", a_dag_our * up, " (should be |↓⟩)")
println("a†|↓⟩ = ", a_dag_our * down, " (should be 0)")

# Number operator
n_standard = σ_minus * σ_plus
n_our = a_dag_our * a_our

println("\n\nNumber operators:")
println("Standard n = σ^- σ^+:")
display(n_standard)
println("\nOur n = a† a:")
display(n_our)