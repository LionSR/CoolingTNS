using CoolingTNS
using LinearAlgebra

# System parameters matching the test
N = 4
J = 1.0
h = 0.5

# Create Hamiltonian parameters
ham_params = HamiltonianParameters(
    model=IsingModel(),
    J=J,
    h=h,
    hz=0.0
)

# Create ED backend
backend = EDBackend()

# Construct the system Hamiltonian
H_sys = construct_system_hamiltonian(ham_params, backend, N)

# Find ground state and compute gap
E0, gs = find_ground_state(H_sys, backend)

# Get the full eigenvalue spectrum
eigenvalues = eigvals(Matrix(H_sys))
sort!(eigenvalues)

# Compute the gap
gap = eigenvalues[2] - eigenvalues[1]

println("Ising system parameters:")
println("  N = $N")
println("  J = $J")
println("  h = $h")
println("\nResults:")
println("  Ground state energy: $E0")
println("  First excited state energy: $(eigenvalues[2])")
println("  Energy gap: $gap")
println("\nLowest 5 eigenvalues:")
for i in 1:min(5, length(eigenvalues))
    println("  E[$i] = $(eigenvalues[i])")
end

# Also check the Hamiltonian structure
println("\nHamiltonian size: $(size(H_sys))")
println("Is Hermitian: $(ishermitian(H_sys))")
println("Hamiltonian type: $(typeof(H_sys))")