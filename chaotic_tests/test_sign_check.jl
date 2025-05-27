using CoolingTNS

# Test a simple 2-spin Ising system
N = 2
J = 1.0
h = -2.0

# Create Hamiltonian parameters
ham_params = IsingParameters(N, J, h)

# Build system Hamiltonian for both backends
backend_tn = TNBackend()
backend_ed = EDBackend()

# For TN, we need sites
sites_tn = siteinds("Qubit", N)
H_tn = construct_system_hamiltonian(ham_params, backend_tn, sites_tn)

# For ED
H_ed = construct_system_hamiltonian(ham_params, backend_ed, nothing)

println("ED Hamiltonian:")
display(Matrix(H_ed))
println()

# Check energies
using LinearAlgebra
evals_ed = eigvals(Matrix(H_ed))
println("ED eigenvalues: ", sort(evals_ed))

# For TN, we need to convert to matrix for comparison
# This is tricky with ITensors, so let's just check the ground state energy
using ITensorMPS
dmrg_result = dmrg(H_tn, MPS(sites_tn, "Up"); nsweeps=5, cutoff=1e-10, outputlevel=0)
E0_tn = dmrg_result[1]
println("TN ground state energy from DMRG: ", E0_tn)
println("ED ground state energy: ", minimum(evals_ed))

# Now test evolution
println("\nTesting evolution sign:")

# Initial state |00⟩
ψ0_ed = zero_state_ed(N)
println("Initial state energy (ED): ", expect_ed(H_ed, ψ0_ed))

# Evolve for small time
t = 0.1
ψ_evolved = evolve_ed(H_ed, ψ0_ed, t)
E_after = expect_ed(H_ed, ψ_evolved)
println("Energy after evolution (ED): ", E_after)

# For cooling, energy should decrease toward ground state
E0 = minimum(evals_ed)
E_init = expect_ed(H_ed, ψ0_ed)
println("\nInitial energy: ", E_init)
println("Ground state energy: ", E0)
println("Energy after evolution: ", E_after)
println("Did energy move toward ground state? ", abs(E_after - E0) < abs(E_init - E0))