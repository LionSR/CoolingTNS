using CoolingTNS

# Test parameters
N = 3
ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5)

backend = CoolingTNS.EDBackend()
sim_params = CoolingTNS.UnifiedSimulationParameters(
    CoolingTNS.DensityMatrix(),
    CoolingTNS.ContinuousEvolution()
)

# With auto-computed delta
coupling_params_auto = CoolingTNS.BasicCouplingParameters(
    "XX", 0.1, 2, 1.0, nothing
)

problem_auto = CoolingTNS.setup_problem(
    backend, ham_params, coupling_params_auto, sim_params
)

println("Auto-computed delta: ", problem_auto.extra.coupling_params.delta)
println("Ground state energy: ", problem_auto.e₀)

# Check the system Hamiltonian eigenvalues
H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, backend, N)
H_sys_mat = CoolingTNS.mat(ComplexF64, H_sys)
eigvals_sys = eigvals(Matrix(H_sys_mat))
println("\nSystem eigenvalues: ", sort(real.(eigvals_sys)))
println("System gap (E1 - E0): ", real(eigvals_sys[2]) - real(eigvals_sys[1]))

# Check bath energies with this delta
delta = problem_auto.extra.coupling_params.delta
println("\nBath |0⟩ energy: ", delta/2)
println("Bath |1⟩ energy: ", -delta/2)
println("Total bath excited energy (3 spins): ", -3*delta/2)