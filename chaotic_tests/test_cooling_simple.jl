using CoolingTNS
using LinearAlgebra

# Test parameters matching the failing test
N = 3
ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5)

backend = CoolingTNS.EDBackend()
sim_params = CoolingTNS.UnifiedSimulationParameters(
    CoolingTNS.DensityMatrix(),
    CoolingTNS.ContinuousEvolution();
    pe=0.0
)

coupling_params = CoolingTNS.BasicCouplingParameters(
    "XX",    # coupling
    0.1,     # g
    2,       # steps
    1.0,     # te
    nothing  # delta (auto-compute)
)

problem_setup = CoolingTNS.setup_problem(
    backend, ham_params, coupling_params, sim_params
)

initial_state = CoolingTNS.setup_initial_state(
    problem_setup, sim_params, "product", 0.0
)

# Check initial state energy manually
H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, backend, N)
H_sys_mat = CoolingTNS.mat(ComplexF64, H_sys)
ρ_sys_initial = initial_state.state

println("Initial density matrix size: ", size(ρ_sys_initial))
println("System Hamiltonian size: ", size(H_sys_mat))

E_initial = real(tr(H_sys_mat * ρ_sys_initial))
println("Manual initial energy calculation: ", E_initial)
println("Manual initial energy per site: ", E_initial/N)

# Check the product state configuration
println("\nInitial density matrix diagonal: ", diag(ρ_sys_initial))

# Run one step of cooling
results = CoolingTNS.run_cooling(
    problem_setup,
    initial_state,
    coupling_params,
    sim_params,
    ham_params
)

println("\nEnergy evolution:")
for (i, E) in enumerate(results["E_list"])
    println("Step $(i-1): E = $E, E/N = $(E/N)")
end

println("\nEnergy change: ", results["E_list"][end] - results["E_list"][1])