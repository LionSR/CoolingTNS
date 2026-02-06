using CoolingTNS
using CoolingTNS: prepare_combined_state_ed, evolve_ed, measure_ed!, process_bath_ed_monte_carlo
using LinearAlgebra

# Continue from previous test
backend = CoolingTNS.EDBackend()
ham_params = CoolingTNS.IsingParameters(2, 1.0, 1.0)
coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.3, 1, 2.0, nothing)
sim_params = CoolingTNS.UnifiedSimulationParameters(CoolingTNS.MonteCarloWavefunction(), CoolingTNS.ContinuousEvolution())

problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
H_sb = problem.H_sys_bath
H_sys = problem.H_sys

# Initial system: |↑↑⟩
psi_sys = CoolingTNS.product_state_ed(2, 3)
E_sys_init = CoolingTNS.expect_ed(H_sys, psi_sys)
println("Initial system energy: ", E_sys_init)

# Combined state
psi_comb = prepare_combined_state_ed(psi_sys, 2, "XX")

# Evolve
psi_evolved = evolve_ed(H_sb, psi_comb, 2.0)
println("\nEvolved state non-zero elements:")
for (i, v) in enumerate(psi_evolved.data)
    if abs(v) > 0.01
        println("  Index $i (binary $(bitstring(i-1)[end-3:end])): ", v)
    end
end

# Now test bath measurement
println("\n=== Testing bath measurement ===")
println("Bath qubit positions (1-indexed): [2, 4]")

# Process bath and get system state
psi_sys_after, bath_outcomes = process_bath_ed_monte_carlo(psi_evolved, 2)
println("Bath outcomes: ", bath_outcomes)
println("System state after measurement:")
println("  n_qubits: ", psi_sys_after.n_qubits)
println("  data: ", psi_sys_after.data)

# Check system energy
E_sys_after = CoolingTNS.expect_ed(H_sys, psi_sys_after)
println("\nSystem energy after measurement: ", E_sys_after)
println("Energy change: ", E_sys_after - E_sys_init)
