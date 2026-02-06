using CoolingTNS
using CoolingTNS: prepare_combined_state_ed, evolve_ed, measure_ed!, get_bath_ground_state_ed
using LinearAlgebra

# Debug with N=2
backend = CoolingTNS.EDBackend()
ham_params = CoolingTNS.IsingParameters(2, 1.0, 1.0)
coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.3, 1, 2.0, nothing)
sim_params = CoolingTNS.UnifiedSimulationParameters(CoolingTNS.MonteCarloWavefunction(), CoolingTNS.ContinuousEvolution())

problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
H_sb = problem.H_sys_bath

# Check bath ground state
psi_bath = get_bath_ground_state_ed(2, "XX")
println("Bath ground state (should be |↓↓⟩):")
println("  data: ", psi_bath.data)

# Initial system: |↑↑⟩ (config 3 = binary 11)
psi_sys = CoolingTNS.product_state_ed(2, 3)
println("\nSystem state |↑↑⟩:")
println("  data: ", psi_sys.data)

# Combined state
psi_comb = prepare_combined_state_ed(psi_sys, 2, "XX")
println("\nCombined state (4 qubits):")
nonzero = findall(x -> abs(x) > 1e-10, psi_comb.data)
println("  Non-zero indices: ", nonzero)
println("  Values at those indices: ", psi_comb.data[nonzero])

# Expected layout: sys1(bit0)=1, bath1(bit1)=1, sys2(bit2)=1, bath2(bit3)=1
# For |↑↑⟩ sys and |↓↓⟩ bath:
# sys1=↑=1, bath1=↓=1, sys2=↑=1, bath2=↓=1
# Binary: 1111 = 15, so index 16
println("Expected: index 16 (binary 1111 for |↑↓↑↓⟩ interleaved)")

# Energy of combined state
E_comb = real(psi_comb.data' * H_sb * psi_comb.data)
println("\nCombined state energy: ", E_comb)

# Evolve
psi_evolved = evolve_ed(H_sb, psi_comb, 2.0)
E_evolved = real(psi_evolved.data' * H_sb * psi_evolved.data)
println("Evolved energy: ", E_evolved)

overlap = abs(dot(psi_comb.data, psi_evolved.data))
println("Overlap with initial: ", overlap)
println("State changed: ", overlap < 0.999)
