#!/usr/bin/env julia

"""
Quick test to generate k-space plots for ED simulations.
"""

using CoolingTNS

# Run a small ED simulation with open BC
N = 6
ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5; bc=:open)
coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.3, 30, 2.0, nothing)
sim_params = CoolingTNS.UnifiedSimulationParameters(
    CoolingTNS.MonteCarloWavefunction(),
    CoolingTNS.ContinuousEvolution();
    pe=0.0
)

backend = CoolingTNS.EDBackend()
problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
state = CoolingTNS.setup_initial_state(backend, problem, sim_params, "product", Dict{String, Any}())

println("Running cooling simulation with open BC...")
results = CoolingTNS.run_cooling(problem, state, coupling_params, sim_params, ham_params)

# Save results
filename = "test_kspace_open_BC.h5"
using HDF5
h5open(filename, "w") do file
    for (key, value) in results
        write(file, string(key), value)
    end
    write(file, "delta", problem.extra.coupling_params.delta)
end

println("\nGenerating k-space plots...")
CoolingTNS.plot_momentum_distribution(filename; save_fig=true)
CoolingTNS.plot_momentum_distribution_heatmap(filename; save_fig=true)

println("Plots saved!")