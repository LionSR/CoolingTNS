#!/usr/bin/env julia

"""
Benchmark ED simulations to test speed improvement from caching evolution operators.
"""

using CoolingTNS
using BenchmarkTools

println("="^60)
println("ED Backend Speed Benchmark")
println("="^60)

# Test parameters
N = 8  # System size
steps = 20  # Number of cooling steps

println("\nTest parameters:")
println("- System size: N = $N")
println("- Cooling steps: $steps")
println("- Problem: Transverse field Ising with PBC")

# Create parameters
ham_params = CoolingTNS.IsingParameters(N, 1.0, 2.0, :periodic)
coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.3, steps, 2.0, nothing)

# Test both methods
for method in ["monte_carlo", "density_matrix"]
    println("\n" * "-"^40)
    println("Testing with $method method")
    println("-"^40)
    
    # Create simulation parameters
    sim_method = method == "monte_carlo" ? 
        CoolingTNS.MonteCarloWavefunction() : 
        CoolingTNS.DensityMatrix()
    
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        sim_method,
        CoolingTNS.ContinuousEvolution();
        pe=0.0
    )
    
    # Setup problem
    backend = CoolingTNS.EDBackend()
    
    # Time the setup (includes initial Hamiltonian diagonalization)
    setup_time = @elapsed begin
        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        state = CoolingTNS.setup_initial_state(backend, problem, sim_params, "product", Dict{String, Any}())
    end
    
    println("Setup time: $(round(setup_time, digits=3)) seconds")
    
    # Time the cooling simulation
    cooling_time = @elapsed begin
        results = CoolingTNS.run_cooling(problem, state, coupling_params, sim_params, ham_params)
    end
    
    println("Cooling time: $(round(cooling_time, digits=3)) seconds")
    println("Time per step: $(round(cooling_time/steps, digits=4)) seconds")
    
    # Check if results are reasonable
    E_initial = results["E_list"][1]
    E_final = results["E_list"][end]
    println("\nResults:")
    println("- Initial energy/N: $(round(E_initial/N, digits=4))")
    println("- Final energy/N: $(round(E_final/N, digits=4))")
    println("- Energy reduced: $(E_final < E_initial ? "✓" : "✗")")
end

println("\n" * "="^60)
println("Benchmark complete!")
println("\nNote: The evolution operator U = exp(-iHt) is now cached,")
println("so it's computed only once per simulation instead of at every step.")
println("="^60)