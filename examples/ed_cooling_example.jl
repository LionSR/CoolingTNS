#!/usr/bin/env julia
"""
Example: Running ED cooling simulations with CoolingTNS

This example demonstrates how to use the exact diagonalization (ED) method
for cooling simulations using the new dispatch architecture.
"""

using CoolingTNS

# Example 1: Basic ED simulation with density matrix
println("Example 1: ED with density matrix method")
println("-" * 40)

# Set up parameters
N = 5
ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5)  # N, J, hx, hz
coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.1, 50, 5.0, nothing)

# Create backend and simulation parameters
backend = CoolingTNS.EDBackend()
sim_method = CoolingTNS.DensityMatrix()
evolution_method = CoolingTNS.ContinuousEvolution()

sim_params = CoolingTNS.create_sim_params(backend; 
    sim_method=sim_method,
    evolution_method=evolution_method,
    pe=0.0  # No noise
)

# Setup problem
cooling_problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)

# Create initial state (all down state using theta parameterization)
initial_state = CoolingTNS.setup_initial_state(
    cooling_problem, 
    sim_params,
    "theta",
    -0.5  # All down state
)

# Run cooling simulation
results = CoolingTNS.run_cooling(
    cooling_problem,
    initial_state,
    coupling_params,
    sim_params,
    ham_params
)

println("Ground state energy density: e₀/N = $(cooling_problem.e₀/N)")
println("Final energy density: E/N = $(results["E_list"][end]/N)")
println("Final ground state overlap: $(results["GS_overlap_list"][end])")
println("Final purity: $(results["purity_list"][end])")

# Example 2: Monte Carlo wavefunction method with noise
println("\n\nExample 2: ED with Monte Carlo wavefunction method")
println("-" * 40)

# Set up parameters for smaller system
N2 = 4
ham_params2 = CoolingTNS.NiIsingParameters(N2, 1.0, -1.05, 0.5)
coupling_params2 = CoolingTNS.BasicCouplingParameters("YY", 0.2, 30, 3.0, nothing)

# Create simulation parameters with MC method
sim_method2 = CoolingTNS.MonteCarloWavefunction()
sim_params2 = CoolingTNS.create_sim_params(backend;
    sim_method=sim_method2,
    evolution_method=evolution_method,
    pe=0.01,  # Noise strength
    n_trajectories=50
)

# Setup and run
cooling_problem2 = CoolingTNS.setup_problem(backend, ham_params2, coupling_params2, sim_params2)

initial_state2 = CoolingTNS.setup_initial_state(
    cooling_problem2,
    sim_params2,
    "product",  # Product state
    0.0
)

results2 = CoolingTNS.run_cooling(
    cooling_problem2,
    initial_state2,
    coupling_params2,
    sim_params2,
    ham_params2
)

println("Ground state energy density: e₀/N = $(cooling_problem2.e₀/N2)")
println("Final energy density: E/N = $(results2["E_list"][end]/N2)")
println("Final ground state overlap: $(results2["GS_overlap_list"][end])")
println("Number of trajectories: $(sim_params2.extra.n_trajectories)")
println("Noise strength: pe = $(sim_params2.extra.pe)")

# Example 3: Direct command line usage
println("\n\nExample 3: Command line usage")
println("-" * 40)
println("You can run ED simulations directly from the command line:")
println()
println("# Density matrix method (exact, includes all quantum correlations):")
println("julia Cooling.jl --N 6 --problem niIsing --backend ED --sim_method density_matrix \\")
println("    --evolution_method continuous --coupling XX --g 0.15 --te 4.0 --steps 100")
println()
println("# Monte Carlo wavefunction (stochastic trajectories, better scaling):")
println("julia Cooling.jl --N 6 --problem niIsing --backend ED --sim_method monte_carlo \\")
println("    --evolution_method continuous --n_trajectories 200 --coupling ZZ --g 0.1 --te 2.0 --steps 50 --peInt 5")

# Example 4: Comparing different backends
println("\n\nExample 4: Backend comparison")
println("-" * 40)

# Small system where all backends work
N_compare = 4
ham_params_compare = CoolingTNS.NiIsingParameters(N_compare, 1.0, -1.05, 0.5)
coupling_params_compare = CoolingTNS.BasicCouplingParameters("XX", 0.2, 20, 2.0, nothing)

backends = [
    (CoolingTNS.EDBackend(), CoolingTNS.DensityMatrix(), "ED (Density Matrix)"),
    (CoolingTNS.EDBackend(), CoolingTNS.MonteCarloWavefunction(), "ED (Monte Carlo)"),
    (CoolingTNS.TNBackend(), CoolingTNS.MonteCarloWavefunction(), "TN (MPS)"),
    (CoolingTNS.TNBackend(), CoolingTNS.DensityMatrix(), "TN (MPO)")
]

for (backend, sim_method, name) in backends
    println("\nRunning with backend: $name")
    
    try
        # Create appropriate evolution method
        evolution_method = if backend isa CoolingTNS.TNBackend && sim_method isa CoolingTNS.DensityMatrix
            CoolingTNS.TrotterEvolution()
        else
            CoolingTNS.ContinuousEvolution()
        end
        
        # Create simulation parameters
        sim_params = CoolingTNS.create_sim_params(backend;
            sim_method=sim_method,
            evolution_method=evolution_method,
            Dmax=30,
            n_trajectories=(sim_method isa CoolingTNS.MonteCarloWavefunction ? 10 : 1)
        )
        
        # Setup and run
        cooling_problem = CoolingTNS.setup_problem(backend, ham_params_compare, coupling_params_compare, sim_params)
        initial_state = CoolingTNS.setup_initial_state(cooling_problem, sim_params, "product", 0.0)
        
        results = CoolingTNS.run_cooling(
            cooling_problem,
            initial_state,
            coupling_params_compare,
            sim_params,
            ham_params_compare
        )
        
        println("  ✓ Backend $name completed successfully")
        println("    Final energy density: $(results["E_list"][end]/N_compare)")
        println("    Final overlap: $(results["GS_overlap_list"][end])")
    catch e
        println("  ✗ Backend $name failed: $e")
    end
end

println("\n" * "="^60)
println("ED implementation with new dispatch architecture!")
println("Key features:")
println("  • Clean backend selection (EDBackend vs TNBackend)")
println("  • Explicit simulation method choice (DensityMatrix vs MonteCarloWavefunction)")
println("  • Consistent evolution method specification")
println("  • Type-safe parameter structures")
println("  • Unified interface across all backends")
println("="^60)