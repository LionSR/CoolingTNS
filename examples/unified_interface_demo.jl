#!/usr/bin/env julia
"""
Demonstration of the unified cooling simulation interface.

This example shows how the same code can run different backends
(ED, TN) with different simulation methods using the new dispatch architecture.
"""

using CoolingTNS
using Statistics

# Common parameters for all backends
function setup_parameters()
    N = 4  # Small system for all backends
    
    # Create parameter structures
    ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5)  # N, J, hx, hz
    coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.15, 20, 2.0, nothing)
    
    return ham_params, coupling_params
end

# Run cooling with specified backend and methods
function run_cooling_demo(backend::CoolingTNS.CoolingBackend, 
                         sim_method::CoolingTNS.SimulationMethod,
                         evolution_method::CoolingTNS.EvolutionMethod,
                         name::String)
    println("\n" * "="^60)
    println("Running cooling with $name")
    println("="^60)
    
    # Get parameters
    ham_params, coupling_params = setup_parameters()
    
    # Create simulation parameters
    sim_params = CoolingTNS.create_sim_params(backend;
        sim_method=sim_method,
        evolution_method=evolution_method,
        pe=0.0,  # No noise
        cutoff=1e-6,
        Dmax=30,
        tau=0.1,
        n_trajectories=(sim_method isa CoolingTNS.MonteCarloWavefunction ? 50 : 1)
    )
    
    # Setup problem
    println("Setting up problem...")
    cooling_problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
    println("  Ground state energy: e₀/N = $(cooling_problem.e₀/ham_params.N)")
    
    # Setup initial state
    println("Setting up initial state...")
    initial_state = CoolingTNS.setup_initial_state(
        cooling_problem, 
        sim_params,
        "product",  # Default alternating up/down
        0.0
    )
    
    # Run cooling
    println("Running cooling simulation...")
    t_start = time()
    results = CoolingTNS.run_cooling(
        cooling_problem,
        initial_state,
        coupling_params,
        sim_params,
        ham_params
    )
    t_elapsed = time() - t_start
    
    # Analyze results
    N = ham_params.N
    E_initial = results["E_list"][1]
    E_final = results["E_list"][end]
    GS_overlap_initial = results["GS_overlap_list"][1]
    GS_overlap_final = results["GS_overlap_list"][end]
    
    println("\nResults:")
    println("  Initial energy/N: $(E_initial/N)")
    println("  Final energy/N: $(E_final/N)")
    println("  Energy reduction: $(E_initial - E_final)")
    println("  Initial GS overlap: $GS_overlap_initial")
    println("  Final GS overlap: $GS_overlap_final")
    println("  Simulation time: $(round(t_elapsed, digits=2))s")
    
    return results
end

# Main demonstration
function main()
    println("CoolingTNS Unified Interface Demonstration")
    println("==========================================")
    
    # Define all backend/method combinations to test
    configurations = [
        (CoolingTNS.EDBackend(), CoolingTNS.DensityMatrix(), CoolingTNS.ContinuousEvolution(), "ED (Density Matrix)"),
        (CoolingTNS.EDBackend(), CoolingTNS.MonteCarloWavefunction(), CoolingTNS.ContinuousEvolution(), "ED (Monte Carlo)"),
        (CoolingTNS.TNBackend(), CoolingTNS.MonteCarloWavefunction(), CoolingTNS.ContinuousEvolution(), "TN (MPS)"),
        (CoolingTNS.TNBackend(), CoolingTNS.DensityMatrix(), CoolingTNS.TrotterEvolution(), "TN (MPO)"),
        (CoolingTNS.TNBackend(), CoolingTNS.MonteCarloWavefunction(), CoolingTNS.TrotterEvolution(), "TN (Trotter-MPS)")
    ]
    
    results_dict = Dict()
    
    for (backend, sim_method, evolution_method, name) in configurations
        try
            results = run_cooling_demo(backend, sim_method, evolution_method, name)
            results_dict[name] = results
        catch e
            println("\nError with $name: $e")
        end
    end
    
    # Compare results
    println("\n" * "="^60)
    println("Comparison of Final Results")
    println("="^60)
    println("Backend                  | E_final/N    | GS Overlap")
    println("-"^55)
    
    ham_params, _ = setup_parameters()
    N = ham_params.N
    
    for (name, results) in sort(collect(results_dict))
        E_final = results["E_list"][end] / N
        GS_overlap = results["GS_overlap_list"][end]
        println("$(rpad(name, 24)) | $(round(E_final, digits=4))     | $(round(GS_overlap, digits=4))")
    end
    
    println("\nKey observations:")
    println("- ED provides exact results (benchmark)")
    println("- TN methods provide good approximations with controlled error")
    println("- Different simulation methods (DensityMatrix vs MonteCarloWavefunction)")
    println("- Different evolution methods (Continuous vs Trotter)")
    println("- The unified interface with dispatch makes it easy to compare methods")
    
    # Demonstrate the dispatch pattern
    println("\n" * "="^60)
    println("Dispatch Pattern Demonstration")
    println("="^60)
    println("The new architecture uses multiple dispatch to route to correct implementations:")
    println()
    println("1. Backend dispatch:")
    println("   - EDBackend() → Exact diagonalization methods")
    println("   - TNBackend() → Tensor network methods")
    println()
    println("2. Simulation method dispatch:")
    println("   - DensityMatrix() → Full density matrix evolution")
    println("   - MonteCarloWavefunction() → Stochastic trajectories")
    println()
    println("3. Evolution method dispatch:")
    println("   - ContinuousEvolution() → Matrix exponential evolution")
    println("   - TrotterEvolution() → Trotter decomposition")
    println()
    println("The combination determines the actual implementation used!")
end

# Run if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end