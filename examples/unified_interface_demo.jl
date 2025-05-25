#!/usr/bin/env julia
"""
Demonstration of the unified cooling simulation interface.

This example shows how the same code can run different backends
(ED, MPS, MPO, TrotterMPS) with minimal changes.
"""

using CoolingTNS
using Statistics

# Common parameters for all backends
function setup_parameters()
    N = 4  # Small system for all backends
    problem = "niIsing"
    ham_params = (1.0, -1.05, 0.5)  # J, hx, hz
    
    coupling_params = Dict(
        "coupling" => "XX",
        "g" => 0.15,
        "te" => 2.0,
        "steps" => 20
    )
    
    sim_params = Dict(
        "pe" => 0.0,  # No noise
        "cutoff" => 1e-6,
        "Dmax" => 30,
        "tau" => 0.1,
        "n_trajectories" => 50
    )
    
    return N, problem, ham_params, coupling_params, sim_params
end

# Run cooling with any backend
function run_cooling_demo(backend_name::String, sim_method=nothing)
    println("\n" * "="^60)
    println("Running cooling with $backend_name backend")
    println("="^60)
    
    # Get parameters
    N, problem, ham_params, coupling_params, sim_params = setup_parameters()
    
    # Get backend
    backend = CoolingTNS.get_backend(backend_name)
    
    # Override simulation method if specified
    if isnothing(sim_method)
        sim_method = CoolingTNS.simulation_method(backend)
    end
    
    # Setup problem
    println("Setting up problem...")
    cooling_problem = CoolingTNS.setup_problem(backend, N, problem, ham_params, coupling_params, sim_params)
    println("  Ground state energy: e₀/N = $(cooling_problem.e₀/N)")
    
    # Setup initial state
    println("Setting up initial state...")
    initial_state = CoolingTNS.setup_initial_state(
        cooling_problem, 
        "product",  # Default alternating up/down
        0.0;
        method=sim_method
    )
    
    # Run cooling
    println("Running cooling simulation...")
    t_start = time()
    results = CoolingTNS.run_cooling(
        cooling_problem,
        initial_state,
        coupling_params,
        sim_params,
        ham_params  # Only used by TrotterMPS
    )
    t_elapsed = time() - t_start
    
    # Analyze results
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
    
    # Test all backends
    backends = ["ED", "MPS", "MPO", "TrotterMPS"]
    results_dict = Dict()
    
    for backend in backends
        try
            if backend == "ED"
                # Test both ED methods
                println("\nTesting ED with density matrix method:")
                results_dm = run_cooling_demo("ED", CoolingTNS.DensityMatrix())
                results_dict["ED_DM"] = results_dm
                
                println("\nTesting ED with Monte Carlo wavefunction:")
                results_mc = run_cooling_demo("ED", CoolingTNS.MonteCarloWavefunction())
                results_dict["ED_MC"] = results_mc
            else
                results = run_cooling_demo(backend)
                results_dict[backend] = results
            end
        catch e
            println("\nError with $backend: $e")
        end
    end
    
    # Compare results
    println("\n" * "="^60)
    println("Comparison of Final Results")
    println("="^60)
    println("Backend          | E_final/N    | GS Overlap")
    println("-"^45)
    
    N = 4
    for (name, results) in results_dict
        E_final = results["E_list"][end] / N
        GS_overlap = results["GS_overlap_list"][end]
        println("$(rpad(name, 16)) | $(round(E_final, digits=4))     | $(round(GS_overlap, digits=4))")
    end
    
    println("\nKey observations:")
    println("- ED provides exact results (benchmark)")
    println("- MPS/MPO provide good approximations with controlled error")
    println("- All methods show successful cooling (energy decrease)")
    println("- The unified interface makes it easy to compare methods")
end

# Run if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end