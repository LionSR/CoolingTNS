#!/usr/bin/env julia

"""
Test script for ED backend with periodic/antiperiodic boundary conditions and k-space measurements.
"""

using CoolingTNS
using LinearAlgebra
using HDF5
using Printf

function test_ed_kspace()
    println("Testing ED backend with k-space measurements...")
    
    # Test parameters
    N = 6  # Small system for ED
    
    # Test both periodic and antiperiodic BC
    for bc in [:periodic, :antiperiodic]
        println("\n=== Testing with $bc boundary conditions ===")
        
        # Create Hamiltonian parameters
        ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5; bc=bc)
        
        # Coupling parameters
        coupling_params = CoolingTNS.BasicCouplingParameters(
            "XX",     # coupling type
            0.3,      # g
            20,       # steps
            2.0,      # te
            nothing   # delta (will be computed)
        )
        
        # Simulation parameters for ED with Monte Carlo
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0,  # No noise for this test
            n_trajectories=1
        )
        
        # Setup problem
        backend = CoolingTNS.EDBackend()
        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        
        # Initial state - product state
        state = CoolingTNS.setup_initial_state(
            backend, 
            problem,
            sim_params,
            "product",  # init_state_type
            Dict{String, Any}()  # No special parameters
        )
        
        # Run cooling
        println("Running cooling simulation...")
        results = CoolingTNS.run_cooling(problem, state, coupling_params, sim_params, ham_params)
        
        # Check if k-space data was collected
        if haskey(results, "momentum_dist") && haskey(results, "k_values")
            k_values = results["k_values"]
            momentum_dist = results["momentum_dist"]
            
            println("K-space measurement successful!")
            println("Number of k points: $(length(k_values))")
            println("K values: $k_values")
            
            # Print initial and final momentum distributions
            println("\nInitial momentum distribution:")
            for (k, n_k) in zip(k_values, momentum_dist[1, :])
                @printf("k = %3d: n_k = %.4f\n", k, n_k)
            end
            
            println("\nFinal momentum distribution:")
            for (k, n_k) in zip(k_values, momentum_dist[end, :])
                @printf("k = %3d: n_k = %.4f\n", k, n_k)
            end
            
            # Save results for plotting
            filename = "test_ed_kspace_$(bc)_N$(N).h5"
            h5open(filename, "w") do file
                for (key, value) in results
                    write(file, string(key), value)
                end
                write(file, "ham_params_bc", string(bc))
                write(file, "delta", problem.extra.coupling_params.delta)
            end
            println("\nResults saved to $filename")
            
            # Test plotting functions if available
            try
                CoolingTNS.plot_momentum_distribution(filename; save_fig=false)
                CoolingTNS.plot_momentum_distribution_heatmap(filename; save_fig=false)
                println("Plotting functions work correctly!")
            catch e
                println("Plotting error (might be due to missing display): $e")
            end
            
        else
            println("WARNING: No k-space data found in results!")
        end
        
        # Also test with density matrix method
        println("\n--- Testing with density matrix method ---")
        sim_params_dm = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0
        )
        
        problem_dm = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params_dm)
        state_dm = CoolingTNS.setup_initial_state(
            backend, 
            problem_dm,
            sim_params_dm,
            "identity",  # Start from maximally mixed
            Dict{String, Any}()
        )
        
        # Run short cooling
        coupling_params_short = CoolingTNS.BasicCouplingParameters(
            "XX", 0.3, 5, 2.0, problem_dm.extra.coupling_params.delta
        )
        results_dm = CoolingTNS.run_cooling(problem_dm, state_dm, coupling_params_short, sim_params_dm, ham_params)
        
        if haskey(results_dm, "momentum_dist")
            println("Density matrix k-space measurement also successful!")
        else
            println("WARNING: No k-space data for density matrix method!")
        end
    end
    
    println("\n=== All tests completed ===")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_ed_kspace()
end