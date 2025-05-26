#!/usr/bin/env julia

using ITensors
using ITensorMPS
using LinearAlgebra
using SparseArrays
using Printf

# Include main module
include("src/CoolingTNS.jl")
using .CoolingTNS

# Test configuration
N = 3
steps = 20  # More steps to see cooling
te = 0.5    # Evolution time per step

println("=== Comparison of Working Methods ===")
println("System: N=$N, Ising model")
println("Steps: $steps, Evolution time per step: $te")
println("Initial state: all up |000⟩")
println("Ground state energy/N ≈ -2.083")
println()

# Working method combinations
test_configs = [
    # ED Backend
    ("ED", "density_matrix", "continuous"),
    ("ED", "monte_carlo", "continuous"),
    ("ED", "density_matrix", "trotter"),
    ("ED", "monte_carlo", "trotter"),
    
    # TN Backend (excluding density_matrix + continuous which doesn't work)
    ("TN", "monte_carlo", "continuous"),
    ("TN", "density_matrix", "trotter"),
    ("TN", "monte_carlo", "trotter"),
]

# Store results
results = Dict()

for (backend, sim_method, evolution_method) in test_configs
    config_name = "$backend-$sim_method-$evolution_method"
    print("Testing $config_name... ")
    flush(stdout)
    
    # Create arguments
    args = [
        "--N", "$N",
        "--problem", "Ising",
        "--backend", backend,
        "--sim_method", sim_method,
        "--evolution_method", evolution_method,
        "--coupling", "XX",
        "--g", "0.1",
        "--te", "$te",
        "--steps", "$steps",
        "--init_state", "product",  # All up
        "--Dmax", "64",
        "--tau", "0.1",
        "--n_trajectories", "5"  # For Monte Carlo
    ]
    
    try
        # Parse and setup
        parsed = CoolingTNS.parse_commandline(args)
        problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed)
        backend_obj = CoolingTNS.get_backend(parsed["backend"])
        
        # Create sim params
        sim_method_obj = sim_method == "density_matrix" ? CoolingTNS.DensityMatrix() : CoolingTNS.MonteCarloWavefunction()
        evolution_method_obj = evolution_method == "continuous" ? CoolingTNS.ContinuousEvolution() : CoolingTNS.TrotterEvolution()
        
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            sim_method_obj, evolution_method_obj;
            Dmax=parsed["Dmax"],
            cutoff=parsed["cutoff"],
            tau=parsed["tau"],
            pe=parsed["peInt"]*1e-3,
            n_trajectories=parsed["n_trajectories"]
        )
        
        # Setup and run
        cooling_problem = CoolingTNS.setup_problem(backend_obj, ham_params, coupling_params, sim_params)
        initial_state = CoolingTNS.setup_initial_state(cooling_problem, sim_params, parsed["init_state"], parsed["theta"])
        
        # Run cooling
        results_dict = CoolingTNS.run_cooling(cooling_problem, initial_state, coupling_params, sim_params, ham_params)
        
        # Extract results
        E_list = results_dict["E_list"]
        E_init = E_list[1]
        E_final = CoolingTNS.mean_last_window(E_list, min(10, length(E_list)))
        
        results[config_name] = (
            E_init = E_init/N,
            E_final = E_final/N,
            E_list = E_list ./ N,
            success = E_final < E_init
        )
        
        println(E_final < E_init ? "✅ Cooling" : "❌ Heating")
        
    catch e
        println("❌ Error: $(typeof(e))")
        results[config_name] = (
            E_init = NaN,
            E_final = NaN,
            E_list = Float64[],
            success = false
        )
    end
end

# Display results table
println("\n\n" * "="^70)
println("RESULTS SUMMARY")
println("="^70)
println(@sprintf("%-30s %10s %10s %10s %8s", "Method", "E_init/N", "E_final/N", "Change", "Status"))
println("-"^70)

for (config, res) in sort(collect(results))
    if !isnan(res.E_init)
        change = res.E_final - res.E_init
        status = res.success ? "✅ Cool" : "❌ Heat"
        println(@sprintf("%-30s %10.4f %10.4f %+10.4f %8s", config, res.E_init, res.E_final, change, status))
    else
        println(@sprintf("%-30s %10s %10s %10s %8s", config, "ERROR", "ERROR", "N/A", "❌ Fail"))
    end
end

# Compare ED vs TN for same methods
println("\n\n" * "="^70)
println("ED vs TN COMPARISON")
println("="^70)

method_pairs = [
    ("monte_carlo", "continuous"),
    ("density_matrix", "trotter"),
    ("monte_carlo", "trotter")
]

for (sim, evol) in method_pairs
    ed_key = "ED-$sim-$evol"
    tn_key = "TN-$sim-$evol"
    
    if haskey(results, ed_key) && haskey(results, tn_key) && 
       !isnan(results[ed_key].E_init) && !isnan(results[tn_key].E_init)
        
        ed_res = results[ed_key]
        tn_res = results[tn_key]
        
        println("\n$sim + $evol:")
        println("  Initial: ED=$(ed_res.E_init), TN=$(tn_res.E_init), diff=$(abs(ed_res.E_init - tn_res.E_init))")
        println("  Final:   ED=$(ed_res.E_final), TN=$(tn_res.E_final), diff=$(abs(ed_res.E_final - tn_res.E_final))")
        
        # Show energy evolution
        if length(ed_res.E_list) > 0 && length(tn_res.E_list) > 0
            println("\n  Energy evolution (E/N):")
            println("  Step    ED        TN       |Diff|")
            for i in [1, 5, 10, 15, length(ed_res.E_list)]
                if i <= length(ed_res.E_list) && i <= length(tn_res.E_list)
                    diff = abs(ed_res.E_list[i] - tn_res.E_list[i])
                    println(@sprintf("  %3d  %8.4f  %8.4f  %8.4f", i, ed_res.E_list[i], tn_res.E_list[i], diff))
                end
            end
        end
        
        # Check agreement
        final_diff = abs(ed_res.E_final - tn_res.E_final)
        if final_diff < 0.01 && ed_res.success == tn_res.success
            println("\n  ✅ ED and TN agree well")
        else
            println("\n  ⚠️  ED and TN show differences (|ΔE/N| = $final_diff)")
        end
    end
end

println("\n\nExpected behavior: All methods should show energy decreasing (cooling)")
println("Ground state energy/N ≈ -2.083")