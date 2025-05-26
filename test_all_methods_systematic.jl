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
steps = 10
te = 1.0  # Longer evolution time per step for more cooling

println("=== Systematic Comparison of All Methods ===")
println("System: N=$N, Ising model")
println("Steps: $steps, Evolution time per step: $te")
println("Initial state: all up |000⟩")
println()

# Test configurations
test_configs = [
    # ED Backend tests
    ("ED", "density_matrix", "continuous", Dict()),
    ("ED", "monte_carlo", "continuous", Dict("n_trajectories" => 10)),
    ("ED", "density_matrix", "trotter", Dict("tau" => 0.1)),
    ("ED", "monte_carlo", "trotter", Dict("tau" => 0.1, "n_trajectories" => 10)),
    
    # TN Backend tests
    ("TN", "monte_carlo", "continuous", Dict("Dmax" => 64)),
    ("TN", "density_matrix", "trotter", Dict("Dmax" => 64, "tau" => 0.1)),
    ("TN", "monte_carlo", "trotter", Dict("Dmax" => 64, "tau" => 0.1, "n_trajectories" => 10)),
]

# Store results
results = []

for (backend, sim_method, evolution_method, extra_params) in test_configs
    config_name = "$backend-$sim_method-$evolution_method"
    println("\n" * "="^60)
    println("Testing: $config_name")
    println("="^60)
    
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
        "--init_state", "product"
    ]
    
    # Add extra parameters
    for (key, value) in extra_params
        push!(args, "--$key", "$value")
    end
    
    try
        # Parse arguments
        parsed = CoolingTNS.parse_commandline(args)
        
        # Setup
        problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed)
        backend_obj = CoolingTNS.get_backend(parsed["backend"])
        
        # Create sim params
        sim_method_obj = if sim_method == "density_matrix"
            CoolingTNS.DensityMatrix()
        else
            CoolingTNS.MonteCarloWavefunction()
        end
        
        evolution_method_obj = if evolution_method == "continuous"
            CoolingTNS.ContinuousEvolution()
        else
            CoolingTNS.TrotterEvolution()
        end
        
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            sim_method_obj, evolution_method_obj;
            Dmax=get(parsed, "Dmax", 20),
            cutoff=parsed["cutoff"],
            tau=get(parsed, "tau", 0.1),
            pe=parsed["peInt"]*1e-3,
            n_trajectories=get(parsed, "n_trajectories", 1)
        )
        
        # Setup problem
        cooling_problem = CoolingTNS.setup_problem(backend_obj, ham_params, coupling_params, sim_params)
        
        # Setup initial state
        initial_state = CoolingTNS.setup_initial_state(
            cooling_problem,
            sim_params,
            parsed["init_state"],
            parsed["theta"]
        )
        
        # Run cooling
        println("Running cooling simulation...")
        results_dict = CoolingTNS.run_cooling(
            cooling_problem,
            initial_state,
            coupling_params,
            sim_params,
            ham_params
        )
        
        # Extract results
        E_list = results_dict["E_list"]
        E_init = E_list[1]
        E_final = CoolingTNS.mean_last_window(E_list, min(50, length(E_list)))
        overlap_final = CoolingTNS.mean_last_window(results_dict["GS_overlap_list"], min(50, length(results_dict["GS_overlap_list"])))
        
        # Store results
        push!(results, (
            config = config_name,
            E_init = E_init/N,
            E_final = E_final/N,
            E_ground = cooling_problem.e₀/N,
            overlap_final = overlap_final,
            cooling = E_final < E_init,
            energy_list = E_list ./ N
        ))
        
        println("\nResults:")
        println("  Initial energy/N: $(E_init/N)")
        println("  Final energy/N: $(E_final/N)")
        println("  Ground state energy/N: $(cooling_problem.e₀/N)")
        println("  Final overlap: $overlap_final")
        println("  Status: $(E_final < E_init ? "✅ COOLING" : "❌ HEATING")")
        
    catch e
        println("\nERROR: $e")
        if isa(e, MethodError)
            println("Missing method: $(e.f)($(join(typeof.(e.args), ", ")))")
        end
    end
end

# Summary
println("\n\n" * "="^80)
println("SUMMARY")
println("="^80)
println("Ground state energy/N ≈ $(-2.083)")
println()
println(@sprintf("%-30s %10s %10s %10s %8s %10s", 
        "Method", "E_init/N", "E_final/N", "Change", "Overlap", "Status"))
println("-"^80)

for res in results
    change = res.E_final - res.E_init
    status = res.cooling ? "✅ Cool" : "❌ Heat"
    println(@sprintf("%-30s %10.6f %10.6f %+10.6f %8.4f %10s",
            res.config, res.E_init, res.E_final, change, res.overlap_final, status))
end

# Compare ED vs TN for same methods
println("\n\nED vs TN Comparison:")
println("-"^60)

# Group by method
method_groups = Dict()
for res in results
    parts = split(res.config, "-")
    method = join(parts[2:end], "-")
    if !haskey(method_groups, method)
        method_groups[method] = Dict()
    end
    method_groups[method][parts[1]] = res
end

for (method, backends) in method_groups
    if haskey(backends, "ED") && haskey(backends, "TN")
        ed_res = backends["ED"]
        tn_res = backends["TN"]
        
        diff_init = abs(ed_res.E_init - tn_res.E_init)
        diff_final = abs(ed_res.E_final - tn_res.E_final)
        
        println("\n$method:")
        println("  Initial difference: $diff_init")
        println("  Final difference: $diff_final")
        
        # Plot energy evolution
        if length(ed_res.energy_list) > 0 && length(tn_res.energy_list) > 0
            println("  Energy evolution:")
            println("  Step    ED E/N      TN E/N     Difference")
            max_steps = min(length(ed_res.energy_list), length(tn_res.energy_list), 5)
            for i in 1:max_steps
                diff = abs(ed_res.energy_list[i] - tn_res.energy_list[i])
                println(@sprintf("  %3d  %10.6f  %10.6f  %10.6f", 
                        i, ed_res.energy_list[i], tn_res.energy_list[i], diff))
            end
        end
        
        if diff_final < 0.01
            println("  ✅ ED and TN match well")
        else
            println("  ⚠️  ED and TN show differences")
        end
    end
end