#!/usr/bin/env julia

# Direct test of ED cooling
push!(LOAD_PATH, @__DIR__)
using CoolingTNS

# Test parameters
N = 4
args = Dict{String,Any}(
    "N" => N,
    "problem" => "Ising",
    "backend" => "ED",
    "sim_method" => "monte_carlo",
    "evolution_method" => "continuous",
    "J" => 1.0,
    "h" => -2.0,
    "g" => 0.2,
    "te" => 1.0,
    "coupling" => "XX",
    "steps" => 5,
    "tau" => 0.1,
    "Dmax" => 20,
    "cutoff" => 1e-6,
    "n_trajectories" => 1,
    "peInt" => 0,
    "init_state" => "product",
    "theta" => 0.0,
    "search_method" => "Random",
    "num_trials" => 10,
    "window_size" => 50,
    "hx" => -1.05,
    "hz" => 0.5
)

# Run cooling
println("Running ED cooling with N=$N...")
result = CoolingTNS.run_cooling(args)

# Display results
println("\nResults:")
println("Initial energy/N: ", result["E_list"][1]/N)
println("Final energy/N: ", result["E_list"][end]/N)
println("Energy change: ", result["E_list"][end] - result["E_list"][1])
println("Ground state overlap: ", result["GS_overlap_list"][end])

# Compare with TN
println("\n\nRunning TN cooling for comparison...")
args["backend"] = "TN"
result_tn = CoolingTNS.run_cooling(args)

println("\nTN Results:")
println("Initial energy/N: ", result_tn["E_list"][1]/N)
println("Final energy/N: ", result_tn["E_list"][end]/N)
println("Energy change: ", result_tn["E_list"][end] - result_tn["E_list"][1])
println("Ground state overlap: ", result_tn["GS_overlap_list"][end])