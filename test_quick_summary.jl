#!/usr/bin/env julia

# Quick test to summarize key differences between methods
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using CoolingTNS

println("=== Quick Summary of TN vs ED Methods ===")
println()

# Test with N=3 for faster execution
args = Dict{String,Any}(
    "N" => 3,
    "problem" => "Ising",
    "J" => 1.0,
    "h" => -2.0,
    "g" => 0.2,
    "coupling" => "XX",
    "te" => 0.5,
    "steps" => 2,
    "tau" => 0.1,
    "init_state" => "product",
    "theta" => 0.0,
    "Dmax" => 10,
    "cutoff" => 1e-10,
    "peInt" => 0,
    "n_trajectories" => 5
)

println("Testing with N=3, 2 steps, te=0.5 for quick results")
println()

# Test ED density matrix continuous
println("1. ED/density_matrix/continuous:")
args["backend"] = "ED"
args["sim_method"] = "density_matrix"
args["evolution_method"] = "continuous"

try
    problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(args)
    backend_obj = CoolingTNS.EDBackend()
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.DensityMatrix(), 
        CoolingTNS.ContinuousEvolution();
        tau = 0.1,
        n_trajectories = 5,
        Dmax = 10
    )
    
    cooling_problem = CoolingTNS.setup_problem(backend_obj, ham_params, coupling_params, sim_params)
    initial_state = CoolingTNS.setup_initial_state(cooling_problem, sim_params, args["init_state"], args["theta"])
    
    E_initial = CoolingTNS.expect_ed(cooling_problem.H_sys, initial_state.state)
    result = CoolingTNS.run_cooling(cooling_problem, initial_state, coupling_params, sim_params, ham_params)
    E_final = result["E_list"][end]
    
    println("   Initial E/N = $(E_initial/3)")
    println("   Final E/N = $(E_final/3)")
    println("   Cooling: $(E_final < E_initial ? "✓" : "✗")")
catch e
    println("   ERROR: $(typeof(e))")
end

# Test ED monte carlo continuous
println("\n2. ED/monte_carlo/continuous:")
args["sim_method"] = "monte_carlo"

try
    problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(args)
    backend_obj = CoolingTNS.EDBackend()
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.MonteCarloWavefunction(), 
        CoolingTNS.ContinuousEvolution();
        tau = 0.1,
        n_trajectories = 5,
        Dmax = 10
    )
    
    cooling_problem = CoolingTNS.setup_problem(backend_obj, ham_params, coupling_params, sim_params)
    initial_state = CoolingTNS.setup_initial_state(cooling_problem, sim_params, args["init_state"], args["theta"])
    
    E_initial = CoolingTNS.expect_ed(cooling_problem.H_sys, initial_state.state)
    result = CoolingTNS.run_cooling(cooling_problem, initial_state, coupling_params, sim_params, ham_params)
    E_final = result["E_list"][end]
    
    println("   Initial E/N = $(E_initial/3)")
    println("   Final E/N = $(E_final/3)")
    println("   Cooling: $(E_final < E_initial ? "✓" : "✗")")
catch e
    println("   ERROR: $(typeof(e))")
end

# Test TN monte carlo continuous
println("\n3. TN/monte_carlo/continuous:")
args["backend"] = "TN"
args["sim_method"] = "monte_carlo"
args["evolution_method"] = "continuous"

try
    problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(args)
    backend_obj = CoolingTNS.TNBackend()
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.MonteCarloWavefunction(), 
        CoolingTNS.ContinuousEvolution();
        tau = 0.1,
        n_trajectories = 5,
        Dmax = 10
    )
    
    cooling_problem = CoolingTNS.setup_problem(backend_obj, ham_params, coupling_params, sim_params)
    initial_state = CoolingTNS.setup_initial_state(cooling_problem, sim_params, args["init_state"], args["theta"])
    
    E_initial = real(inner(initial_state.state', cooling_problem.H_sys, initial_state.state))
    result = CoolingTNS.run_cooling(cooling_problem, initial_state, coupling_params, sim_params, ham_params)
    E_final = result["E_list"][end]
    
    println("   Initial E/N = $(E_initial/3)")
    println("   Final E/N = $(E_final/3)")
    println("   Cooling: $(E_final < E_initial ? "✓" : "✗")")
catch e
    println("   ERROR: $(typeof(e))")
end

println("\n=== Key Observations ===")
println("1. ED density matrix continuous: Shows proper cooling")
println("2. ED monte carlo methods: May show energy decrease beyond ground state (unphysical)")
println("3. TN methods: Should match ED density matrix results")
println("\nThe issue with ED Monte Carlo going below ground state needs investigation.")