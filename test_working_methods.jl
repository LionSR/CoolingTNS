#!/usr/bin/env julia

# Test only the methods we know are working
push!(LOAD_PATH, joinpath(@__DIR__, "src"))
using CoolingTNS
using Printf

# Test parameters
const N = 4
const COUPLING = "XX"
const G = 0.2
const TE = 1.0
const STEPS = 5
const TAU = 0.1

# Common arguments
base_args = Dict{String,Any}(
    "N" => N,
    "problem" => "Ising",
    "J" => 1.0,
    "h" => -2.0,
    "g" => G,
    "coupling" => COUPLING,
    "te" => TE,
    "steps" => STEPS,
    "tau" => TAU,
    "init_state" => "product",
    "theta" => 0.0,
    "Dmax" => 100,
    "cutoff" => 1e-10,
    "peInt" => 0,
    "n_trajectories" => 10
)

# Test configurations - only the ones we expect to work
test_configs = [
    # ED methods first (they seem to work)
    ("ED", "density_matrix", "continuous"),
    ("ED", "density_matrix", "trotter"),
    ("ED", "monte_carlo", "continuous"),
    ("ED", "monte_carlo", "trotter"),
    # TN methods
    ("TN", "monte_carlo", "continuous"),
]

# Results storage
results = Dict()

println("=== Testing Working Methods ===")
println("System: N=$N, Ising model, h=-2.0")
println("Coupling: $COUPLING, g=$G")
println("Evolution: te=$TE, steps=$STEPS")
println()

# Run each configuration
for (backend, sim_method, evolution_method) in test_configs
    config_name = "$backend/$sim_method/$evolution_method"
    
    print("$config_name: ")
    flush(stdout)
    
    # Update arguments
    test_args = copy(base_args)
    test_args["backend"] = backend
    test_args["sim_method"] = sim_method
    test_args["evolution_method"] = evolution_method
    
    try
        # Setup parameters
        problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(test_args)
        
        # Get backend and create simulation parameters
        backend_obj = backend == "TN" ? CoolingTNS.TNBackend() : CoolingTNS.EDBackend()
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            sim_method == "density_matrix" ? CoolingTNS.DensityMatrix() : CoolingTNS.MonteCarloWavefunction(), 
            evolution_method == "continuous" ? CoolingTNS.ContinuousEvolution() : CoolingTNS.TrotterEvolution();
            tau = TAU,
            n_trajectories = 10,
            Dmax = 100
        )
        
        # Setup problem
        cooling_problem = CoolingTNS.setup_problem(backend_obj, ham_params, coupling_params, sim_params)
        
        # Initial state
        initial_state = CoolingTNS.setup_initial_state(cooling_problem, sim_params, test_args["init_state"], test_args["theta"])
        
        # Get initial energy
        if backend == "ED"
            E_initial = CoolingTNS.expect_ed(cooling_problem.H_sys, initial_state.state)
        else
            # For TN, need to handle MPS case
            E_initial = real(inner(initial_state.state', cooling_problem.H_sys, initial_state.state))
        end
        
        # Run cooling
        result = CoolingTNS.run_cooling(cooling_problem, initial_state, coupling_params, sim_params, ham_params)
        
        # Store results
        E_final = result["E_list"][end]
        overlap_final = result["GS_overlap_list"][end]
        
        results[config_name] = (
            E_initial = E_initial,
            E_final = E_final,
            overlap_final = overlap_final,
            energy_change = E_final - E_initial,
            ground_energy = cooling_problem.e₀
        )
        
        cooling = E_final < E_initial ? "✓" : "✗"
        println("E/N: $(E_initial/N) → $(E_final/N), Cooling: $cooling")
        
    catch e
        println("ERROR: $(typeof(e))")
        results[config_name] = e
    end
end

# Summary table
println("\n=== Summary ===")
println("Method                           | Initial E/N | Final E/N  | ΔE        | Cooling?")
println("-"^80)

for (backend, sim_method, evolution_method) in test_configs
    config_name = "$backend/$sim_method/$evolution_method"
    
    if haskey(results, config_name)
        r = results[config_name]
        if isa(r, Exception)
            @printf("%-32s | ERROR\n", config_name)
        else
            cooling = r.energy_change < 0 ? "✓" : "✗"
            @printf("%-32s | %11.3f | %10.3f | %9.3f | %s\n", 
                    config_name, r.E_initial/N, r.E_final/N, r.energy_change, cooling)
        end
    end
end

# Compare ED methods
println("\n=== ED Method Comparison ===")
ed_methods = filter(x -> x[1] == "ED", test_configs)
if all(haskey(results, "$backend/$sim/$evol") && !isa(results["$backend/$sim/$evol"], Exception) 
       for (backend, sim, evol) in ed_methods)
    
    println("\nAll ED methods should give similar results:")
    for (_, sim_method, evolution_method) in ed_methods
        config_name = "ED/$sim_method/$evolution_method"
        r = results[config_name]
        println("  $sim_method/$evolution_method: E/N = $(r.E_initial/N) → $(r.E_final/N)")
    end
end