#!/usr/bin/env julia

# Comprehensive comparison of TN vs ED backends
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
    "te" => TE,
    "coupling" => COUPLING,
    "steps" => STEPS,
    "tau" => TAU,
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

# Test configurations
test_configs = [
    ("TN", "density_matrix", "continuous"),
    ("TN", "density_matrix", "trotter"),
    ("TN", "monte_carlo", "continuous"),
    ("TN", "monte_carlo", "trotter"),
    ("ED", "density_matrix", "continuous"),
    ("ED", "density_matrix", "trotter"),
    ("ED", "monte_carlo", "continuous"),
    ("ED", "monte_carlo", "trotter"),
]

# Results storage
results = Dict()

println("=== Comprehensive TN vs ED Comparison ===")
println("System: N=$N, Ising model, h=-2.0")
println("Coupling: $COUPLING, g=$G")
println("Evolution: te=$TE, steps=$STEPS")
println()

# Run each configuration
for (backend, sim_method, evolution_method) in test_configs
    config_name = "$backend/$sim_method/$evolution_method"
    
    # Skip TN density matrix continuous (not supported by TDVP)
    if backend == "TN" && sim_method == "density_matrix" && evolution_method == "continuous"
        println("$config_name: SKIPPED (TDVP doesn't support MPO evolution)")
        continue
    end
    
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
        problem = CoolingTNS.setup_problem(backend_obj, ham_params, coupling_params, sim_params)
        
        # Initial state
        initial_state = CoolingTNS.setup_initial_state(problem, sim_params, test_args["init_state"], test_args["theta"])
        
        # Get initial energy
        if backend == "ED"
            E_initial = CoolingTNS.expect_ed(problem.H_sys, initial_state.state)
        else
            # For TN, need to handle MPS case
            E_initial = real(inner(initial_state.state', problem.H_sys, initial_state.state))
        end
        
        # Run cooling
        result = CoolingTNS.run_cooling(problem, initial_state, coupling_params, sim_params, ham_params)
        
        # Store results
        E_final = result["E_list"][end]
        overlap_final = result["GS_overlap_list"][end]
        
        results[config_name] = (
            E_initial = E_initial,
            E_final = E_final,
            overlap_final = overlap_final,
            energy_change = E_final - E_initial,
            ground_energy = problem.e₀
        )
        
        println("E/N: $(E_initial/N) → $(E_final/N), ΔE=$(E_final - E_initial)")
        
    catch e
        println("ERROR: $e")
        results[config_name] = (error = e)
    end
end

# Summary
println("\n=== Summary ===")
println("Ground state energy/N ≈ $(-2.094)")  # Approximate for N=4 Ising
println("\nMethod                           | Initial E/N | Final E/N  | ΔE        | Cooling?")
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
    else
        @printf("%-32s | SKIPPED\n", config_name)
    end
end

# Direct comparison
println("\n=== Direct TN vs ED Comparison ===")
println("(Only comparing methods that exist for both backends)")

for (_, sim_method, evolution_method) in test_configs[2:end]  # Skip first TN/DM/Cont
    tn_name = "TN/$sim_method/$evolution_method"
    ed_name = "ED/$sim_method/$evolution_method"
    
    if haskey(results, tn_name) && haskey(results, ed_name) && 
       !isa(results[tn_name], Exception) && !isa(results[ed_name], Exception)
        
        tn = results[tn_name]
        ed = results[ed_name]
        
        println("\n$sim_method/$evolution_method:")
        println("  TN: E/N = $(tn.E_initial/N) → $(tn.E_final/N)")
        println("  ED: E/N = $(ed.E_initial/N) → $(ed.E_final/N)")
        println("  Initial energy match: $(abs(tn.E_initial - ed.E_initial) < 1e-6 ? "✓" : "✗ (diff=$(abs(tn.E_initial - ed.E_initial)))")")
        println("  Both cooling: $((tn.energy_change < 0 && ed.energy_change < 0) ? "✓" : "✗")")
    end
end