#!/usr/bin/env julia

# Quick test of single TN vs ED comparison
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

println("=== Single TN vs ED Comparison ===")
println("System: N=$N, Ising model, h=-2.0")
println("Coupling: $COUPLING, g=$G")
println("Evolution: te=$TE, steps=$STEPS")
println("Testing: ED/density_matrix/continuous")
println()

# Test ED density matrix continuous
test_args = copy(base_args)
test_args["backend"] = "ED"
test_args["sim_method"] = "density_matrix"
test_args["evolution_method"] = "continuous"

# Setup parameters
problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(test_args)

# Get backend and create simulation parameters
backend_obj = CoolingTNS.EDBackend()
sim_params = CoolingTNS.UnifiedSimulationParameters(
    CoolingTNS.DensityMatrix(), 
    CoolingTNS.ContinuousEvolution();
    tau = TAU,
    n_trajectories = 10,
    Dmax = 100
)

# Setup problem
cooling_problem = CoolingTNS.setup_problem(backend_obj, ham_params, coupling_params, sim_params)

# Initial state
initial_state = CoolingTNS.setup_initial_state(cooling_problem, sim_params, test_args["init_state"], test_args["theta"])

# Get initial energy
E_initial = CoolingTNS.expect_ed(cooling_problem.H_sys, initial_state.state)
println("Initial energy: $E_initial (E/N = $(E_initial/N))")

# Run cooling
println("Running cooling...")
result = CoolingTNS.run_cooling(cooling_problem, initial_state, coupling_params, sim_params, ham_params)

# Get final energy
E_final = result["E_list"][end]
println("Final energy: $E_final (E/N = $(E_final/N))")
println("Energy change: $(E_final - E_initial)")
println("Cooling: $(E_final < E_initial ? "✓" : "✗")")

# Show energy evolution
println("\nEnergy evolution:")
for (i, E) in enumerate(result["E_list"])
    println("  Step $i: E/N = $(E/N)")
end