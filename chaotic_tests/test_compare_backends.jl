using ITensors
using ITensorMPS
using LinearAlgebra
using SparseArrays
using KrylovKit

# Include necessary files
include("src/CoolingTNS.jl")
using .CoolingTNS

# Small test case that should run quickly
N = 2
problem = "Ising"
coupling = "XX"
g = 0.1
te = 0.5
steps = 2

# Common parameters
params_base = Dict{String, Any}(
    "N" => N,
    "problem" => problem,
    "coupling" => coupling,
    "g" => g,
    "te" => te,
    "steps" => steps,
    "init_state" => "product",
    "J" => 1.0,
    "h" => 2.0,
    "hz" => 0.5,
    "hx" => -1.05,
    "tau" => 0.1,
    "Dmax" => 20,
    "cutoff" => 1e-6,
    "peInt" => 0,
    "n_trajectories" => 1,
    "num_trials" => 10,
    "search_method" => "Random",
    "window_size" => 50
)

println("=== Comparing TN and ED Backends ===")
println("System: N=$N, $problem model, $coupling coupling, g=$g")
println()

# Test ED backend with density matrix
println("--- ED Backend (Density Matrix, Continuous) ---")
params_ed = copy(params_base)
params_ed["backend"] = "ED"
params_ed["sim_method"] = "density_matrix"
params_ed["evolution_method"] = "continuous"

parsed_params_ed = CoolingTNS.parse_commandline(params_ed)
common_params_ed = CoolingTNS.setup_common_parameters(parsed_params_ed)

# Run one step
problem_ed = CoolingTNS.setup_problem_unified(common_params_ed)
initial_state_ed = CoolingTNS.setup_initial_state(parsed_params_ed["init_state"], problem_ed, common_params_ed)

# Measure initial energy
initial_energy_ed = CoolingTNS.measure_energy(initial_state_ed, problem_ed)
println("Initial energy/N: $(initial_energy_ed/N)")

# Run cooling for one step
evolved_state_ed = CoolingTNS.run_cooling_step(initial_state_ed, problem_ed, common_params_ed)
final_energy_ed = CoolingTNS.measure_energy(evolved_state_ed, problem_ed)
println("After 1 step energy/N: $(final_energy_ed/N)")
println("Energy change: $(final_energy_ed - initial_energy_ed)")

# Test TN backend with density matrix
println("\n--- TN Backend (Density Matrix, Continuous) ---")
params_tn = copy(params_base)
params_tn["backend"] = "TN"
params_tn["sim_method"] = "density_matrix"  
params_tn["evolution_method"] = "continuous"

parsed_params_tn = CoolingTNS.parse_commandline(params_tn)
common_params_tn = CoolingTNS.setup_common_parameters(parsed_params_tn)

# Run one step
problem_tn = CoolingTNS.setup_problem_unified(common_params_tn)
initial_state_tn = CoolingTNS.setup_initial_state(parsed_params_tn["init_state"], problem_tn, common_params_tn)

# Measure initial energy
initial_energy_tn = CoolingTNS.measure_energy(initial_state_tn, problem_tn)
println("Initial energy/N: $(initial_energy_tn/N)")

# Run cooling for one step
evolved_state_tn = CoolingTNS.run_cooling_step(initial_state_tn, problem_tn, common_params_tn)
final_energy_tn = CoolingTNS.measure_energy(evolved_state_tn, problem_tn)
println("After 1 step energy/N: $(final_energy_tn/N)")
println("Energy change: $(final_energy_tn - initial_energy_tn)")

println("\n=== Summary ===")
println("ED Backend: $(initial_energy_ed/N) → $(final_energy_ed/N) (change: $(final_energy_ed - initial_energy_ed))")
println("TN Backend: $(initial_energy_tn/N) → $(final_energy_tn/N) (change: $(final_energy_tn - initial_energy_tn))")
println("\nGround state energy/N ≈ $(problem_ed.ground_energy/N)")
println("\nExpected behavior: Energy should decrease (cooling)")
if final_energy_ed > initial_energy_ed
    println("❌ ED Backend is heating instead of cooling!")
else
    println("✅ ED Backend is cooling correctly")
end
if final_energy_tn > initial_energy_tn  
    println("❌ TN Backend is heating instead of cooling!")
else
    println("✅ TN Backend is cooling correctly")
end