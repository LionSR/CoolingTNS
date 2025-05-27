#!/usr/bin/env julia

using ITensors
using ITensorMPS
using LinearAlgebra
using SparseArrays

# Include main module
include("src/CoolingTNS.jl")
using .CoolingTNS

# Simple test comparing ED and TN backends
N = 3
println("=== Simple Backend Comparison (N=$N) ===")
println("Testing density matrix + continuous evolution")
println("Initial state: all up |000⟩")

# Common arguments
base_args = [
    "--N", "$N",
    "--problem", "Ising",
    "--coupling", "XX",
    "--g", "0.1",
    "--te", "0.5",
    "--steps", "5",
    "--sim_method", "density_matrix",
    "--evolution_method", "continuous",
    "--init_state", "product"  # This should now be all-up
]

println("\n--- Running ED Backend ---")
args_ed = vcat(base_args, ["--backend", "ED"])
parsed_ed = CoolingTNS.parse_commandline(args_ed)

# Manually run to capture intermediate results
problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_ed)
backend_ed = CoolingTNS.get_backend(parsed_ed["backend"])

# Create sim params
sim_method = CoolingTNS.DensityMatrix()
evolution_method = CoolingTNS.ContinuousEvolution()
sim_params_ed = CoolingTNS.UnifiedSimulationParameters(
    sim_method, evolution_method;
    Dmax=parsed_ed["Dmax"],
    cutoff=parsed_ed["cutoff"],
    tau=parsed_ed["tau"],
    pe=parsed_ed["peInt"]*1e-3,
    n_trajectories=parsed_ed["n_trajectories"]
)

# Setup problem
cooling_problem_ed = CoolingTNS.setup_problem(backend_ed, ham_params, coupling_params, sim_params_ed)

# Check initial state
initial_state_ed = CoolingTNS.setup_initial_state(
    cooling_problem_ed,
    sim_params_ed,
    parsed_ed["init_state"],
    parsed_ed["theta"]
)

println("Initial state type: $(typeof(initial_state_ed.state))")

# Check if it's really all-up
if isa(initial_state_ed.state, CoolingTNS.EDDensityMatrix)
    ρ_init = initial_state_ed.state.data
    println("Initial density matrix diagonal:")
    for i in 1:min(8, size(ρ_init, 1))
        if abs(ρ_init[i,i]) > 1e-10
            state_str = string(i-1, base=2, pad=N)
            println("  |$state_str⟩: $(ρ_init[i,i])")
        end
    end
end

# Measure initial energy
E_init_ed = CoolingTNS.expect_ed(cooling_problem_ed.H_sys, initial_state_ed.state)
println("\nInitial energy/N: $(E_init_ed/N)")

# Run cooling
results_ed = CoolingTNS.run_cooling(
    cooling_problem_ed,
    initial_state_ed,
    coupling_params,
    sim_params_ed,
    ham_params
)

E_final_ed = CoolingTNS.mean_last_window(results_ed["E_list"], 50)
println("Final energy/N: $(E_final_ed/N)")
println("Ground state energy/N: $(cooling_problem_ed.e₀/N)")

if E_final_ed > E_init_ed
    println("❌ ED Backend is HEATING!")
else
    println("✅ ED Backend is cooling")
end

# Now test TN backend
println("\n\n--- Running TN Backend ---")
args_tn = vcat(base_args, ["--backend", "TN", "--Dmax", "64"])
parsed_tn = CoolingTNS.parse_commandline(args_tn)

backend_tn = CoolingTNS.get_backend(parsed_tn["backend"])
sim_params_tn = CoolingTNS.UnifiedSimulationParameters(
    sim_method, evolution_method;
    Dmax=parsed_tn["Dmax"],
    cutoff=parsed_tn["cutoff"],
    tau=parsed_tn["tau"],
    pe=parsed_tn["peInt"]*1e-3,
    n_trajectories=parsed_tn["n_trajectories"]
)

cooling_problem_tn = CoolingTNS.setup_problem(backend_tn, ham_params, coupling_params, sim_params_tn)
initial_state_tn = CoolingTNS.setup_initial_state(
    cooling_problem_tn,
    sim_params_tn,
    parsed_tn["init_state"],
    parsed_tn["theta"]
)

# Check TN initial state
if isa(initial_state_tn.state, MPO)
    # For MPO, we can check by converting small system to matrix
    println("TN initial state is MPO")
end

# Measure initial energy
E_init_tn = real(inner(initial_state_tn.state, cooling_problem_tn.H_sys))
println("\nInitial energy/N: $(E_init_tn/N)")

# Run cooling
results_tn = CoolingTNS.run_cooling(
    cooling_problem_tn,
    initial_state_tn,
    coupling_params,
    sim_params_tn,
    ham_params
)

E_final_tn = CoolingTNS.mean_last_window(results_tn["E_list"], 50)
println("Final energy/N: $(E_final_tn/N)")

if E_final_tn > E_init_tn
    println("❌ TN Backend is HEATING!")
else
    println("✅ TN Backend is cooling")
end

# Compare results
println("\n\n=== Comparison ===")
println("Initial energy/N: ED=$(E_init_ed/N), TN=$(E_init_tn/N)")
println("Final energy/N: ED=$(E_final_ed/N), TN=$(E_final_tn/N)")
println("Ground state energy/N: $(cooling_problem_ed.e₀/N)")

# Plot energy evolution
println("\nEnergy evolution:")
println("Step  ED energy/N   TN energy/N")
for i in 1:min(length(results_ed["E_list"]), length(results_tn["E_list"]))
    @printf("%3d   %10.6f   %10.6f\n", i, results_ed["E_list"][i]/N, results_tn["E_list"][i]/N)
end