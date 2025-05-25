#!/usr/bin/env julia

using Pkg
Pkg.activate("..")  # Activate parent project

using CoolingTNS
using PyPlot
using Statistics

println("============================================================")
println("Running TN cooling method comparison")
println("============================================================")

# Common parameters
N = 4
problem = "niIsing"
J = 1.0
hx = -1.05
hz = 0.5
coupling = "XX"
g = 0.05
te = 2.0
steps = 100
Dmax = 64
cutoff = 1e-6

# Storage for results
results = Dict()
methods = []

# 1. TN + Monte Carlo + Continuous
println("\n1. Running MPS with Monte Carlo (continuous evolution)...")
args = Dict(
    "N" => N,
    "problem" => problem,
    "J" => J,
    "hx" => hx,
    "hz" => hz,
    "coupling" => coupling,
    "g" => g,
    "te" => te,
    "steps" => steps,
    "backend" => "TN",
    "sim_method" => "monte_carlo",
    "evolution_method" => "continuous",
    "Dmax" => Dmax,
    "cutoff" => cutoff,
    "init_state" => "product",
    "n_trajectories" => 1,
    "peInt" => 0
)

# Setup parameters using CoolingTNS functions
prob, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(args)
backend = CoolingTNS.TNBackend()
sim_params = CoolingTNS.create_sim_params(
    backend;
    sim_method=CoolingTNS.MonteCarloWavefunction(),
    evolution_method=CoolingTNS.ContinuousEvolution(),
    Dmax=Dmax,
    cutoff=cutoff,
    tau=0.1,
    n_trajectories=1,
    pe=0.0
)

# Setup problem and run
problem, initial_state = CoolingTNS.setup_problem(ham_params, backend, sim_params, coupling_params)
result = CoolingTNS.run_cooling(problem, initial_state, coupling_params, sim_params, ham_params)

# Store results
results["MPS_MC_Continuous"] = result
push!(methods, "MPS + Monte Carlo + Continuous")

# Get ground state energy for comparison
e₀ = CoolingTNS.compute_ground_state_dmrg(ham_params, backend, sim_params)

# 2. TN + Density Matrix + Continuous
println("\n2. Running MPO with Density Matrix (continuous evolution)...")
sim_params_dm = CoolingTNS.create_sim_params(
    backend;
    sim_method=CoolingTNS.DensityMatrix(),
    evolution_method=CoolingTNS.ContinuousEvolution(),
    Dmax=Dmax,
    cutoff=cutoff,
    tau=0.1,
    n_trajectories=1,
    pe=0.0
)

problem_dm, initial_state_dm = CoolingTNS.setup_problem(ham_params, backend, sim_params_dm, coupling_params)
result_dm = CoolingTNS.run_cooling(problem_dm, initial_state_dm, coupling_params, sim_params_dm, ham_params)

results["MPO_DM_Continuous"] = result_dm
push!(methods, "MPO + Density Matrix + Continuous")

# 3. TN + Monte Carlo + Trotter
println("\n3. Running MPS with Monte Carlo (Trotter evolution)...")
sim_params_trotter = CoolingTNS.create_sim_params(
    backend;
    sim_method=CoolingTNS.MonteCarloWavefunction(),
    evolution_method=CoolingTNS.TrotterEvolution(),
    Dmax=Dmax,
    cutoff=cutoff,
    tau=0.1,
    n_trajectories=1,
    pe=0.0
)

problem_trotter, initial_state_trotter = CoolingTNS.setup_problem(ham_params, backend, sim_params_trotter, coupling_params)
result_trotter = CoolingTNS.run_cooling(problem_trotter, initial_state_trotter, coupling_params, sim_params_trotter, ham_params)

results["MPS_MC_Trotter"] = result_trotter
push!(methods, "MPS + Monte Carlo + Trotter")

# Plotting
println("\n4. Generating plots...")

# Create figure with subplots
fig, (ax1, ax2) = subplots(2, 1, figsize=(10, 10))

# Colors for different methods
colors = ["blue", "red", "green"]

# Plot energy evolution
ax1.set_title("Energy Evolution for Different TN Methods", fontsize=14)
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Energy per spin", fontsize=12)
ax1.axhline(y=e₀/N, color="black", linestyle="--", label="Ground state")

for (i, (key, label)) in enumerate(zip(keys(results), methods))
    E_list = results[key]["E_list"]
    steps_array = 0:length(E_list)-1
    ax1.plot(steps_array, E_list ./ N, colors[i], label=label, linewidth=2)
end

ax1.legend(loc="upper right")
ax1.grid(true, alpha=0.3)
ax1.set_ylim([-1.3, -0.5])

# Plot ground state overlap evolution
ax2.set_title("Ground State Overlap Evolution", fontsize=14)
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Ground State Overlap", fontsize=12)

for (i, (key, label)) in enumerate(zip(keys(results), methods))
    overlap_list = results[key]["GS_overlap_list"]
    steps_array = 0:length(overlap_list)-1
    ax2.plot(steps_array, overlap_list, colors[i], label=label, linewidth=2)
end

ax2.legend(loc="lower right")
ax2.grid(true, alpha=0.3)
ax2.set_ylim([0, 1])

# Add text with final values
textstr = "Final Values:\n"
for (key, label) in zip(keys(results), methods)
    E_final = results[key]["E_list"][end] / N
    overlap_final = results[key]["GS_overlap_list"][end]
    textstr *= f"$label:\n  E/N = {E_final:.4f}, Overlap = {overlap_final:.4f}\n"
end
textstr *= f"\nGround state E/N = {e₀/N:.4f}"

# Add text box
props = Dict("boxstyle" => "round", "facecolor" => "wheat", "alpha" => 0.5)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
         verticalalignment="top", bbox=props)

tight_layout()

# Save plot
plot_file = "TN_methods_comparison_N$(N)_steps$(steps).pdf"
savefig(plot_file)
println("\nPlot saved to: $plot_file")

# Also save individual plots for each method
for (i, (key, label)) in enumerate(zip(keys(results), methods))
    fig2, (ax3, ax4) = subplots(2, 1, figsize=(8, 8))
    
    # Energy
    E_list = results[key]["E_list"]
    steps_array = 0:length(E_list)-1
    ax3.plot(steps_array, E_list ./ N, colors[i], linewidth=2)
    ax3.axhline(y=e₀/N, color="black", linestyle="--", label="Ground state")
    ax3.set_title("Energy Evolution - $label", fontsize=12)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Energy per spin")
    ax3.grid(true, alpha=0.3)
    ax3.legend()
    
    # Overlap
    overlap_list = results[key]["GS_overlap_list"]
    ax4.plot(steps_array, overlap_list, colors[i], linewidth=2)
    ax4.set_title("Ground State Overlap - $label", fontsize=12)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Overlap")
    ax4.grid(true, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    tight_layout()
    
    # Save individual plot
    method_file = replace(key, "_" => "-") * "_N$(N).pdf"
    savefig(method_file)
    println("Saved individual plot: $method_file")
end

println("\n============================================================")
println("Test completed successfully!")
println("============================================================")