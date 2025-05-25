using CoolingTNS
using LinearAlgebra

println("Testing ED Cooling with Monte Carlo Wavefunction Method")
println("==================================================")

# Test with small system
N = 3
problem = "niIsing"
coupling = "XX"
g = 0.2
te = 2.0
steps = 20
Dmax = 100  # Not used for ED
J = 1.0
hx = -1.05
hz = 0.5
n_trajectories = 10  # Number of Monte Carlo trajectories

println("System size: N = $N")
println("Coupling: $coupling, g = $g")
println("Evolution time per step: te = $te")
println("Number of trajectories: $n_trajectories")
println("==================================================")

# Setup parameters directly
ham_params = problem == "niIsing" ? (J, hx, hz) : (J, hx)
coupling_params = Dict(
    "coupling" => coupling,
    "g" => g,
    "te" => te,
    "steps" => steps,
    "Δ" => -1.5  # Set a reasonable detuning
)
sim_params = Dict{String,Real}(
    "Dmax" => Dmax,
    "pe" => 0.0,
    "n_trajectories" => n_trajectories
)
sim_method = CoolingTNS.MonteCarloWavefunction()

# Get the backend
backend = CoolingTNS.EDBackend()

# Setup the problem
problem_obj = CoolingTNS.setup_problem(backend, N, problem, ham_params, coupling_params, sim_params)

# Create initial state
initial_state = CoolingTNS.setup_initial_state(problem_obj, "product", 0.0; method=sim_method)

# Run cooling simulation
results = CoolingTNS.run_cooling(problem_obj, initial_state, coupling_params, sim_params, ham_params)

# Show results
println("\nCooling simulation completed!")
println("Method: $(results["method"])")
println("Number of trajectories: $(results["n_trajectories"])")
println("\nEnergy evolution (average over trajectories):")
E_list = results["E_list"]
for i in 1:min(length(E_list), 10)
    println("Step $(i-1): E/N = $(E_list[i]/N)")
end
if length(E_list) > 10
    println("...")
    println("Step $(length(E_list)-1): E/N = $(E_list[end]/N)")
end

println("\nGround state overlap evolution:")
GS_list = results["GS_overlap_list"]
for i in 1:min(length(GS_list), 10)
    println("Step $(i-1): overlap = $(GS_list[i])")
end
if length(GS_list) > 10
    println("...")
    println("Step $(length(GS_list)-1): overlap = $(GS_list[end])")
end

# Check if energy is decreasing
energy_decreasing = true
for i in 2:length(E_list)
    if E_list[i] > E_list[i-1] + 1e-10
        energy_decreasing = false
        break
    end
end

println("\nEnergy decreasing monotonically: $energy_decreasing")
println("Initial energy: E/N = $(E_list[1]/N)")
println("Final energy: E/N = $(E_list[end]/N)")
println("Energy reduction: ΔE/N = $((E_list[end] - E_list[1])/N)")