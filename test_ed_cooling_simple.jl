using CoolingTNS

println("Testing ED Cooling with Simple Parameters")
println("="^50)

# Simple parameters to avoid numerical issues
N = 2  # Smaller system
coupling = "XX"
g = 0.01  # Much smaller coupling
te = 0.1  # Shorter time steps
steps = 5  # Fewer steps
n_trajectories = 2  # Fewer trajectories

println("System size: N = $N")
println("Coupling: $coupling, g = $g")
println("Evolution time per step: te = $te")
println("Number of trajectories: $n_trajectories")
println("Steps: $steps")
println("="^50)

# Set up the problem
ham_params = (1.0, -1.05, 0.5)  # J, hx, hz for niIsing problem

# Coupling parameters
coupling_params = Dict(
    "coupling" => coupling,
    "g" => g,
    "steps" => steps,
    "te" => te
)

# Simulation parameters
sim_params = Dict{String, Real}(
    "n_trajectories" => n_trajectories,
    "pe" => 0.0  # No noise
)

# Set up the problem using the proper function
problem = setup_problem(EDBackend(), N, "niIsing", ham_params, coupling_params, sim_params)

# Monte Carlo Wavefunction initial state  
initial_state = setup_initial_state(problem, "product", 0.0; method=MonteCarloWavefunction())

try
    println("Running ED cooling with Monte Carlo wavefunction method ($n_trajectories trajectories)")
    
    # Run the cooling simulation
    results = run_cooling(problem, initial_state, coupling_params, sim_params, ham_params)
    
    println("Success! Energy evolution:")
    println("Initial energy: $(results["E_list"][1])")
    println("Final energy: $(results["E_list"][end])")
    println("Energy change: $(results["E_list"][end] - results["E_list"][1])")
    
    if results["E_list"][end] < results["E_list"][1]
        println("✓ Energy decreased during cooling!")
    else
        println("⚠ Energy did not decrease")
    end
    
    println("\nGround state overlap evolution:")
    println("Initial overlap: $(results["GS_overlap_list"][1])")
    println("Final overlap: $(results["GS_overlap_list"][end])")
    
catch e
    println("Error: $e")
    rethrow(e)
end