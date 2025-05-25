using CoolingTNS

println("Testing Typed Parameter System")
println("="^50)

# Test parameters
N = 2
coupling = "XX"
g = 0.01
te = 0.1
steps = 3
n_trajectories = 2

println("System size: N = $N")
println("Coupling: $coupling, g = $g")
println("Evolution time per step: te = $te")
println("Number of trajectories: $n_trajectories")
println("Steps: $steps")
println("="^50)

# Create typed coupling parameters
coupling_params = create_coupling_params(coupling, g, steps, te)
println("Created coupling parameters: $(typeof(coupling_params))")
println("  coupling: $(coupling_params.coupling)")
println("  g: $(coupling_params.g)")
println("  steps: $(coupling_params.steps)")
println("  te: $(coupling_params.te)")

# Create typed simulation parameters for Monte Carlo
sim_params = create_sim_params(MonteCarloWavefunction(); n_trajectories=n_trajectories, pe=0.0)
println("\nCreated sim parameters: $(typeof(sim_params))")
println("  n_trajectories: $(sim_params.n_trajectories)")
println("  pe: $(sim_params.pe)")
println("  parallel: $(sim_params.parallel)")

# Test conversion to dict for backward compatibility
coupling_dict = to_dict(coupling_params)
sim_dict = to_dict(sim_params)
println("\nDictionary conversion test:")
println("  coupling_dict: $coupling_dict")
println("  sim_dict: $sim_dict")

# Set up the problem using typed parameters
ham_params = (1.0, -1.05, 0.5)  # J, hx, hz for niIsing problem

try
    println("\n" * "="^50)
    println("Setting up problem with typed parameters...")
    
    # Set up the problem
    problem = setup_problem(EDBackend(), N, "niIsing", ham_params, coupling_params, sim_params)
    println("✓ Problem setup successful")
    
    # Set up initial state
    initial_state = setup_initial_state(problem, "product", 0.0; method=MonteCarloWavefunction())
    println("✓ Initial state setup successful")
    
    # Run cooling with typed parameters directly
    println("\nRunning cooling simulation with typed parameters...")
    
    # Debug: Check what parameters are being passed
    println("Coupling params type: $(typeof(coupling_params))")
    if coupling_params isa Dict
        println("Coupling dict keys: $(keys(coupling_params))")
    end
    
    results = run_cooling_ed(
        problem.H_sys,
        problem.H_sys_bath,
        problem.ϕ₀,
        initial_state.state,
        coupling_params,
        sim_params
    )
    
    println("✓ Simulation completed successfully!")
    println("Results type: $(typeof(results))")
    
    if results isa MonteCarloResults
        println("Monte Carlo specific fields:")
        println("  Number of trajectories: $(results.n_trajectories)")
        println("  Energy trajectory shape: $(size(results.E_trajectories))")
        println("  Final energy: $(results.E_list[end])")
        println("  Final overlap: $(results.GS_overlap_list[end])")
        println("  Energy std at end: $(results.E_std[end])")
    end
    
    # Test backward compatibility by converting to dict
    results_dict = results_to_dict(results)
    println("\n✓ Results successfully converted to dict for backward compatibility")
    println("Dict keys: $(keys(results_dict))")
    
catch e
    println("Error: $e")
    rethrow(e)
end