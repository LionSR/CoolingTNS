using CoolingTNS

parsed_args = CoolingTNS.parse_commandline()

# Setup common parameters using new dispatch architecture
problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)

# Set backend and simulation method for optimization plots (default to TN with MPS)
backend = CoolingTNS.TNBackend()
sim_method = CoolingTNS.MonteCarloWavefunction()
evolution_method = CoolingTNS.ContinuousEvolution()

# Create simulation parameters
sim_params = CoolingTNS.create_sim_params(backend; 
    sim_method=sim_method, 
    evolution_method=evolution_method,
    Dmax=parsed_args["Dmax"], 
    cutoff=parsed_args["cutoff"], 
    tau=parsed_args["tau"], 
    pe=parsed_args["peInt"]*1e-3,
    n_trajectories=parsed_args["n_trajectories"])

# Setup problem using unified interface
cooling_problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
println("The ground state energy density is e₀/N = $(cooling_problem.e₀/ham_params.N)")

search_params = Dict(
    "search_method" => parsed_args["search_method"],
    "num_trials" => parsed_args["num_trials"]
)

# Set the desired system sizes for plotting
N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Plot energy error and final overlap vs system size
CoolingTNS.plot_vs_N(ham_name, coupling_params, sim_params, N_values, cooling_problem.e₀; is_optimization=true)

# Uncomment the following lines to plot for multiple pe values
# pe_values = 0:10
# CoolingTNS.plot_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, pe_values, cooling_problem.e₀; is_optimization=true)

