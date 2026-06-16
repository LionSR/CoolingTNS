using CoolingTNS

include(joinpath(@__DIR__, "scripts", "plotting", "plotting.jl"))

parsed_args = CoolingTNS.parse_commandline()

# Common parameters (typed)
problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)

# Optimization runs are typically TN + MC + continuous, but you can override these.
backend = CoolingTNS.TNBackend()
sim_method = CoolingTNS.MonteCarloWavefunction()
evolution_method = CoolingTNS.ContinuousEvolution()

sim_params = CoolingTNS.create_sim_params(
    backend;
    sim_method=sim_method,
    evolution_method=evolution_method,
    Dmax=parsed_args["Dmax"],
    cutoff=parsed_args["cutoff"],
    tau=parsed_args["tau"],
    pe=parsed_args["peInt"] * 1e-3,
    n_trajectories=parsed_args["n_trajectories"],
)

# Used to form the optimization filename suffix: "_Search<method>trials<num_trials>"
search_params = Dict(
    "search_method" => parsed_args["search_method"],
    "num_trials" => parsed_args["num_trials"],
)

# System sizes for scaling plots
N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Plot energy density and final overlap vs system size
plot_vs_N(
    ham_params,
    coupling_params,
    sim_params,
    backend,
    N_values;
    is_optimization=true,
    search_params=search_params,
)

# # Uncomment to scan multiple noise levels (peInt)
# pe_values = 0:10
# plot_vs_N_pe_range(
#     ham_params,
#     coupling_params,
#     sim_params,
#     backend,
#     N_values,
#     pe_values;
#     is_optimization=true,
#     search_params=search_params,
# )
