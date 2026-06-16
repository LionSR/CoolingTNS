using CoolingTNS

include(joinpath(@__DIR__, "scripts", "plotting", "plotting.jl"))

parsed_args = CoolingTNS.parse_commandline()
CoolingTNS.normalize_optimization_args!(parsed_args)

# Common parameters (typed)
problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)

# Optimization plots must use the same backend and method naming convention as
# the optimization driver, otherwise the expected result filenames differ.
backend = CoolingTNS.get_backend(parsed_args["backend"])
sim_params = CoolingTNS.create_sim_params_from_args(parsed_args)

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
