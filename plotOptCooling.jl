using CoolingTNS

method = "MPS"
parsed_args = CoolingTNS.parse_commandline()

N, problem, ham_params, ham_name, pe, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
sim_params = CoolingTNS.create_sim_params(parsed_args, pe, method)

sites, H_sys, ϕ₀, e₀, H_sys_bath = CoolingTNS.setup_problem_mps(problem, N, ham_params, coupling_params, sim_params)
println("The ground state energy density is e₀/N = $(e₀/N)")

search_params = Dict(
    "search_method" => parsed_args["search_method"],
    "num_trials" => parsed_args["num_trials"]
)

# Set the desired system sizes for plotting
N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Plot energy error and final overlap vs system size
CoolingTNS.plot_vs_N(ham_name, coupling_params, sim_params, N_values, e₀; is_optimization=true)

# Uncomment the following lines to plot for multiple pe values
# pe_values = 0:10
# CoolingTNS.plot_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, pe_values, e₀; is_optimization=true)

