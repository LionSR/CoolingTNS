using CoolingTNS

parsed_args = CoolingTNS.parse_commandline()

# Common parameters (typed)
problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)

# Backend + simulation parameters
backend = CoolingTNS.get_backend(parsed_args["backend"])
sim_params = CoolingTNS.create_sim_params(
    backend;
    sim_method=CoolingTNS.get_sim_method(parsed_args["sim_method"]),
    evolution_method=CoolingTNS.get_evolution_method(parsed_args["evolution_method"]),
    Dmax=parsed_args["Dmax"],
    cutoff=parsed_args["cutoff"],
    tau=parsed_args["tau"],
    pe=parsed_args["peInt"] * 1e-3,
    n_trajectories=parsed_args["n_trajectories"],
)

# System sizes for scaling plots
N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# # Final energy density and overlap vs system size
# CoolingTNS.plot_vs_N(ham_params, coupling_params, sim_params, backend, N_values)

# # Energy error and overlap vs system size for multiple noise levels
# peInt_range = 0:10
# CoolingTNS.plot_vs_N_pe_range(ham_params, coupling_params, sim_params, backend, N_values, peInt_range)

# Cooling curves for multiple noise strengths
peInt_range = 0:1:10
CoolingTNS.plot_cooling_curve_noise(ham_params, coupling_params, sim_params, peInt_range; backend=backend)
