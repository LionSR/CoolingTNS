using CoolingTNS

parsed_args = CoolingTNS.parse_commandline()

N, problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
sim_params = CoolingTNS.create_sim_params(parsed_args)

# Set the desired system sizes for plotting
N_values = [10, 20, 30, 40]

# # Plot energy error and final overlap vs system size
# CoolingTNS.plot_vs_N(ham_name, coupling_params, sim_params, N_values)

# # Call the plotting function with a range of peInt values
# peInt_range = 0:10
# CoolingTNS.plot_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, peInt_range)

# Plot cooling curve for different noise strengths
N = parsed_args["N"]
peInt_range = 0:1:10  # Range of noise strengths to plot
CoolingTNS.plot_cooling_curve_noise(ham_name, N, coupling_params, sim_params, peInt_range)
