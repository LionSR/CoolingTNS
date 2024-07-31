using CoolingTNS

parsed_args = CoolingTNS.parse_commandline()

N, problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
sim_params = CoolingTNS.create_sim_params(parsed_args)

# Set the desired system sizes for plotting
N_values = [10, 20, 30, 40, 50]

# Plot energy error and final overlap vs system size
# CoolingTNS.plot_energy_error_and_overlap_vs_N(ham_name, coupling_params, sim_params, N_values)

# Call the plotting function with a range of peInt values
peInt_range = 0:10
CoolingTNS.plot_energy_error_and_overlap_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, peInt_range)
