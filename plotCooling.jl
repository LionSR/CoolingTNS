using CoolingTNS

# Helper function to create simulation parameters from new argument structure  
function create_sim_params_new(parsed_args)
    backend = CoolingTNS.get_backend(parsed_args["backend"])
    
    sim_method_str = parsed_args["sim_method"]
    evolution_method_str = parsed_args["evolution_method"]
    
    sim_method = if sim_method_str == "density_matrix"
        CoolingTNS.DensityMatrix()
    elseif sim_method_str == "monte_carlo"
        CoolingTNS.MonteCarloWavefunction()
    else
        error("Unknown simulation method: $sim_method_str")
    end
    
    evolution_method = if evolution_method_str == "continuous"
        CoolingTNS.ContinuousEvolution()
    elseif evolution_method_str == "trotter"
        CoolingTNS.TrotterEvolution()
    else
        error("Unknown evolution method: $evolution_method_str")
    end
    
    return CoolingTNS.create_sim_params(backend; sim_method=sim_method, evolution_method=evolution_method, 
                                       Dmax=parsed_args["Dmax"], cutoff=parsed_args["cutoff"], 
                                       tau=parsed_args["tau"], pe=parsed_args["peInt"]*1e-3,
                                       n_trajectories=parsed_args["n_trajectories"])
end

parsed_args = CoolingTNS.parse_commandline()

N, problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
sim_params = create_sim_params_new(parsed_args)

# Set the desired system sizes for plotting
N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# # Plot energy error and final overlap vs system size
# CoolingTNS.plot_vs_N(ham_name, coupling_params, sim_params, N_values)

# Call the plotting function with a range of peInt values
# peInt_range = 0:10
# CoolingTNS.plot_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, peInt_range)

# Plot cooling curve for different noise strengths
# N = parsed_args["N"]
peInt_range = 0:1:10  # Range of noise strengths to plot
CoolingTNS.plot_cooling_curve_noise(ham_name, N, coupling_params, sim_params, peInt_range)
