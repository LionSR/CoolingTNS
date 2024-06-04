using ArgParse
using CoolingTNS

# Parse command-line arguments
parsed_args = CoolingTNS.parse_commandline()
println("Parsed args:")
for (arg, val) in parsed_args
    println("  $arg  =>  $val")
end

# Unpack parsed arguments
problem = parsed_args["problem"]
ham_params, ham_name = CoolingTNS.extract_ham_params(problem, parsed_args)

pe = parsed_args["pe"]
if parsed_args["peInt"] > 0
    pe = parsed_args["peInt"] * 1e-3
    pe = round(pe, digits=4)
end

sim_params = Dict(
    "method" => parsed_args["method"],
    "cutoff" => parsed_args["cutoff"],
    "Dmax" => parsed_args["Dmax"],
    "tau" => parsed_args["tau"],
    "pe" => pe
)

coupling_params = Dict(
    "g" => parsed_args["g"],
    "te" => parsed_args["te"],
    "steps" => parsed_args["steps"],
    "coupling" => parsed_args["coupling"]
)

# Set the desired system sizes for plotting
# N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
N_values = [10, 30, 40, 50, 60, 70, 80, 90, 100]
# N_values = [40, 50, 60, 70, 80, 90, 100]


# Plot energy error and final overlap vs system size
# CoolingTNS.plot_energy_error_and_overlap_vs_N(ham_name, coupling_params, sim_params, N_values)

# Call the plotting function with a range of peInt values
CoolingTNS.plot_energy_error_and_overlap_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, 0:9)
# CoolingTNS.plot_energy_error_and_overlap_vs_N_pe_range(ham_name, coupling_params, sim_params, N_values, 0:10)