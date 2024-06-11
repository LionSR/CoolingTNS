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

N = parsed_args["N"]
pe = parsed_args["pe"]
peInt = parsed_args["peInt"]
if peInt > 0
    pe = peInt * 1e-3
    pe = round(pe, digits=4)
end

sim_params = Dict(
    "method" => parsed_args["method"],
    "cutoff" => parsed_args["cutoff"],
    "Dmax" => parsed_args["Dmax"],
    "tau" => parsed_args["tau"],
    "pe" => pe,
    "peInt" => peInt
)

coupling_params = Dict(
    "g" => parsed_args["g"],
    "te" => parsed_args["te"],
    "steps" => parsed_args["steps"],
    "coupling" => parsed_args["coupling"]
)

sites, H_sys, ϕ₀, e₀, H_sys_bath = CoolingTNS.setup_problem_mps(problem, N, ham_params, coupling_params, sim_params)
println("The ground state energy density is e₀/N = $(e₀/N)")

search_params = Dict(
    "search_method" => parsed_args["search_method"],
    "num_trials" => parsed_args["num_trials"]
)

# Set the desired system sizes for plotting
N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# N_values = [10, 20, 30, 40, 50, 60, 70]
# N_values = [10, 20, 30, 40, 50, 60]
# N_values = [20, 30, 40, 50, 60]

# Plot energy error and final overlap vs system size
CoolingTNS.plotOptimal_energy_error_and_overlap_vs_N(ham_name, coupling_params, sim_params, search_params, N_values, e₀)

# pe_values = [0,1,3,4,5,6,9]

# Call the plotting function with a range of peInt values
# CoolingTNS.plotOptimal_energy_error_and_overlap_vs_N_pe_range(ham_name, coupling_params, sim_params, search_params, N_values, pe_values, e₀)

