using JSON, Base64, Hyperopt, Random, Statistics, ArgParse, HDF5
using CoolingTNS


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--search_method"
        help = "method for hyperparameter search (valid choices: Random, Grid, Bayesian)"
        default = "Random"
        "--problem"
        help = "type of problem to solve (valid choices: Ising, niIsing)"
        default = "niIsing"
        "--N"
        help = "number of spins in the system"
        default = 20
        arg_type = Int
        "--J"
        help = "coupling constant J"
        default = 1.0
        arg_type = Float64
        "--h"
        help = "magnetic field h"
        default = 2.0
        arg_type = Float64
        "--hx"
        help = "x-component of the magnetic field"
        default = -1.05
        arg_type = Float64
        "--hz"
        help = "z-component of the magnetic field"
        default = 0.5
        arg_type = Float64
        "--Dmax"
        help = "maximum bond dimension"
        default = 20
        arg_type = Int
        "--cutoff"
        help = "truncation error cutoff"
        default = 1E-8
        arg_type = Float64
        "--num_trials"
        help = "number of trials for the search"
        default = 10
        arg_type = Int
        "--steps"
        help = "number of steps in the cooling simulation"
        default = 50
        arg_type = Int
        "--k"
        help = "number of energy densities to average"
        default = 50
        arg_type = Int
    end
    return parse_args(s)
end

parsed_args = parse_commandline()
println("Parsed args:")
for (arg, val) in parsed_args
    println("  $arg  =>  $val")
end

problem = parsed_args["problem"]  # Problem type: "Ising", "niIsing"

global N = parsed_args["N"]  # Number of spins
global k = parsed_args["k"]  # Number of energy densities to average
global Dmax = parsed_args["Dmax"]  # Maximum bond dimension
global cutoff = parsed_args["cutoff"]  # Truncation error cutoff

global num_trials = parsed_args["num_trials"]  # Number of trials for the search
global steps = parsed_args["steps"]  # Number of steps in the cooling simulation
search_method = parsed_args["search_method"]  # Options: "Random", "Grid", "Bayesian"
method = "MPS"

ham_params, ham_name = CoolingTNS.extract_ham_params(problem, parsed_args)

function run_cooling_simulation(problem, ham_params, params)
    """
    Runs the CoolingTNS.jl simulation with the given parameters and returns the performance metric.
    """
    # Extract parameters from the dictionary
    g = params["g"]
    te = params["te"]

    sites, H_sys, H_sys_bath, ϕ0, ψ_s, e₀ =
        CoolingTNS.setup_problem_mps(problem, N, ham_params, g)

    E_list, GS_overlap_list, nb_list = CoolingTNS.run_cooling_mps(
        sites,
        H_sys,
        H_sys_bath,
        ψ_s,
        ϕ0,
        steps;
        te=te,
        cutoff=cutoff,
        Dmax=Dmax,
    )

    # Calculate the average of the last k energy densities and the last k ground state overlaps as the performance metrics
    Efinal_density_avg = mean(E_list[end-k+1:end]) / N
    GS_overlap_avg = mean(GS_overlap_list[end-k+1:end])

    return Efinal_density_avg, GS_overlap_avg
end

function objective_function(problem, params)
    Efinal_density, _ = run_cooling_simulation(problem, ham_params, params)
    return Efinal_density
end

# Initial guess for the parameters
initial_params = Dict(
    "g" => 0.3,
    "te" => 2.0,
)

# Define the search space for control parameters
search_space =
    Dict("g" => range(0.1, 0.5, length=5), "te" => range(1.0, 3.0, length=5))


if search_method == "Random"
    # best_params, best_objective = CoolingTNS.random_search(problem, objective_function, search_space, num_trials, initial_params)
    best_params, best_objective = CoolingTNS.hyperopt_random_search(problem, objective_function, search_space, num_trials, initial_params)
elseif search_method == "Grid"
    num_iterations = 1
    best_params, best_objective = CoolingTNS.iterative_grid_search(problem, objective_function, search_space, num_iterations, initial_params)
elseif search_method == "Bayesian"
    best_params, best_objective = CoolingTNS.hyperopt_bayesian_optimization(problem, objective_function, search_space, num_trials, initial_params)
else
    error("Invalid search method: $search_method")
end


println("Optimization Result:")


# Save the optimization results to a hdf5 file
filename = "OptimizedCooling_Problem$(ham_name)Ns$(N)Nb$(N)_Search$(search_method)trials$(num_trials)steps$(steps)_Method$(method)Dmax$(Dmax)"

# Perform the simulation with optimal control parameters
global steps = steps * 4  # Increase the number of steps for a better estimate
global k = k * 4
Efinal_density, GS_overlap_final = run_cooling_simulation(problem, ham_params, best_params)

println("Final energy density: ", Efinal_density)
println("Final ground state overlap: ", GS_overlap_final)
println("Optimal control parameters:")
for (param, val) in best_params
    println("$param: $val")
end

h5open("ResultsOpt/$(filename).h5", "w") do file
    write(file, "Final energy density", Efinal_density)
    write(file, "Final ground state overlap", GS_overlap_final)
    for (key, value) in parsed_args
        write(file, string(key), value)
    end
    
    group = create_group(file, "Optimal control parameters")
    for (param, val) in best_params
        write(group, param, val)
    end
end
println("Optimization results saved to: ", filename)
