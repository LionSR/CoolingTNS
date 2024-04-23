using Hyperopt, Random, Statistics, ArgParse, HDF5
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

problem = parsed_args["problem"]
N = parsed_args["N"]
k = parsed_args["k"]
Dmax = parsed_args["Dmax"]
cutoff = parsed_args["cutoff"]
num_trials = parsed_args["num_trials"]
steps = parsed_args["steps"]
search_method = parsed_args["search_method"]
method = "MPS"

ham_params, ham_name = CoolingTNS.extract_ham_params(problem, parsed_args)

sim_params = Dict(
    "cutoff" => cutoff,
    "Dmax" => Dmax,
    "k" => k
)

init_coupling_params = Dict(
    "g" => 0.3,
    "te" => 2.0,
    "steps" => steps,
    "coupling" => "XX"
)

function run_cooling_simulation(problem, ham_params, coupling_params)
    g = coupling_params["g"]
    te = coupling_params["te"]
    
    sites, H_sys, ϕ₀, e₀, H_sys_bath = CoolingTNS.setup_problem_mps(problem, N, ham_params, coupling_params, sim_params)
    ψ_s = CoolingTNS.setup_init_state_mps(sites)

    E_list, GS_overlap_list, nb_list = CoolingTNS.run_cooling_mps(
        sites,
        H_sys,
        ϕ₀,
        H_sys_bath,
        ψ_s,
        coupling_params,
        sim_params
    )

    Efinal_density_avg = mean(E_list[end-k+1:end]) / N
    GS_overlap_avg = mean(GS_overlap_list[end-k+1:end])

    return Efinal_density_avg, GS_overlap_avg
end

function objective_function(problem, coupling_params)
    Efinal_density, _ = run_cooling_simulation(problem, ham_params, coupling_params)
    return Efinal_density
end

search_space = Dict("g" => range(0.1, 0.5, length=5), "te" => range(1.0, 3.0, length=5))

if search_method == "Random"
    best_coupling_params, best_objective = CoolingTNS.hyperopt_random_search(problem, objective_function, search_space, num_trials, init_coupling_params)
elseif search_method == "Grid"
    num_iterations = 1
    best_coupling_params, best_objective = CoolingTNS.iterative_grid_search(problem, objective_function, search_space, num_iterations, init_coupling_params)
elseif search_method == "Bayesian"
    best_coupling_params, best_objective = CoolingTNS.hyperopt_bayesian_optimization(problem, objective_function, search_space, num_trials, init_coupling_params)
else
    error("Invalid search method: $search_method")
end

println("Optimization Result:")
for (param, val) in best_coupling_params
    println("$param: $val")
end

filename = "OptimizedCooling_Problem$(ham_name)Ns$(N)Nb$(N)_Paramssteps$(steps)_Sim$(method)Dmax$(Dmax)_Search$(search_method)trials$(num_trials)"

best_coupling_params["steps"] = steps * 4

Efinal_density, GS_overlap_final = run_cooling_simulation(problem, ham_params, best_coupling_params)
println("Final energy density: ", Efinal_density)
println("Final ground state overlap: ", GS_overlap_final)
println("Optimal control parameters:")

h5open("ResultsOpt/$(filename).h5", "w") do file
    write(file, "Final energy density", Efinal_density)
    write(file, "Final ground state overlap", GS_overlap_final)
    for (key, value) in parsed_args
        write(file, string(key), value)
    end
    
    group = create_group(file, "Optimal control parameters")
    for (param, val) in best_coupling_params
        write(group, param, val)
    end
end
println("Optimization results saved to: ", filename)
