if Sys.islinux()
    println("Using MKL")
    using MKL
end
using Hyperopt, Random, Statistics, ArgParse, HDF5
using CoolingTNS

method = "MPS"
parsed_args = CoolingTNS.parse_commandline()
println("Parsed args:")
for (arg, val) in parsed_args
    println("  $arg  =>  $val")
end

# Unpack parsed arguments
num_trials = parsed_args["num_trials"]
search_method = parsed_args["search_method"]


N = parsed_args["N"]
problem = parsed_args["problem"]
k = parsed_args["k"]
steps = parsed_args["steps"]
Dmax = parsed_args["Dmax"]

ham_params, ham_name = CoolingTNS.extract_ham_params(problem, parsed_args)

pe = parsed_args["pe"]
peInt = parsed_args["peInt"]
if peInt > 0
    pe = peInt * 1e-3
    pe = round(pe, digits=4)
end

sim_params = Dict(
    "cutoff" => parsed_args["cutoff"],
    "Dmax" => parsed_args["Dmax"],
    "k" => parsed_args["k"],
    "pe" => pe,
)

init_coupling_params = Dict(
    "g" => parsed_args["g"],
    "te" => parsed_args["te"],
    "steps" => parsed_args["steps"],
    "coupling" => parsed_args["coupling"]
)

sites, H_sys, ϕ₀, e₀, H_sys_bath = CoolingTNS.setup_problem_mps(problem, N, ham_params, init_coupling_params, sim_params)
println("The ground state energy density is e₀/N = $(e₀/N)")
ham_sys_bath_fn = problem == "Ising" ? CoolingTNS.ham_ising_sys_bath : CoolingTNS.ham_niising_sys_bath

function objective_function(coupling_params)
    ψ_s = CoolingTNS.setup_init_state_mps(sites)
    H_sys_bath = ham_sys_bath_fn(N, sites, ham_params, coupling_params)

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
    # GS_overlap_avg = mean(GS_overlap_list[end-k+1:end])

    return Efinal_density_avg
end

search_space = Dict("g" => range(0.1, 0.5, length=5), "te" => range(1.0, 3.0, length=5))

if search_method == "Random"
    best_coupling_params, best_objective = CoolingTNS.hyperopt_random_search(objective_function, search_space, num_trials, init_coupling_params)
elseif search_method == "Grid"
    num_iterations = 1
    best_coupling_params, best_objective = CoolingTNS.iterative_grid_search(objective_function, search_space, num_iterations, init_coupling_params)
elseif search_method == "Bayesian"
    best_coupling_params, best_objective = CoolingTNS.hyperopt_bayesian_optimization(objective_function, search_space, num_trials, init_coupling_params)
else
    error("Invalid search method: $search_method")
end

println("Optimization Result:")
for (param, val) in best_coupling_params
    println("$param: $val")
end

filename = "OptimizedCooling_Ham$(ham_name)Ns$(N)Nb$(N)_ParamsSteps$(steps)_Sim$(method)Dmax$(Dmax)peInt$(peInt)_Search$(search_method)trials$(num_trials)"

best_coupling_params["steps"] = steps * 4

H_sys_bath = ham_sys_bath_fn(N, sites, ham_params, best_coupling_params)
ψ_s = CoolingTNS.setup_init_state_mps(sites)
Efinal_density_list, GS_overlap_final_list, _ = CoolingTNS.run_cooling_mps(sites, H_sys, ϕ₀, H_sys_bath, ψ_s, best_coupling_params, sim_params)
Efinal_density = mean(Efinal_density_list[end-k+1:end]) / N
GS_overlap_final = mean(GS_overlap_final_list[end-k+1:end])
println("Final energy density: ", Efinal_density)
println("Final ground state overlap: ", GS_overlap_final)
println("Optimal control parameters:")

CoolingTNS.plot_energy_and_overlap(Efinal_density_list, GS_overlap_final_list, e₀, N, filename; moving_average=true)

h5open("ResultsOpt/$(filename).h5", "w") do file
    write(file, "Final energy density", Efinal_density)
    write(file, "Final ground state overlap", GS_overlap_final)
    write(file, "e₀", e₀)
    # write(file, "N", N)
    for (key, value) in parsed_args
        write(file, string(key), value)
    end

    group = create_group(file, "Optimal control parameters")
    for (param, val) in best_coupling_params
        write(group, param, val)
    end

    group = create_group(file, "Simulation parameters")
    for (param, val) in sim_params
        write(group, param, val)
    end

    group = create_group(file, "Parsed arguments")
    for (param, val) in parsed_args
        write(group, param, val)
    end

    group = create_group(file, "Energy density list")
    write(group, "Efinal_density_list", Efinal_density_list)

    group = create_group(file, "Ground state overlap list")
    write(group, "GS_overlap_final_list", GS_overlap_final_list)
end
println("Optimization results saved to: ", filename)

