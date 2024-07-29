if Sys.islinux()
    using MKL
end
using Hyperopt, Random, Statistics, HDF5
using CoolingTNS

method = "MPS"
parsed_args = CoolingTNS.parse_commandline()

N, problem, ham_params, ham_name, pe, init_coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
sim_params = CoolingTNS.create_sim_params(parsed_args, pe, method)

# Additional parameters specific to optimization
num_trials = parsed_args["num_trials"]
search_method = parsed_args["search_method"]
k = parsed_args["k"]
steps = parsed_args["steps"]
Dmax = parsed_args["Dmax"]

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

    Efinal_density_avg = CoolingTNS.mean_last_window(E_list, window_size) / N
    return Efinal_density_avg
end

search_space = Dict("g" => range(0.1, 0.5, length=5), "te" => range(1.0, 3.0, length=5))

best_coupling_params, best_objective = if search_method == "Random"
    CoolingTNS.hyperopt_random_search(objective_function, search_space, num_trials, init_coupling_params)
elseif search_method == "Grid"
    CoolingTNS.iterative_grid_search(objective_function, search_space, 1, init_coupling_params)
elseif search_method == "Bayesian"
    CoolingTNS.hyperopt_bayesian_optimization(objective_function, search_space, num_trials, init_coupling_params)
else
    error("Invalid search method: $search_method")
end

println("Optimization Result:")
for (param, val) in best_coupling_params
    println("$param: $val")
end

filename = CoolingTNS.create_filename(ham_name, N, best_coupling_params, sim_params)
search_params = Dict("search_method" => search_method, "num_trials" => num_trials)
filename = "Optimize$(filename)_$(CoolingTNS.create_search_name_part(search_params))"

best_coupling_params["steps"] = steps * 4

H_sys_bath = ham_sys_bath_fn(N, sites, ham_params, best_coupling_params)
ψ_s = CoolingTNS.setup_init_state_mps(sites)
E_list, GS_overlap_list, nb_list = CoolingTNS.run_cooling_mps(sites, H_sys, ϕ₀, H_sys_bath, ψ_s, best_coupling_params, sim_params)
E_final = CoolingTNS.mean_last_window(E_list, k)
Edensity_final = E_final / N
GS_overlap_final = CoolingTNS.mean_last_window(GS_overlap_list, k)
println("Final energy density: ", Edensity_final)
println("Final ground state overlap: ", GS_overlap_final)

CoolingTNS.save_results(filename, e₀, E_list, GS_overlap_list, E_final, Edensity_final, GS_overlap_final, ham_name, parsed_args, nb_list)
CoolingTNS.plot_energy_and_overlap(E_list, GS_overlap_list, e₀, N, filename; moving_average=true)

