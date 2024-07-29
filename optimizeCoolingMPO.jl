if Sys.islinux()
    using MKL
end
using Hyperopt, Random, Statistics, HDF5
using CoolingTNS

method = "MPO"
parsed_args = CoolingTNS.parse_commandline()

N, problem, ham_params, ham_name, pe, init_coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
sim_params = CoolingTNS.create_sim_params(parsed_args, pe, method)

# Additional parameters specific to optimization
num_trials = parsed_args["num_trials"]
search_method = parsed_args["search_method"]
k = parsed_args["k"]
steps = parsed_args["steps"]

sites, H_sys, ϕ₀, e₀, gates = CoolingTNS.setup_problem_mpo(problem, N, ham_params, init_coupling_params, sim_params)
println("The ground state energy density is e₀/N = $(e₀/N)")

function objective_function(coupling_params)
    ρ_s = CoolingTNS.setup_init_state_mpo(sites)
    
    E_list, GS_overlap_list = CoolingTNS.run_cooling_mpo(
        sites,
        H_sys,
        ϕ₀,
        gates,
        ρ_s,
        coupling_params,
        sim_params,
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

ρ_s = CoolingTNS.setup_init_state_mpo(sites)
E_list, GS_overlap_list = CoolingTNS.run_cooling_mpo(
    sites,
    H_sys,
    ϕ₀,
    gates,
    ρ_s,
    best_coupling_params,
    sim_params,
)

E_final = mean(E_list[end-k+1:end])
Edensity_final = E_final / N
GS_overlap_final = mean(GS_overlap_list[end-k+1:end])
println("Final energy density: ", Edensity_final)
println("Final ground state overlap: ", GS_overlap_final)

CoolingTNS.save_results(filename, e₀, E_list, GS_overlap_list, E_final, Edensity_final, GS_overlap_final, ham_name, parsed_args)
CoolingTNS.plot_energy_and_overlap(E_list, GS_overlap_list, e₀, N, filename; moving_average=true)


