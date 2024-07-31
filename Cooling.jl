if Sys.islinux()
    using MKL
end
using CoolingTNS

function run_cooling(parsed_args)
    println(parsed_args)

    N, problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
    sim_params = CoolingTNS.create_sim_params(parsed_args)

    method = parsed_args["method"]

    if method == "MPS"
        sites, H_sys, ϕ₀, e₀, H_sys_bath = CoolingTNS.setup_problem_mps(problem, N, ham_params, coupling_params, sim_params)
        initial_state = CoolingTNS.setup_init_state_mps(sites)
        results = CoolingTNS.run_cooling_mps(
            sites,
            H_sys,
            ϕ₀,
            H_sys_bath,
            initial_state,
            coupling_params,
            sim_params
        )
    elseif method == "MPO"
        sites, H_sys, ϕ₀, e₀, gates = CoolingTNS.setup_problem_mpo(problem, N, ham_params, coupling_params, sim_params)
        initial_state = CoolingTNS.setup_init_state_mpo(sites)
        results = CoolingTNS.run_cooling_mpo(
            sites,
            H_sys,
            ϕ₀,
            gates,
            initial_state,
            coupling_params,
            sim_params,
        )
    else
        error("Invalid method: $method. Choose either 'MPS' or 'MPO'.")
    end

    println("The ground state energy density is e₀/N = $(e₀/N)")

    window_size = parsed_args["window_size"]
    E_list = results["E_list"]
    GS_overlap_list = results["GS_overlap_list"]
    E_final = CoolingTNS.mean_last_window(E_list, window_size)
    Edensity_final = E_final / N
    GS_overlap_final = CoolingTNS.mean_last_window(GS_overlap_list, window_size)
    println("After cooling: E_final/N=$Edensity_final, GS_overlap_final=$GS_overlap_final")

    filename = CoolingTNS.create_filename(ham_name, N, coupling_params, sim_params)
    nb_list = method == "MPS" ? results["nb_list"] : nothing
    CoolingTNS.save_results(filename, e₀, E_list, GS_overlap_list, E_final, Edensity_final, GS_overlap_final, ham_name, parsed_args, nb_list)
    CoolingTNS.plot_energy_and_overlap(E_list, GS_overlap_list, e₀, N, filename; moving_average=true)
end

# Parse command line arguments and run the cooling simulation
parsed_args = CoolingTNS.parse_commandline()
run_cooling(parsed_args)
