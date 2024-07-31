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
        sites, H_sys, ϕ₀, e₀, H_sys_bath = CoolingTNS.setup_problem_mps(N, problem, ham_params, coupling_params, sim_params)
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
        sites, H_sys, ϕ₀, e₀, gates = CoolingTNS.setup_problem_mpo(N, problem, ham_params, coupling_params, sim_params)
        initial_state = CoolingTNS.setup_init_state_mpo(sites)
        results = CoolingTNS.run_cooling_mpo(
            sites,
            H_sys,
            ϕ₀,
            gates,
            initial_state,
            coupling_params,
            sim_params
        )
    elseif method == "TrotterMPS"
        sites, H_sys, H_total, ϕ₀, e₀, gates = CoolingTNS.setup_problem_trotter_mps(N, problem, ham_params, coupling_params, sim_params)
        initial_state = CoolingTNS.setup_init_state_mps(sites)
        results = CoolingTNS.run_cooling_trotter_mps(
            sites,
            H_sys,
            H_total,
            ϕ₀,
            gates,
            initial_state,
            coupling_params,
            sim_params,
            ham_params
        )
    else
        error("Invalid method: $method. Choose 'MPS', 'MPO', or 'TrotterMPS'.")
    end

    println("The ground state energy density is e₀/N = $(e₀/N)")

    window_size = parsed_args["window_size"]
    E_final = CoolingTNS.mean_last_window(results["E_list"], window_size)
    Edensity_final = E_final / N
    GS_overlap_final = CoolingTNS.mean_last_window(results["GS_overlap_list"], window_size)
    println("After cooling: E_final/N=$Edensity_final, GS_overlap_final=$GS_overlap_final")

    filename = CoolingTNS.create_filename(ham_name, N, coupling_params, sim_params)
    results["E_final"] = E_final
    results["Edensity_final"] = Edensity_final
    results["GS_overlap_final"] = GS_overlap_final
    CoolingTNS.save_results(filename, results, e₀, ham_name, parsed_args)
    CoolingTNS.plot_energy_and_overlap(results["E_list"], results["GS_overlap_list"], e₀, N, filename; moving_average=true)
end

# Parse command line arguments and run the cooling simulation
parsed_args = CoolingTNS.parse_commandline()
run_cooling(parsed_args)
