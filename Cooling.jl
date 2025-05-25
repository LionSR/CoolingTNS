if Sys.islinux()
    using MKL
end

using CoolingTNS

function run_cooling(parsed_args)
    println(parsed_args)

    # Setup common parameters
    N, problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
    sim_params = CoolingTNS.create_sim_params(parsed_args)

    # Get backend from method string
    backend = CoolingTNS.get_backend(parsed_args["method"])
    
    # Get simulation method (relevant for ED)
    ed_method = get(parsed_args, "ed_method", "")
    sim_method = CoolingTNS.get_simulation_method(backend, ed_method)
    
    # Setup problem using unified interface
    cooling_problem = CoolingTNS.setup_problem(backend, N, problem, ham_params, coupling_params, sim_params)
    
    # Setup initial state using unified interface
    initial_state = CoolingTNS.setup_initial_state(
        cooling_problem, 
        parsed_args["init_state"], 
        parsed_args["theta"];
        method=sim_method
    )
    
    # Run cooling simulation using unified interface
    results = CoolingTNS.run_cooling(
        cooling_problem,
        initial_state,
        coupling_params,
        sim_params,
        ham_params  # Only used by TrotterMPS
    )
    
    # Post-processing (same for all methods)
    e₀ = cooling_problem.e₀
    println("The ground state energy density is e₀/N = $(e₀/N)")

    window_size = parsed_args["window_size"]
    E_final = CoolingTNS.mean_last_window(results["E_list"], window_size)
    Edensity_final = E_final / N
    GS_overlap_final = CoolingTNS.mean_last_window(results["GS_overlap_list"], window_size)
    println("After cooling: E_final/N=$Edensity_final, GS_overlap_final=$GS_overlap_final")

    # Save results
    filename = CoolingTNS.create_filename(ham_name, N, coupling_params, sim_params)
    results["E_final"] = E_final
    results["Edensity_final"] = Edensity_final
    results["GS_overlap_final"] = GS_overlap_final
    CoolingTNS.save_results(filename, results, e₀, ham_name, parsed_args)
    
    # Plot results
    CoolingTNS.plot_energy_and_overlap(results["E_list"], results["GS_overlap_list"], e₀, N, filename; moving_average=true)
end

# Parse command line arguments and run the cooling simulation
parsed_args = CoolingTNS.parse_commandline()
run_cooling(parsed_args)
