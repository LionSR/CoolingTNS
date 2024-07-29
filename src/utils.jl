function extract_ham_params(problem, parsed_args)
    if problem == "Ising"
        J, h = parsed_args["J"], parsed_args["h"]
        ham_params = (J, h)
        ham_name = "$(problem)J$(J)h$(h)"
    elseif problem == "niIsing"
        J, hx, hz = parsed_args["J"], parsed_args["hx"], parsed_args["hz"]
        ham_params = (J, hx, hz)
        ham_name = "$(problem)J$(J)hx$(hx)hz$(hz)"
    else
        error("Unknown problem type: $problem")
    end
    return ham_params, ham_name
end

function setup_system(problem, N, sites_sys, ham_params)
    H_sys = if problem == "Ising"
        ham_ising(N, sites_sys, ham_params)
    elseif problem == "niIsing"
        ham_niising(N, sites_sys, ham_params)
    else
        error("Unknown problem type: $problem")
    end

    Δ, e₀, ϕ₀ = compute_energy_gap_and_ground_state(H_sys, sites_sys)
    return H_sys, Δ, e₀, ϕ₀
end

function setup_common_parameters(parsed_args)
    N = parsed_args["N"]
    problem = parsed_args["problem"]
    ham_params, ham_name = extract_ham_params(problem, parsed_args)

    pe = parsed_args["pe"]
    if parsed_args["peInt"] > 0
        pe = parsed_args["peInt"] * 1e-3
        pe = round(pe, digits=4)
    end

    coupling_params = Dict(
        "g" => parsed_args["g"],
        "te" => parsed_args["te"],
        "steps" => parsed_args["steps"],
        "coupling" => parsed_args["coupling"]
    )

    return N, problem, ham_params, ham_name, pe, coupling_params
end

function create_sim_params(parsed_args, pe, method)
    sim_params = Dict(
        "cutoff" => parsed_args["cutoff"],
        "Dmax" => parsed_args["Dmax"],
        "pe" => pe,
        "peInt" => parsed_args["peInt"],
        "method" => method
    )
    
    if method == "MPO"
        sim_params["trotter_steps"] = Int(parsed_args["te"] / parsed_args["tau"])
        sim_params["tau"] = parsed_args["tau"]
    end
    
    return sim_params
end

function mean_last_window(list, window_size)
    return mean(list[max(1, end-window_size+1):end])
end

function save_results(filename, e₀, E_list, GS_overlap_list, E_final, Edensity_final, GS_overlap_final, ham_name, parsed_args, nb_list=nothing)
    h5open("Results/$(filename).h5", "w") do file
        write(file, "e₀", e₀)
        write(file, "E_list", E_list)
        write(file, "GS_overlap_list", GS_overlap_list)
        write(file, "E_final", E_final)
        write(file, "Edensity_final", Edensity_final)
        write(file, "GS_overlap_final", GS_overlap_final)
        write(file, "ham_name", ham_name)
        if nb_list !== nothing
            write(file, "nb_list", nb_list)
        end
        for (key, value) in parsed_args
            write(file, string(key), value)
        end
    end
    println("Data saved to $(filename) with Hamiltonian information and argparse variables")
end

function create_search_name_part(search_params)
    return "Search$(search_params["search_method"])trials$(search_params["num_trials"])"
end

function create_filename(ham_name, N, coupling_params, sim_params)
    ham_name_part = if isa(N, Array)
        "Ham$(ham_name)Nmin$(minimum(N))Nmax$(maximum(N))"
    else
        "Ham$(ham_name)Ns$(N)Nb$(N)"
    end
    coupling_name_part = "Coupling$(coupling_params["coupling"])g$(coupling_params["g"])te$(coupling_params["te"])steps$(coupling_params["steps"])"
    sim_name_part = "Sim$(sim_params["method"])"
    
    if sim_params["method"] == "MPO"
        sim_name_part *= "tau$(sim_params["tau"])"
    elseif sim_params["method"] == "MPS"
        sim_name_part *= "Dmax$(sim_params["Dmax"])"
    end
    
    sim_params["peInt"] > 0 && (sim_name_part *= "peInt$(sim_params["peInt"])")
    
    return "Cooling_$(ham_name_part)_$(coupling_name_part)_$(sim_name_part)"
end
