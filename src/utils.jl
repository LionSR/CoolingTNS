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

    coupling_params = Dict(
        "g" => parsed_args["g"],
        "te" => parsed_args["te"],
        "steps" => parsed_args["steps"],
        "coupling" => parsed_args["coupling"]
    )

    return N, problem, ham_params, ham_name, coupling_params
end

function create_sim_params(parsed_args)
    pe = parsed_args["peInt"] > 0 ? round(parsed_args["peInt"] * 1e-3, digits=4) : 0
    
    return Dict(
        "cutoff" => parsed_args["cutoff"],
        "Dmax" => parsed_args["Dmax"],
        "pe" => pe,
        "peInt" => parsed_args["peInt"],
        "method" => parsed_args["method"],
        "trotter_steps" => Int(parsed_args["te"] / parsed_args["tau"]),
        "tau" => parsed_args["tau"]
    )
end

function mean_last_window(list, window_size)
    return mean(list[max(1, end-window_size+1):end])
end

function save_results(filename, result, e₀, ham_name, parsed_args; is_optimization=false)
    directory = is_optimization ? "ResultsOpt" : "Results"
    h5open(joinpath(directory, "$(filename).h5"), "w") do file
        write(file, "e₀", e₀)
        for (key, value) in result
            write(file, string(key), value)
        end
        write(file, "ham_name", ham_name)
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
    ham_name_part = isa(N, Array) ? "Ham$(ham_name)Nmin$(minimum(N))Nmax$(maximum(N))" : "Ham$(ham_name)Ns$(N)Nb$(N)"
    coupling_name_part = "Coupling$(coupling_params["coupling"])g$(coupling_params["g"])te$(coupling_params["te"])steps$(coupling_params["steps"])"
    sim_name_part = "Sim$(sim_params["method"])Dmax$(sim_params["Dmax"])tau$(sim_params["tau"])"
    sim_name_part *= sim_params["peInt"] > 0 ? "peInt$(sim_params["peInt"])" : ""
    
    return join(["Cooling", ham_name_part, coupling_name_part, sim_name_part], "_")
end
