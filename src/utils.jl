# Import parameter types
include("parameter_types.jl")

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

# Legacy setup_system function removed - use setup_system_dispatch.jl instead

function setup_common_parameters(parsed_args)
    N = parsed_args["N"]
    problem = parsed_args["problem"]
    
    # Create proper HamiltonianParameters struct
    if problem == "Ising"
        ham_params = IsingParameters(parsed_args["J"], parsed_args["h"])
        ham_name = "$(problem)J$(parsed_args["J"])h$(parsed_args["h"])"
    elseif problem == "niIsing"
        ham_params = NiIsingParameters(parsed_args["J"], parsed_args["hx"], parsed_args["hz"])
        ham_name = "$(problem)J$(parsed_args["J"])hx$(parsed_args["hx"])hz$(parsed_args["hz"])"
    elseif problem == "Rydberg"
        # Add default Rydberg parameters if needed
        Ω = get(parsed_args, "Omega", 1.0)
        Δ = get(parsed_args, "Delta", 0.0)
        V = get(parsed_args, "V", 1.0)
        ham_params = RydbergParameters(Ω, Δ, V)
        ham_name = "$(problem)Omega$(Ω)Delta$(Δ)V$(V)"
    else
        error("Unknown problem type: $problem")
    end

    # Create proper CouplingParameters struct
    coupling_params = BasicCouplingParameters(
        parsed_args["coupling"],
        parsed_args["g"],
        parsed_args["steps"],
        parsed_args["te"],
        get(parsed_args, "delta", nothing)
    )

    return N, problem, ham_params, ham_name, coupling_params
end

# Legacy function - deprecated, use cooling_interface.jl create_sim_params instead
function create_sim_params_legacy(parsed_args)
    pe = parsed_args["peInt"] > 0 ? round(parsed_args["peInt"] * 1e-3, digits=4) : 0
    
    backend_str = get(parsed_args, "method", get(parsed_args, "backend", "TN"))
    
    return Dict(
        "cutoff" => parsed_args["cutoff"],
        "Dmax" => parsed_args["Dmax"],
        "pe" => pe,
        "peInt" => parsed_args["peInt"],
        "method" => backend_str,
        "backend" => backend_str,
        "trotter_steps" => Int(parsed_args["te"] / parsed_args["tau"]),
        "tau" => parsed_args["tau"],
        "n_trajectories" => get(parsed_args, "n_trajectories", 100)
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

function create_filename(ham_name, N, coupling_params::CouplingParameters, sim_params::UnifiedSimulationParameters)
    ham_name_part = isa(N, Array) ? "Ham$(ham_name)Nmin$(minimum(N))Nmax$(maximum(N))" : "Ham$(ham_name)Ns$(N)Nb$(N)"
    coupling_name_part = "Coupling$(coupling_params.coupling)g$(coupling_params.g)te$(coupling_params.te)steps$(coupling_params.steps)"
    
    # Determine backend string based on simulation method and evolution method
    if sim_params.sim_method isa MonteCarloWavefunction && sim_params.evolution_method isa ContinuousEvolution
        method_str = "MPS"
    elseif sim_params.sim_method isa DensityMatrix && sim_params.evolution_method isa TrotterEvolution
        method_str = "MPO"
    elseif sim_params.sim_method isa MonteCarloWavefunction && sim_params.evolution_method isa TrotterEvolution
        method_str = "TrotterMPS"
    else
        method_str = "TN"
    end
    
    sim_name_part = "Sim$(method_str)Dmax$(sim_params.Dmax)"
    if sim_params.evolution_method isa TrotterEvolution
        sim_name_part *= "tau$(sim_params.tau)"
    end
    if sim_params.pe > 0
        pe_int = Int(round(sim_params.pe * 1000))
        sim_name_part *= "peInt$(pe_int)"
    end
    
    return join(["Cooling", ham_name_part, coupling_name_part, sim_name_part], "_")
end

# Legacy wrapper for old Dict-based calls
function create_filename(ham_name, N, coupling_params::Dict, sim_params::Dict)
    # Convert to new format
    cp = BasicCouplingParameters(
        coupling_params["coupling"],
        coupling_params["g"],
        coupling_params["steps"],
        coupling_params["te"]
    )
    
    # Create a minimal UnifiedSimulationParameters
    sp = UnifiedSimulationParameters(
        MonteCarloWavefunction(),
        ContinuousEvolution();
        Dmax=get(sim_params, "Dmax", 100),
        tau=get(sim_params, "tau", 0.1),
        pe=get(sim_params, "pe", 0.0)
    )
    
    return create_filename(ham_name, N, cp, sp)
end
