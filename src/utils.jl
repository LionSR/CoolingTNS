# parameter_types.jl already included by CoolingTNS.jl

# Legacy functions removed - use setup_common_parameters and dispatch instead

function setup_common_parameters(parsed_args)
    N = parsed_args["N"]
    problem = parsed_args["problem"]
    
    # Create proper HamiltonianParameters struct
    if problem == "Ising"
        ham_params = IsingParameters(N, parsed_args["J"], parsed_args["h"])
        ham_name = "$(problem)J$(parsed_args["J"])h$(parsed_args["h"])"
    elseif problem == "niIsing"
        ham_params = NiIsingParameters(N, parsed_args["J"], parsed_args["hx"], parsed_args["hz"])
        ham_name = "$(problem)J$(parsed_args["J"])hx$(parsed_args["hx"])hz$(parsed_args["hz"])"
    elseif problem == "Rydberg"
        # Add default Rydberg parameters if needed
        Ω = get(parsed_args, "Omega", 1.0)
        Δ = get(parsed_args, "Delta", 0.0)
        V = get(parsed_args, "V", 1.0)
        ham_params = RydbergParameters(N, Ω, Δ, V)
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

    return problem, ham_params, ham_name, coupling_params
end

"""
    create_sim_params(backend::CoolingBackend; sim_method=nothing, evolution_method=nothing, kwargs...)

Create UnifiedSimulationParameters with intelligent defaults based on backend.
"""
function create_sim_params(backend::CoolingBackend; 
                          sim_method=nothing, 
                          evolution_method=nothing, 
                          kwargs...)
    # Use defaults if not specified
    sim_method = isnothing(sim_method) ? default_simulation_method(backend) : sim_method
    evolution_method = isnothing(evolution_method) ? default_evolution_method(backend) : evolution_method
    
    # Use direct constructor with dispatch
    return UnifiedSimulationParameters(sim_method, evolution_method; kwargs...)
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

function create_filename(ham_name, ham_params::HamiltonianParameters, coupling_params::CouplingParameters, sim_params::UnifiedSimulationParameters, backend::CoolingBackend)
    N = ham_params.N
    
    # More concise ham name (remove "Ham" prefix and shorten)
    ham_part = isa(N, Array) ? "$(ham_name)_N$(minimum(N))-$(maximum(N))" : "$(ham_name)_N$(N)"
    
    # Concise coupling part (remove "Coupling" prefix)
    coupling_part = "$(coupling_params.coupling)_g$(coupling_params.g)_t$(coupling_params.te)_s$(coupling_params.steps)"
    
    # Concise backend/sim part
    backend_str = backend isa TNBackend ? "TN" : "ED"
    sim_part = backend_str
    
    # Add key method parameters only
    if backend isa TNBackend && sim_params.Dmax != 100  # Only add if not default
        sim_part *= "_D$(sim_params.Dmax)"
    end
    
    if sim_params.evolution_method isa TrotterEvolution
        sim_part *= "_tau$(sim_params.tau)"
    end
    
    if sim_params.pe > 0
        pe_int = Int(round(sim_params.pe * 1000))
        sim_part *= "_pe$(pe_int)"
    end
    
    # Add delta if specified
    if coupling_params.delta !== nothing
        delta_str = @sprintf("%.3f", coupling_params.delta)
        coupling_part *= "_d$(delta_str)"
    end
    
    return join([ham_part, coupling_part, sim_part], "_")
end

