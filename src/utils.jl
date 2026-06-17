# parameter_types.jl already included by CoolingTNS.jl

# Legacy functions removed - use setup_common_parameters and dispatch instead

function setup_common_parameters(parsed_args)
    N = parsed_args["N"]
    problem = parsed_args["problem"]
    
    # Create proper HamiltonianParameters struct
    # Get boundary condition
    bc_str = get(parsed_args, "bc", "open")
    bc = Symbol(bc_str)
    
    if problem == "Ising"
        ham_params = IsingParameters(N, parsed_args["J"], parsed_args["h"], bc)
    elseif problem == "niIsing"
        ham_params = NiIsingParameters(N, parsed_args["J"], parsed_args["hx"], parsed_args["hz"], bc)
    elseif problem == "Rydberg"
        # Add default Rydberg parameters if needed
        Ω = get(parsed_args, "Omega", 1.0)
        Δ = get(parsed_args, "Delta", 0.0)
        V = get(parsed_args, "V", 1.0)
        ham_params = RydbergParameters(N, Ω, Δ, V, bc)
    else
        error("Unknown problem type: $problem")
    end
    
    # Generate ham_name using the HamiltonianParameters method
    ham_name = hamiltonian_name(ham_params)

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

"""
    create_sim_params_from_args(parsed_args)

Construct `UnifiedSimulationParameters` from command-line arguments parsed by
[`parse_commandline`](@ref). This is the single translation from CLI strings and
integer noise units to the typed simulation-parameter object.
"""
function create_sim_params_from_args(parsed_args)
    backend = get_backend(parsed_args["backend"])
    return create_sim_params(
        backend;
        sim_method=get_sim_method(parsed_args["sim_method"]),
        evolution_method=get_evolution_method(parsed_args["evolution_method"]),
        Dmax=parsed_args["Dmax"],
        cutoff=parsed_args["cutoff"],
        tau=parsed_args["tau"],
        pe=parsed_args["peInt"] * 1e-3,
        n_trajectories=parsed_args["n_trajectories"],
    )
end

"""
    normalize_optimization_args!(parsed_args)

Normalize optimization command-line arguments in place.

The current parser uses the explicit fields `backend`, `sim_method`, and
`evolution_method`. Older optimization scripts may instead pass a `method`
field. In that legacy convention, `MPS` denotes TN Monte Carlo wavefunction
evolution and `MPO` denotes TN density-matrix Trotter evolution.
"""
function normalize_optimization_args!(parsed_args)
    parsed_args["backend"] = get(parsed_args, "backend", "TN")

    if haskey(parsed_args, "method")
        if parsed_args["method"] == "MPS"
            parsed_args["sim_method"] = "monte_carlo"
            parsed_args["evolution_method"] = "continuous"
        elseif parsed_args["method"] == "MPO"
            parsed_args["sim_method"] = "density_matrix"
            parsed_args["evolution_method"] = "trotter"
        end
    end

    return parsed_args
end


function mean_last_window(list, window_size)
    return mean(list[max(1, end-window_size+1):end])
end

"""
    relative_energy(E, E_GS) -> Float64

Relative energy figure of merit used in the paper:

    e = |(E - E_GS)/E_GS|.

Here `E_GS` is the (non-degenerate) ground state energy of the system Hamiltonian.
The inputs can be either total energies or energy densities, as long as `E` and
`E_GS` use the same normalization.

Returns `NaN` if `E_GS == 0`.
"""
function relative_energy(E, E_GS)
    iszero(E_GS) && return NaN
    return abs((E - E_GS) / E_GS)
end

"""
Top-level HDF5 group containing the canonical parsed command-line arguments.
"""
const HDF5_PARSED_ARGS_GROUP = "parsed_args"

function _write_hdf5_dataset_unless_present(parent, key::AbstractString, value)
    if key in keys(parent)
        return false
    end
    write(parent, key, value)
    return true
end

function _create_hdf5_group_or_error(file, group_name::AbstractString)
    if group_name in keys(file)
        error("Cannot write parsed command-line metadata group '$group_name'; a result dataset already uses that top-level name.")
    end
    return create_group(file, group_name)
end

function _assert_hdf5_result_keys_available(result)
    for key in keys(result)
        if string(key) == HDF5_PARSED_ARGS_GROUP
            error("Cannot write result dataset '$(HDF5_PARSED_ARGS_GROUP)'; that top-level HDF5 name is reserved for parsed command-line metadata.")
        end
    end
    return nothing
end

function save_results(filename, result, e₀, ham_name, parsed_args; is_optimization=false)
    _assert_hdf5_result_keys_available(result)

    directory = is_optimization ? "ResultsOpt" : "Results"
    mkpath(directory)
    h5open(joinpath(directory, "$(filename).h5"), "w") do file
        write(file, "e₀", e₀)
        for (key, value) in result
            write(file, string(key), value)
        end
        write(file, "ham_name", ham_name)

        # /parsed_args is the canonical configuration namespace. Top-level
        # mirrors are kept only for legacy readers and skipped on collisions.
        parsed_args_group = _create_hdf5_group_or_error(file, HDF5_PARSED_ARGS_GROUP)
        for (key, value) in parsed_args
            arg_key = string(key)
            write(parsed_args_group, arg_key, value)
            _write_hdf5_dataset_unless_present(file, arg_key, value)
        end
    end
    println("Data saved to $(filename) with Hamiltonian information and argparse variables")
end

function create_search_name_part(search_params)
    return "Search$(search_params["search_method"])trials$(search_params["num_trials"])"
end

function create_filename(
    ham_params::HamiltonianParameters,
    coupling_params::MultiFrequencyCouplingParameters,
    sim_params::UnifiedSimulationParameters,
    backend::CoolingBackend,
)
    # Ham group: HamIsingJ1.0h1.0 (no underscores within group)
    ham_name = hamiltonian_name(ham_params)
    ham_group = "Ham$(ham_name)"

    # Coupling group: include only summary info about the Δ-grid (to keep names short)
    R = length(coupling_params.delta_values)
    δmin = minimum(coupling_params.delta_values)
    δmax = maximum(coupling_params.delta_values)
    dmin_str = @sprintf("%.3f", δmin)
    dmax_str = @sprintf("%.3f", δmax)

    schedule_str = if coupling_params.schedule == :round_robin
        "rr"
    elseif coupling_params.schedule == :random
        "rand"
    else
        # Should not happen (validated in constructor), but keep filename stable.
        replace(string(coupling_params.schedule), "_" => "")
    end

    coupling_group =
        "Coupling$(coupling_params.coupling)g$(coupling_params.g)te$(coupling_params.te)steps$(coupling_params.steps)" *
        "multiR$(R)dmin$(dmin_str)dmax$(dmax_str)sched$(schedule_str)"

    if coupling_params.randomize_times
        coupling_group *= "randt"
    end

    # Sim group: SimTNDmax100 or SimED (no underscores within group)
    backend_str = backend isa TNBackend ? "TN" : "ED"
    sim_method_str = sim_params.sim_method isa DensityMatrix ? "DM" : "MC"
    sim_group = "Sim$(backend_str)$(sim_method_str)"

    if backend isa TNBackend && sim_params.Dmax != 100
        sim_group *= "Dmax$(sim_params.Dmax)"
    end

    if sim_params.evolution_method isa TrotterEvolution
        sim_group *= "tau$(sim_params.tau)"
    end

    if sim_params.pe > 0
        pe_int = Int(round(sim_params.pe * 1000))
        sim_group *= "pe$(pe_int)"
    end

    return "Cooling_$(ham_group)_$(coupling_group)_$(sim_group)"
end

function create_filename(ham_params::HamiltonianParameters, coupling_params::CouplingParameters, sim_params::UnifiedSimulationParameters, backend::CoolingBackend)
    # Ham group: HamIsingJ1.0h1.0 (no underscores within group)
    ham_name = hamiltonian_name(ham_params)
    ham_group = "Ham$(ham_name)"
    
    # Coupling group: CouplingXXg0.1te10.0steps100 (no underscores within group)
    coupling_group = "Coupling$(coupling_params.coupling)g$(coupling_params.g)te$(coupling_params.te)steps$(coupling_params.steps)"
    
    # Add delta if specified
    if coupling_params.delta !== nothing
        delta_str = @sprintf("%.3f", coupling_params.delta)
        coupling_group *= "delta$(delta_str)"
    end
    
    # Sim group: SimTNDmax100 or SimED (no underscores within group)
    backend_str = backend isa TNBackend ? "TN" : "ED"
    sim_method_str = sim_params.sim_method isa DensityMatrix ? "DM" : "MC"
    sim_group = "Sim$(backend_str)$(sim_method_str)"
    
    # Add key method parameters to sim group
    if backend isa TNBackend && sim_params.Dmax != 100  # Only add if not default
        sim_group *= "Dmax$(sim_params.Dmax)"
    end
    
    # Add other sim parameters if non-default
    if sim_params.evolution_method isa TrotterEvolution
        sim_group *= "tau$(sim_params.tau)"
    end
    
    if sim_params.pe > 0
        pe_int = Int(round(sim_params.pe * 1000))
        sim_group *= "pe$(pe_int)"
    end
    
    # Join the three groups with underscores
    return "Cooling_$(ham_group)_$(coupling_group)_$(sim_group)"
end

# Backward-compatible overload for plotting workflows that only have `ham_name`
# (as produced by `hamiltonian_name`) plus an explicit system size N.
function create_filename(
    ham_name::AbstractString,
    N::Union{Int, Vector{Int}},
    coupling_params::CouplingParameters,
    sim_params::UnifiedSimulationParameters,
    backend::CoolingBackend,
)
    template = parse_hamiltonian_name(ham_name)
    actual_N = N isa Vector ? N[1] : N
    ham_params = HamiltonianParameters(template.model, actual_N, template.params, template.bc)
    return create_filename(ham_params, coupling_params, sim_params, backend)
end

# Backward compatibility for legacy plotting scripts that pass Dicts instead of typed parameters.
function create_filename(
    ham_name::AbstractString,
    N::Union{Int, Vector{Int}},
    coupling_params::Dict,
    sim_params::Dict,
)
    coupling = BasicCouplingParameters(
        coupling_params["coupling"],
        coupling_params["g"],
        coupling_params["steps"],
        coupling_params["te"],
        get(coupling_params, "delta", nothing),
    )

    backend = haskey(sim_params, "method") && sim_params["method"] == "ED" ? EDBackend() : TNBackend()

    sim_method =
        haskey(sim_params, "sim_method") && sim_params["sim_method"] == "density_matrix" ?
        DensityMatrix() : MonteCarloWavefunction()
    evolution_method =
        haskey(sim_params, "evolution_method") && sim_params["evolution_method"] == "trotter" ?
        TrotterEvolution() : ContinuousEvolution()

    sim = UnifiedSimulationParameters(
        sim_method,
        evolution_method;
        Dmax=get(sim_params, "Dmax", 100),
        cutoff=get(sim_params, "cutoff", 1e-8),
        tau=get(sim_params, "tau", 0.1),
        pe=get(sim_params, "peInt", 0) / 1000.0,
        n_trajectories=get(sim_params, "n_trajectories", 1),
    )

    template = parse_hamiltonian_name(ham_name)
    actual_N = N isa Vector ? N[1] : N
    ham_params = HamiltonianParameters(template.model, actual_N, template.params, template.bc)

    return create_filename(ham_params, coupling, sim, backend)
end
