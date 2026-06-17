if Sys.islinux()
    using MKL
end

using CoolingTNS

function plot_energy_and_overlap end
function plot_momentum_distribution end

const _COOLING_PLOTTING_LOADED = Ref(false)

function load_cooling_plotting_utilities!()
    if !_COOLING_PLOTTING_LOADED[]
        include(joinpath(@__DIR__, "scripts", "plotting", "plotting.jl"))
        _COOLING_PLOTTING_LOADED[] = true
    end
    return nothing
end

# Helper function to create simulation parameters from new argument structure
function create_sim_params_new(parsed_args)
    backend = CoolingTNS.get_backend(parsed_args["backend"])
    sim_method = CoolingTNS.get_sim_method(parsed_args["sim_method"])
    evolution_method = CoolingTNS.get_evolution_method(parsed_args["evolution_method"])

    return CoolingTNS.create_sim_params(backend; sim_method=sim_method, evolution_method=evolution_method,
                                       Dmax=parsed_args["Dmax"], cutoff=parsed_args["cutoff"],
                                       tau=parsed_args["tau"], pe=parsed_args["peInt"]*1e-3,
                                       n_trajectories=parsed_args["n_trajectories"])
end

function run_cooling(parsed_args)
    println(parsed_args)

    # Setup common parameters
    problem, ham_params, ham_name, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
    
    # Get backend and create simulation parameters
    backend = CoolingTNS.get_backend(parsed_args["backend"])
    sim_params = create_sim_params_new(parsed_args)
    
    # Setup problem using unified interface
    cooling_problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
    
    # Setup initial state using unified interface
    initial_state = CoolingTNS.setup_initial_state(
        cooling_problem, 
        sim_params,
        parsed_args["init_state"], 
        parsed_args["theta"]
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
    println("The ground state energy density is e₀/N = $(e₀/ham_params.N)")

    window_size = parsed_args["window_size"]
    E_final = CoolingTNS.mean_last_window(results[CoolingTNS.RESULT_ENERGY], window_size)
    Edensity_final = E_final / ham_params.N
    GS_overlap_final = CoolingTNS.mean_last_window(results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP], window_size)
    println("After cooling: E_final/N=$Edensity_final, GS_overlap_final=$GS_overlap_final")

    # Save results - add scalar values as new keys
    filename = CoolingTNS.create_filename(ham_params, coupling_params, sim_params, backend)
    # Create new dictionary with all results
    save_data = Dict(results...)  # Copy existing results
    save_data["E_final"] = [E_final]
    save_data["Edensity_final"] = [Edensity_final]
    save_data["GS_overlap_final"] = [GS_overlap_final]
    save_data["delta"] = cooling_problem.extra.coupling_params.delta
    CoolingTNS.save_results(filename, save_data, e₀, ham_name, parsed_args)
    
    # Plot results
    load_cooling_plotting_utilities!()
    plot_energy = getfield(@__MODULE__, :plot_energy_and_overlap)
    Base.invokelatest(
        plot_energy,
        results[CoolingTNS.RESULT_ENERGY],
        results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP],
        e₀,
        ham_params.N,
        filename;
        moving_average=true,
    )
    
    # Generate k-space plots for ED simulations with PBC/APBC (only for Ising model)
    if backend isa CoolingTNS.EDBackend && CoolingTNS.supports_ising_fourier_observables(ham_params)
        if haskey(results, CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION) &&
           haskey(results, CoolingTNS.RESULT_K_VALUES)
            println("\nGenerating k-space plots...")
            full_filename = joinpath("Results", "$(filename).h5")
            try
                plot_momentum = getfield(@__MODULE__, :plot_momentum_distribution)
                Base.invokelatest(plot_momentum, full_filename; save_fig=true)
                println("K-space line plot saved.")
            catch e
                println("Warning: Could not generate k-space plots: $e")
            end
        end
    end
end

# Parse command line arguments and run the cooling simulation
parsed_args = CoolingTNS.parse_commandline()
run_cooling(parsed_args)
