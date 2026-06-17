if Sys.islinux()
    using MKL
end
using Random, Statistics, HDF5
using CoolingTNS

function plot_energy_and_overlap end

const _OPT_COOLING_PLOTTING_LOADED = Ref(false)

function load_optimization_plotting_utilities!()
    if !_OPT_COOLING_PLOTTING_LOADED[]
        include(joinpath(@__DIR__, "scripts", "plotting", "plotting.jl"))
        _OPT_COOLING_PLOTTING_LOADED[] = true
    end
    return nothing
end

# Helper function to convert old optimization arguments to new dispatch format
function setup_optimization_params(parsed_args)
    # Set backend - for optimization we'll default to TN (tensor networks)
    parsed_args["backend"] = get(parsed_args, "backend", "TN")
    
    # Set simulation method based on old method if it exists
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

function run_optimization(parsed_args)
    # Convert old arguments to new dispatch format
    parsed_args = setup_optimization_params(parsed_args)
    println(parsed_args)

    # Setup common parameters using new dispatch architecture
    problem, ham_params, ham_name, init_coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
    
    # Get backend and create simulation parameters
    backend = CoolingTNS.get_backend(parsed_args["backend"])
    
    # Convert method strings to types
    sim_method = if parsed_args["sim_method"] == "density_matrix"
        CoolingTNS.DensityMatrix()
    else
        CoolingTNS.MonteCarloWavefunction()
    end
    
    evolution_method = if parsed_args["evolution_method"] == "trotter"
        CoolingTNS.TrotterEvolution()
    else
        CoolingTNS.ContinuousEvolution()
    end
    
    sim_params = CoolingTNS.create_sim_params(backend; 
        sim_method=sim_method, 
        evolution_method=evolution_method,
        Dmax=parsed_args["Dmax"], 
        cutoff=parsed_args["cutoff"], 
        tau=parsed_args["tau"], 
        pe=parsed_args["peInt"]*1e-3,
        n_trajectories=parsed_args["n_trajectories"])
    
    # Additional parameters specific to optimization
    num_trials = parsed_args["num_trials"]
    search_method = parsed_args["search_method"]
    window_size = parsed_args["window_size"]
    steps = parsed_args["steps"]
    
    # Setup problem using unified interface
    cooling_problem = CoolingTNS.setup_problem(backend, ham_params, init_coupling_params, sim_params)

    println("The ground state energy density is e₀/N = $(cooling_problem.e₀/ham_params.N)")

    function objective_function(coupling_dict)
        # Convert dict to proper CouplingParameters
        test_coupling_params = CoolingTNS.BasicCouplingParameters(
            init_coupling_params.coupling,
            coupling_dict["g"],
            init_coupling_params.steps,
            coupling_dict["te"],
            init_coupling_params.delta
        )
        
        # Setup initial state using unified interface
        initial_state = CoolingTNS.setup_initial_state(
            cooling_problem, 
            sim_params,
            "product",  # default initial state
            0.0
        )
        
        # Run cooling simulation using unified interface
        results = CoolingTNS.run_cooling(
            cooling_problem,
            initial_state,
            test_coupling_params,
            sim_params,
            ham_params
        )
        
        Efinal_density_avg = CoolingTNS.mean_last_window(results[CoolingTNS.RESULT_ENERGY], window_size) / ham_params.N
        return Efinal_density_avg
    end

    search_space = Dict("g" => range(0.1, 0.5, length=5), "te" => range(1.0, 3.0, length=5))

    # Simple optimization implementation (can be enhanced with Hyperopt later)
    best_coupling_params = Dict("g" => init_coupling_params.g, "te" => init_coupling_params.te)
    best_objective = objective_function(best_coupling_params)
    
    if search_method == "Random"
        for i in 1:num_trials
            test_params = Dict(
                "g" => rand(search_space["g"]),
                "te" => rand(search_space["te"])
            )
            obj = objective_function(test_params)
            if obj < best_objective
                best_objective = obj
                best_coupling_params = test_params
            end
        end
    else
        @warn "Only Random search implemented for now. Other methods need Hyperopt integration."
    end

    println("Optimization Result:")
    for (param, val) in best_coupling_params
        println("$param: $val")
    end

    # Create final coupling parameters with optimized values
    final_coupling_params = CoolingTNS.BasicCouplingParameters(
        init_coupling_params.coupling,
        best_coupling_params["g"],
        steps * 4,  # Run longer for final result
        best_coupling_params["te"],
        init_coupling_params.delta
    )
    
    filename = CoolingTNS.create_filename(ham_params, final_coupling_params, sim_params, backend)
    search_params = Dict("search_method" => search_method, "num_trials" => num_trials)
    filename = "Optimize$(filename)_$(CoolingTNS.create_search_name_part(search_params))"

    # Run final simulation with optimized parameters
    initial_state = CoolingTNS.setup_initial_state(
        cooling_problem, 
        sim_params,
        "product",
        0.0
    )
    
    results = CoolingTNS.run_cooling(
        cooling_problem,
        initial_state,
        final_coupling_params,
        sim_params,
        ham_params
    )

    E_final = CoolingTNS.mean_last_window(results[CoolingTNS.RESULT_ENERGY], window_size)
    Edensity_final = E_final / ham_params.N
    GS_overlap_final = CoolingTNS.mean_last_window(results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP], window_size)
    println("Final energy density: ", Edensity_final)
    println("Final ground state overlap: ", GS_overlap_final)

    # Save results with optimization metadata
    results["E_final"] = E_final
    results["Edensity_final"] = Edensity_final
    results["GS_overlap_final"] = GS_overlap_final
    results["best_g"] = best_coupling_params["g"]
    results["best_te"] = best_coupling_params["te"]
    
    CoolingTNS.save_results(filename, results, cooling_problem.e₀, ham_name, parsed_args; is_optimization=true)
    load_optimization_plotting_utilities!()
    plot_energy = getfield(@__MODULE__, :plot_energy_and_overlap)
    Base.invokelatest(
        plot_energy,
        results[CoolingTNS.RESULT_ENERGY],
        results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP],
        cooling_problem.e₀,
        ham_params.N,
        filename;
        moving_average=true,
        output_dir="ResultsOpt",
    )
end

# Parse command line arguments and run the optimization
parsed_args = CoolingTNS.parse_commandline()
run_optimization(parsed_args)
