using Hyperopt
using Random

function random_search(problem, objective_function, search_space, num_trials, initial_params)
    best_params = nothing
    best_objective = Inf

    for iter = 1:num_trials
        # Sample control parameters from the search space
        params = copy(initial_params)
        params["g"] = rand(search_space["g"])
        params["te"] = rand(search_space["te"])

        # Evaluate the objective function
        println("\nEvaluating params: g=", params["g"], ", te=", params["te"])
        objective = objective_function(problem, params)

        # Update best parameters if current objective is better
        if objective < best_objective
            best_objective = objective
            best_params = params
        end
    end

    return best_params, best_objective
end

function iterative_grid_search(problem, objective_function, search_space, num_iterations, initial_params)
    best_params = copy(initial_params)
    best_objective = Inf

    for iter = 1:num_iterations
        for g in search_space["g"]
            for te in search_space["te"]
                params = copy(best_params)
                params["g"] = g
                params["te"] = te

                println("\nEvaluating params: ", params["g"], ", ", params["te"])
                objective = objective_function(problem, params)

                if objective < best_objective
                    best_objective = objective
                    best_params = params
                end
            end
        end

        # Refine search space around best parameters for next iteration
        search_space["g"] = range(
            max(best_params["g"] - 0.1, 0.1),
            min(best_params["g"] + 0.1, 0.5),
            length=10
        )
        search_space["te"] = range(
            max(best_params["te"] - 0.5, 1.0),
            min(best_params["te"] + 0.5, 3.0),
            length=10
        )
    end

    return best_params, best_objective
end

function hyperopt_random_search(problem, objective_function, search_space, num_trials, initial_params)
    ho = @hyperopt for iter = num_trials, sampler = RandomSampler(), g = search_space["g"], te = search_space["te"]
        # Create a new dictionary with the updated parameters
        updated_params = merge(initial_params, Dict("g" => g, "te" => te))

        # Evaluate the objective function with the updated parameters
        println("\nEvaluating params: g=", g, ", te=", te)
        @show objective_function(problem, updated_params)
    end

    best_params_tuple, min_f = ho.minimizer, ho.minimum
    best_params = merge(initial_params, Dict("g" => best_params_tuple[1], "te" => best_params_tuple[2]))

    return best_params, min_f
end

function hyperopt_bayesian_optimization(problem, objective_function, search_space, num_trials, initial_params)
    # Use BOHB with the appropriate dimensions specified for continuous variables
    bohb = @hyperopt for iter = num_trials, sampler = Hyperband(R=num_trials, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous()])), g = search_space["g"], te = search_space["te"]
        if state !== nothing
            g, te = state
        end

        # Create a new dictionary with the updated parameters
        updated_params = merge(initial_params, Dict("g" => g, "te" => te))

        # Evaluate the objective function with the updated parameters
        println("\nEvaluating params: g=", g, ", te=", te)
        # @show objective_function(problem, updated_params)

        objective_function(problem, updated_params), (g, te)
    end
    best_params_tuple, min_f = bohb.minimizer, bohb.minimum
    best_params = merge(initial_params, Dict("g" => best_params_tuple[1], "te" => best_params_tuple[2]))
    return best_params, min_f
end