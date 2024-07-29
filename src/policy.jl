using Hyperopt
using Random

function evaluate_params(objective_function, params)
    println("\nEvaluating params: g=", params["g"], ", te=", params["te"])
    return objective_function(params)
end

function update_best(objective, params, best_objective, best_params)
    if objective < best_objective
        return objective, copy(params)
    end
    return best_objective, best_params
end

function random_search(objective_function, search_space, num_trials, initial_params)
    best_params = copy(initial_params)
    best_objective = Inf

    for _ in 1:num_trials
        params = merge(initial_params, Dict("g" => rand(search_space["g"]), "te" => rand(search_space["te"])))
        objective = evaluate_params(objective_function, params)
        best_objective, best_params = update_best(objective, params, best_objective, best_params)
    end

    return best_params, best_objective
end

function iterative_grid_search(objective_function, search_space, num_iterations, initial_params)
    best_params = copy(initial_params)
    best_objective = Inf

    for _ in 1:num_iterations
        for g in search_space["g"], te in search_space["te"]
            params = merge(best_params, Dict("g" => g, "te" => te))
            objective = evaluate_params(objective_function, params)
            best_objective, best_params = update_best(objective, params, best_objective, best_params)
        end

        # Refine search space around best parameters for next iteration
        search_space["g"] = range(max(best_params["g"] - 0.1, 0.1), min(best_params["g"] + 0.1, 0.5), length=10)
        search_space["te"] = range(max(best_params["te"] - 0.5, 1.0), min(best_params["te"] + 0.5, 3.0), length=10)
    end

    return best_params, best_objective
end

function hyperopt_random_search(objective_function, search_space, num_trials, initial_params)
    ho = @hyperopt for _ in 1:num_trials,
        sampler = RandomSampler(),
        g = search_space["g"],
        te = search_space["te"]

        params = merge(initial_params, Dict("g" => g, "te" => te))
        evaluate_params(objective_function, params)
    end

    best_params = merge(initial_params, Dict("g" => ho.minimizer[1], "te" => ho.minimizer[2]))
    return best_params, ho.minimum
end

function hyperopt_bayesian_optimization(objective_function, search_space, num_trials, initial_params)
    bohb = @hyperopt for _ in 1:num_trials,
        sampler = Hyperband(R=num_trials, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous()])),
        g = search_space["g"],
        te = search_space["te"]

        params = merge(initial_params, Dict("g" => g, "te" => te))
        objective = evaluate_params(objective_function, params)
        objective, (g, te)
    end

    best_params = merge(initial_params, Dict("g" => bohb.minimizer[1], "te" => bohb.minimizer[2]))
    return best_params, bohb.minimum
end
