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
    best_params = copy(initial_params)
    best_objective = Inf

    for _ in 1:num_trials
        g = rand(search_space["g"])
        te = rand(search_space["te"])
        params = merge(initial_params, Dict("g" => g, "te" => te))
        objective = evaluate_params(objective_function, params)
        best_objective, best_params = update_best(objective, params, best_objective, best_params)
    end

    return best_params, best_objective
end

function hyperopt_bayesian_optimization(objective_function, search_space, num_trials, initial_params)
    best_params = copy(initial_params)
    best_objective = Inf

    model = GaussianProcesses.GP(2, mean = MeanConst(0.0), kernel = SEArd([0.1, 0.1], 5.0))
    
    for i in 1:num_trials
        if i <= 5
            g = rand(search_space["g"])
            te = rand(search_space["te"])
        else
            acquisition = EI()
            g, te = optimize_acquisition(model, acquisition, search_space)
        end
        
        params = merge(initial_params, Dict("g" => g, "te" => te))
        objective = evaluate_params(objective_function, params)
        best_objective, best_params = update_best(objective, params, best_objective, best_params)
        
        update!(model, [g, te], -objective)
    end

    return best_params, best_objective
end
