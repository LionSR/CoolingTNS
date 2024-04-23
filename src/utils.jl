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