using ArgParse, HDF5, Statistics
using CoolingTNS

function setup_common_parameters(parsed_args)
    N = parsed_args["N"]
    problem = parsed_args["problem"]
    ham_params, ham_name = CoolingTNS.extract_ham_params(problem, parsed_args)

    pe = parsed_args["pe"]
    if parsed_args["peInt"] > 0
        pe = parsed_args["peInt"] * 1e-3
        pe = round(pe, digits=4)
    end

    coupling_params = Dict(
        "g" => parsed_args["g"],
        "te" => parsed_args["te"],
        "steps" => parsed_args["steps"],
        "coupling" => parsed_args["coupling"]
    )

    return N, problem, ham_params, ham_name, pe, coupling_params
end

function create_sim_params(parsed_args, pe, method)
    sim_params = Dict(
        "cutoff" => parsed_args["cutoff"],
        "Dmax" => parsed_args["Dmax"],
        "pe" => pe,
        "method" => method
    )
    
    if method == "MPO"
        sim_params["trotter_steps"] = Int(parsed_args["te"] / parsed_args["tau"])
        sim_params["tau"] = parsed_args["tau"]
    end
    
    return sim_params
end

function save_results(filename, e₀, E_list, GS_overlap_list, E_final, Edensity_final, GS_overlap_final, ham_name, parsed_args, nb_list=nothing)
    h5open("Results/$(filename).h5", "w") do file
        write(file, "e₀", e₀)
        write(file, "E_list", E_list)
        write(file, "GS_overlap_list", GS_overlap_list)
        write(file, "E_final", E_final)
        write(file, "Edensity_final", Edensity_final)
        write(file, "GS_overlap_final", GS_overlap_final)
        write(file, "ham_name", ham_name)
        if nb_list !== nothing
            write(file, "nb_list", nb_list)
        end
        for (key, value) in parsed_args
            write(file, string(key), value)
        end
    end
    println("Data saved to $(filename) with Hamiltonian information and argparse variables")
end

function create_filename(ham_name, N, coupling_params, sim_params)
    ham_name_part = "Ham$(ham_name)Ns$(N)Nb$(N)"
    coupling_name_part = "Coupling$(coupling_params["coupling"])g$(coupling_params["g"])te$(coupling_params["te"])steps$(coupling_params["steps"])"
    sim_name_part = "Sim$(sim_params["method"])"
    
    if sim_params["method"] == "MPO"
        sim_name_part *= "tau$(sim_params["tau"])"
    else
        sim_name_part *= "Dmax$(sim_params["Dmax"])"
    end
    
    sim_params["pe"] > 0 && (sim_name_part *= "pe$(sim_params["pe"])")
    
    return "Cooling_$(ham_name_part)_$(coupling_name_part)_$(sim_name_part)"
end
