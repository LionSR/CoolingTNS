using ArgParse, HDF5
using CoolingTNS

method = "MPS"
parsed_args = CoolingTNS.parse_commandline()
println("Parsed args:")
for (arg, val) in parsed_args
    println("  $arg  =>  $val")
end

# Unpack parsed arguments
N = parsed_args["N"]
problem = parsed_args["problem"]

ham_params, ham_name = CoolingTNS.extract_ham_params(problem, parsed_args)

pe = parsed_args["pe"]
if parsed_args["peInt"] > 0
    pe = parsed_args["peInt"] * 1e-3
    pe = round(pe, digits=4)
end

sim_params = Dict(
    "cutoff" => parsed_args["cutoff"],
    "Dmax" => parsed_args["Dmax"],
    "pe" => pe
)

coupling_params = Dict(
    "g" => parsed_args["g"],
    "te" => parsed_args["te"],
    "steps" => parsed_args["steps"],
    "coupling" => parsed_args["coupling"]
)

sites, H_sys, ϕ₀, e₀, H_sys_bath = CoolingTNS.setup_problem_mps(problem, N, ham_params, coupling_params, sim_params)
println("The ground state energy density is e₀/N = $(e₀/N)")

ψ_s = CoolingTNS.setup_init_state_mps(sites)

E_list, GS_overlap_list, nb_list = CoolingTNS.run_cooling_mps(
    sites,
    H_sys,
    ϕ₀,
    H_sys_bath,
    ψ_s,
    coupling_params,
    sim_params
)

E_final = E_list[end]
Edensity_final = E_final / N
GS_overlap_final = GS_overlap_list[end]
println("After cooling: E_final/N=$Edensity_final, GS_overlap_final=$GS_overlap_final")

ham_name_part = "Ham$(ham_name)Ns$(N)Nb$(N)"
coupling_name_part = "Coupling$(coupling_params["coupling"])g$(parsed_args["g"])te$(parsed_args["te"])steps$(parsed_args["steps"])"
sim_name_part = "Sim$(method)Dmax$(parsed_args["Dmax"])"
filename = "Cooling_$(ham_name_part)_$(coupling_name_part)_$(sim_name_part)"

h5open("Results/$(filename).h5", "w") do file
    write(file, "E_list", E_list)
    write(file, "GS_overlap_list", GS_overlap_list)
    write(file, "nb_list", nb_list)
    write(file, "E_final", E_final)
    write(file, "Edensity_final", Edensity_final)
    write(file, "GS_overlap_final", GS_overlap_final)
    write(file, "ham_name", ham_name)
    for (key, value) in parsed_args
        write(file, string(key), value)
    end
end
println("Data saved to $(filename) with Hamiltonian information and argparse variables")

CoolingTNS.plot_energy_and_overlap(E_list, GS_overlap_list, e₀, N, filename; moving_average=true)
