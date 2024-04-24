using ArgParse, HDF5
using CoolingTNS

method = "MPO"
parsed_args = CoolingTNS.parse_commandline()

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
    "Dmax" => parsed_args["Dmax"],
    "cutoff" => parsed_args["cutoff"],
    "trotter_steps" => Int(parsed_args["te"] / parsed_args["tau"]),
    "tau" => parsed_args["tau"],
    "pe" => pe
)

coupling_params = Dict(
    "g" => parsed_args["g"],
    "te" => parsed_args["te"],
    "steps" => parsed_args["steps"],
    "coupling" => parsed_args["coupling"]
)

sites, H_sys, ϕ₀, e₀, gates = CoolingTNS.setup_problem_mpo(problem, N, ham_params, coupling_params, sim_params)
println("The ground state energy density is e₀/N = $(e₀/N)")

ρ_s = CoolingTNS.setup_init_state_mpo(sites)

E_list, GS_overlap_list = CoolingTNS.run_cooling_mpo(
    sites,
    H_sys,
    ϕ₀,
    gates,
    ρ_s,
    coupling_params,
    sim_params,
)

E_final = E_list[end]
Edensity_final = E_final / N
GS_overlap_final = GS_overlap_list[end]
println("After cooling: E_final/N=$Edensity_final, GS_overlap_final=$GS_overlap_final")

ham_name_part = "Ham$(ham_name)Ns$(N)Nb$(N)"
coupling_name_part = "Coupling$(coupling_params["coupling"])g$(parsed_args["g"])te$(parsed_args["te"])steps$(parsed_args["steps"])"
sim_name_part = "Sim$(method)tau$(parsed_args["tau"])"
pe > 0 && (sim_name_part *= "pe$pe")
filename = "Cooling_$(ham_name_part)_$(coupling_name_part)_$(sim_name_part)"

h5open("Results/$(filename).h5", "w") do file
    write(file, "e₀", e₀)
    write(file, "E_list", E_list)
    write(file, "GS_overlap_list", GS_overlap_list)
    write(file, "E_final", E_final)
    write(file, "Edensity_final", Edensity_final)
    write(file, "GS_overlap_final", GS_overlap_final)
    write(file, "ham_name", ham_name)
    for (key, value) in parsed_args
        write(file, string(key), value)
    end
end
println("Data saved to $(filename) with Hamiltonian information and argparse variables")

CoolingTNS.plot_energy_and_overlap(E_list, GS_overlap_list, e₀, N, filename)
