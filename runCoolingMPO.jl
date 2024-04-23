using ArgParse, HDF5
using CoolingTNS

method = "MPO"
parsed_args = CoolingTNS.parse_commandline()

# Unpack parsed arguments 
problem, N, steps, g, te, tau, cutoff =
    parsed_args["problem"], parsed_args["N"], parsed_args["steps"], parsed_args["g"],
    parsed_args["te"], parsed_args["tau"], parsed_args["cutoff"]

ham_params, ham_name = CoolingTNS.extract_ham_params(problem, parsed_args)

sites, H_sys, ϕ₀, e₀, gates = CoolingTNS.setup_problem_mpo(problem, N, ham_params, g, tau)
println("The ground state energy density is e₀/N = $(e₀/N)")

ρ_s = CoolingTNS.setup_init_state_mpo(sites)

trotter_steps = Int(te / tau)
E_list, GS_overlap_list = CoolingTNS.run_cooling_mpo(
    sites,
    H_sys,
    ρ_s,
    ϕ₀,
    steps,
    trotter_steps=trotter_steps,
    cutoff=cutoff,
    gates=gates,
)

E_final = E_list[end]
Edensity_final = E_final / N
GS_overlap_final = GS_overlap_list[end]
println("After cooling: E_final/N=$Edensity_final, GS_overlap_final=$GS_overlap_final")

filename = "Cooling_Ham$(ham_name)Ns$(N)Nb$(N)_Paramsg$(g)te$(te)steps$(steps)_Method$(method)tau$(tau)"

h5open("Results/$(filename).h5", "w") do file
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

CoolingTNS.plot_energy_and_overlap(E_list, GS_overlap_list, steps, e₀, N, filename)

