using ArgParse, HDF5
using CoolingTNS

parsed_args = CoolingTNS.parse_commandline()
println("Parsed args:")
for (arg, val) in parsed_args
    println("  $arg  =>  $val")
end

# Unpack parsed arguments
problem, N, steps, g, te, cutoff, Dmax =
    parsed_args["problem"], parsed_args["N"], parsed_args["steps"], parsed_args["g"],
    parsed_args["te"], parsed_args["cutoff"], parsed_args["Dmax"]

method = "MPS"

ham_params, ham_name = CoolingTNS.extract_ham_params(problem, parsed_args)

sites, H_sys, H_sys_bath, ϕ₀, ψ_s, e₀ = CoolingTNS.setup_problem_mps(problem, N, ham_params, g)
println("The ground state energy density is e₀/N = $(e₀/N)")

E_list, GS_overlap_list, nb_list = CoolingTNS.run_cooling_mps(
    sites,
    H_sys,
    H_sys_bath,
    ψ_s,
    ϕ₀,
    steps,
    te=te,
    cutoff=cutoff,
    Dmax=Dmax
)

E_final = E_list[end]
Edensity_final = E_final / N
GS_overlap_final = GS_overlap_list[end]
println("After cooling: E_final/N=$Edensity_final, GS_overlap_final=$GS_overlap_final")

filename = "Cooling_Ham$(ham_name)Ns$(N)Nb$(N)_Paramsg$(g)te$(te)Steps$(steps)_Method$(method)Dmax$(Dmax)"

h5open("Results/$(filename).h5", "w") do file
    write(file, "E_list", E_list)
    write(file, "GS_overlap_list", GS_overlap_list)
    write(file, "nb_list", nb_list)
    write(file, "E_final", E_final)
    write(file, "Edensity_final", Edensity_final)
    write(file, "GS_overlap_final", GS_overlap_final)
    write(file, "ham_name", ham_name)
    for (k, v) in parsed_args
        write(file, string(k), v)
    end
end
println("Data saved to $(filename) with Hamiltonian information and argparse variables")

CoolingTNS.plot_energy_and_overlap(E_list, GS_overlap_list, steps, filename, e₀, N)

