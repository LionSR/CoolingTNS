if Sys.islinux()
    using MKL
end
using CoolingTNS

parsed_args = CoolingTNS.parse_commandline()
println(parsed_args)

N, problem, ham_params, ham_name, pe, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
sim_params = CoolingTNS.create_sim_params(parsed_args, pe)

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

filename = CoolingTNS.create_filename(ham_name, N, coupling_params, sim_params)
CoolingTNS.save_results(filename, e₀, E_list, GS_overlap_list, E_final, Edensity_final, GS_overlap_final, ham_name, parsed_args)
CoolingTNS.plot_energy_and_overlap(E_list, GS_overlap_list, e₀, N, filename)
