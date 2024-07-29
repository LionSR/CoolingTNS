if Sys.islinux()
    using MKL
end
using CoolingTNS

method = "MPS"
parsed_args = CoolingTNS.parse_commandline()

N, problem, ham_params, ham_name, pe, coupling_params = CoolingTNS.setup_common_parameters(parsed_args)
sim_params = CoolingTNS.create_sim_params(parsed_args, pe, method)

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

k = parsed_args["k"]
E_final = mean(E_list[end-k+1:end])
Edensity_final = E_final / N
GS_overlap_final = mean(GS_overlap_list[end-k+1:end])
println("After cooling: E_final/N=$Edensity_final, GS_overlap_final=$GS_overlap_final")

filename = create_filename(ham_name, N, coupling_params, sim_params)
save_results(filename, e₀, E_list, GS_overlap_list, E_final, Edensity_final, GS_overlap_final, ham_name, parsed_args, nb_list)

CoolingTNS.plot_energy_and_overlap(E_list, GS_overlap_list, e₀, N, filename; moving_average=true)
