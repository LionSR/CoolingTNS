using ITensors
using ITensorTDVP
using Statistics

function setup_init_state_mps(sites)
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]
    ψ_s = randomMPS(sites_sys, linkdims=1)
    return ψ_s
end


function setup_problem_mps(problem, N, ham_params, g)
    sites = siteinds("S=1/2", 2N)
    sites_sys = sites[1:2:2N-1]

    H_sys, Δ, e₀, ϕ₀ = setup_system(problem, N, sites_sys, ham_params)

    coupling = "XX"
    ham_sys_bath_fn = problem == "Ising" ? ham_ising_sys_bath : ham_niising_sys_bath
    H_sys_bath = ham_sys_bath_fn(N, sites, ham_params, g, Δ, coupling)

    return sites, H_sys, H_sys_bath, ϕ₀, e₀
end


function run_cooling_mps(sites, H_sys, H_sys_bath, ψ_s, ϕ₀, steps; te, cutoff, Dmax)
    N = length(sites) ÷ 2

    E_list = zeros(steps + 1)
    GS_overlap_list = zeros(steps + 1)
    nb_list = zeros(steps + 1)

    E_list[1] = energy(ψ_s, H_sys)
    GS_overlap_list[1] = abs2(inner(ψ_s, ϕ₀))

    println("Cooling starts")
    println("Step 1: energy/N=$(E_list[1]/N), overlap=$(GS_overlap_list[1])")

    for step = 2:steps+1
        ψ_sb = appendzeros_MPS(ψ_s, sites)
        ψ_sb_evolved = evolve_state(H_sys_bath, ψ_sb, te; Dmax, cutoff)

        v_b, ψ_s = sample_bath(ψ_sb_evolved)
        truncate!(ψ_s; cutoff)

        E_list[step] = energy(ψ_s, H_sys)
        GS_overlap_list[step] = abs2(inner(ψ_s, ϕ₀))
        nb_list[step] = mean(v_b .- 1)

        println("Step $step: energy/N=$(E_list[step]/N), overlap=$(GS_overlap_list[step]), DmaxSB=$(maxlinkdim(ψ_sb_evolved)), DmaxS=$(maxlinkdim(ψ_s)), <nb>=$(nb_list[step])")
    end

    println("After cooling: energy/N=$(E_list[end]/N), overlap=$(GS_overlap_list[end])")

    return E_list, GS_overlap_list, nb_list
end

function evolve_state(H, ψ, t; Dmax, cutoff)
    ψ_evolved = tdvp(H, -im * t, ψ; nsweeps=1, reverse_step=false, normalize=true, maxdim=Dmax, cutoff=cutoff, outputlevel=0)
    normalize!(ψ_evolved)
    orthogonalize!(ψ_evolved, 2)
    return ψ_evolved
end
