using ITensors
using ITensorTDVP

function setup_problem_trotter_mps(N, problem, ham_params, coupling_params, sim_params)
    sites = siteinds("S=1/2", 2N)
    sites_sys = sites[1:2:2N-1]
    sites_bath = sites[2:2:2N]

    H_sys, Δ_dmrg, e₀, ϕ₀ = setup_system(N, problem, sites_sys, ham_params)

    Δ = haskey(coupling_params, "Δ") ? coupling_params["Δ"] : Δ_dmrg
    coupling_params["Δ"] = Δ

    gates = build_trotter_circuit_bath_coupling(sites_sys, sites_bath, coupling_params, sim_params)
    
    return sites, H_sys, ϕ₀, e₀, gates
end

function build_trotter_circuit_bath_coupling(sites_sys, sites_bath, coupling_params, sim_params)
    N = length(sites_sys)
    g, Δ, coupling, tau = coupling_params["g"], coupling_params["Δ"], coupling_params["coupling"], sim_params["tau"]
    op1, op2 = parse_coupling(coupling)

    gates = ITensor[]
    for ind in eachindex(sites_sys)
        s1, b1 = sites_sys[ind], sites_bath[ind]
        hb = -Δ / 2 * op("Z", b1)
        hsb = g * op(op1, s1) * op(op2, b1)
        push!(gates, exp(-1.0im * tau / 2 * hb), exp(-1.0im * tau / 2 * hsb))
    end
    append!(gates, reverse(gates))
    return gates
end

function build_trotter_circuit_bath_coupling_ising(sites_sys, sites_bath, coupling_params, sim_params)
    build_trotter_circuit_bath_coupling(sites_sys, sites_bath, coupling_params, sim_params)
end

function build_trotter_circuit_bath_coupling_niising(sites_sys, sites_bath, coupling_params, sim_params)
    build_trotter_circuit_bath_coupling(sites_sys, sites_bath, coupling_params, sim_params)
end

function evolve_state_trotter(H_sys, V, ψ, t; Dmax, cutoff, tau)
    # Evolve with H_sys using TDVP
    ψ_evolved = tdvp(H_sys, -im * t/2, ψ; time_step=-im * tau/2, reverse_step=false, normalize=true, maxdim=Dmax, cutoff=cutoff, outputlevel=0)
    
    # Evolve with V using TEBD-like gate application
    gates = ITensor[]
    for j in 1:length(V)
        push!(gates, exp(-im * t/2 * V[j]))
    end
    ψ_evolved = apply(gates, ψ_evolved; cutoff=cutoff, maxdim=Dmax)
    
    # Evolve with H_sys again
    ψ_evolved = tdvp(H_sys, -im * t/2, ψ_evolved; time_step=-im * tau/2, reverse_step=false, normalize=true, maxdim=Dmax, cutoff=cutoff, outputlevel=0)
    
    normalize!(ψ_evolved)
    orthogonalize!(ψ_evolved, 2)
    return ψ_evolved
end

function run_cooling_trotter_mps(sites, H_sys, ϕ₀, gates, ψ_s, coupling_params, sim_params)
    steps, te = coupling_params["steps"], coupling_params["te"]
    cutoff, Dmax, tau, pe = sim_params["cutoff"], sim_params["Dmax"], sim_params["tau"], sim_params["pe"]
    N = length(sites) ÷ 2

    E_list = zeros(Float64, steps + 1)
    GS_overlap_list = zeros(Float64, steps + 1)
    nb_list = zeros(Float64, steps + 1)

    E_list[1] = real(inner(ψ_s', H_sys, ψ_s))
    GS_overlap_list[1] = abs2(inner(ψ_s, ϕ₀))

    println("Cooling starts")
    println("Step 1: energy/N=$(E_list[1]/N), overlap=$(GS_overlap_list[1])")

    for step = 2:steps+1
        ψ_sb = appendzeros_MPS(ψ_s, sites)
        
        # Apply the Trotter gates
        for _ in 1:Int(te/tau)
            ψ_sb = apply(gates, ψ_sb; cutoff=cutoff, maxdim=Dmax)
        end
        
        if pe > 0
            ψ_sb = apply_depolarizing_noise(ψ_sb, sites, pe)
            orthogonalize!(ψ_sb, 2)
        end

        v_b, ψ_s = sample_bath(ψ_sb)
        truncate!(ψ_s; cutoff)
        normalize!(ψ_s)

        E_list[step] = real(inner(ψ_s', H_sys, ψ_s))
        GS_overlap_list[step] = abs2(inner(ψ_s, ϕ₀))
        nb_list[step] = mean(v_b .- 1)

        println("Step $step: energy/N=$(E_list[step]/N), overlap=$(GS_overlap_list[step]), DmaxSB=$(maxlinkdim(ψ_sb)), DmaxS=$(maxlinkdim(ψ_s)), <nb>=$(nb_list[step])")
    end

    println("After cooling: energy/N=$(E_list[end]/N), overlap=$(GS_overlap_list[end])")

    return Dict(
        "E_list" => E_list,
        "GS_overlap_list" => GS_overlap_list,
        "nb_list" => nb_list,
        "final_state" => ψ_s
    )
end
