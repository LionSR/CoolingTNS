using ITensors

function build_trotter_circuit_ising(sites_sys, sites_bath, ham_params, g, Δ, tau, coupling)
    N = length(sites_sys)
    J, h = ham_params
    op1, op2 = if length(coupling) == 2
        string(coupling[1]), string(coupling[2])
    else
        error("Invalid coupling: $coupling")
    end

    gates = ITensor[]
    for ind in 1:(N-1)
        s1 = sites_sys[ind]
        s2 = sites_sys[ind+1]
        b1 = sites_bath[ind]
        b2 = sites_bath[ind+1]
        hs_j = J * op("Z", s1) * op("Z", s2) + h * op("X", s1) * op("I", s2)

        hb_j = g * op(op1, s1) * op(op2, b1) - Δ / 2 * op("I", s1) * op("Z", b1)
        Gs_j = exp(-1.0im * tau / 2 * hs_j)
        Gb_j = exp(-1.0im * tau / 2 * hb_j)
        push!(gates, Gs_j)
        push!(gates, Gb_j)
    end
    sN = sites_sys[N]
    bN = sites_bath[N]
    hs_N = h * op("X", sN)
    hb_N = -Δ / 2 * op("Z", bN)
    hsb_N = g * op(op1, sN) * op(op2, bN)
    Gs_N = exp(-1.0im * tau / 2 * hs_N)
    Gb_N = exp(-1.0im * tau / 2 * hb_N)
    Gsb_N = exp(-1.0im * tau / 2 * hsb_N)
    push!(gates, Gs_N)
    push!(gates, Gb_N)
    push!(gates, Gsb_N)
    append!(gates, reverse(gates))
end


function build_trotter_circuit_niising(sites_sys, sites_bath, ham_params, g, Δ, tau, coupling)
    N = length(sites_sys)
    J, hx, hz = ham_params
    op1, op2 = if length(coupling) == 2
        string(coupling[1]), string(coupling[2])
    else
        error("Invalid coupling: $coupling")
    end

    gates = ITensor[]
    for ind in 1:(N-1)
        s1 = sites_sys[ind]
        s2 = sites_sys[ind+1]
        b1 = sites_bath[ind]
        b2 = sites_bath[ind+1]

        hs_j = J * op("Z", s1) * op("Z", s2) + hx * op("X", s1) * op("I", s2) + hz * op("Z", s1) * op("I", s2)
        hb_j = g * op(op1, s1) * op(op2, b1) - Δ / 2 * op("I", s1) * op("Z", b1)
        Gs_j = exp(-1.0im * tau / 2 * hs_j)
        Gb_j = exp(-1.0im * tau / 2 * hb_j)
        push!(gates, Gs_j)
        push!(gates, Gb_j)
    end
    sN = sites_sys[N]
    bN = sites_bath[N]
    hs_N = hx * op("X", sN) + hz * op("Z", sN)
    hb_N = -Δ / 2 * op("Z", bN)
    hsb_N = g * op(op1, sN) * op(op2, bN)
    Gs_N = exp(-1.0im * tau / 2 * hs_N)
    Gb_N = exp(-1.0im * tau / 2 * hb_N)
    Gsb_N = exp(-1.0im * tau / 2 * hsb_N)
    push!(gates, Gs_N)
    push!(gates, Gb_N)
    push!(gates, Gsb_N)
    append!(gates, reverse(gates))
end

function setup_problem_mpo(problem, N, ham_params, g, tau)
    sites = siteinds("S=1/2", 2N)
    sites_sys = sites[1:2:2N-1]
    sites_bath = sites[2:2:2N]

    H_sys, Δ, e₀, ϕ₀ = setup_system(problem, N, sites_sys, ham_params)

    ρ_s = MPO(sites_sys, "Id")
    ρ_s = ρ_s ./ √2
    build_trotter_circuit_fn = problem == "Ising" ? build_trotter_circuit_ising : build_trotter_circuit_niising
    gates = build_trotter_circuit_fn(sites_sys, sites_bath, ham_params, g, Δ, tau, "XX")
    return sites, H_sys, ρ_s, e₀, ϕ₀, gates
end

function run_cooling_mpo(sites, H_sys, ρ_s, steps, ϕ₀; trotter_steps, tau, cutoff, gates)
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]

    E_list = zeros(steps + 1)
    GS_overlap_list = zeros(steps + 1)

    E_list[1] = inner(H_sys, ρ_s) / tr(ρ_s)
    GS_overlap_list[1] = real(inner(ϕ₀', ρ_s, ϕ₀))

    println("Cooling starts")
    println("Step 1: energy/N=$(E_list[1]/N), overlap=$(GS_overlap_list[1])")

    for i = 2:steps+1
        ρ_sb = appendzeros_MPO(ρ_s, sites)
        for j = 1:trotter_steps
            ρ_sb = apply(gates, ρ_sb; apply_dag=true, cutoff=cutoff, move_sites_back=true)
        end
        ρ_s = partial_trace_bath(ρ_sb, sites, sites_sys)
        ρ_s = ρ_s / tr(ρ_s)

        E_list[i] = real(inner(H_sys, ρ_s))
        GS_overlap_list[i] = real(inner(ϕ₀', ρ_s, ϕ₀))
        truncate!(ρ_s, cutoff=cutoff)

        println("Step $i: energy/N = $(E_list[i]/N), gs_overlap = $(GS_overlap_list[i]), Dmax=$(maxlinkdim(ρ_s))")
    end
    return E_list, GS_overlap_list
end


nothing
