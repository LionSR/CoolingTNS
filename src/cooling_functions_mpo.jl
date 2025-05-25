using ITensors

function build_trotter_circuit(sites_sys, sites_bath, ham_params, coupling_params, sim_params, problem)
    N = length(sites_sys)
    g, Δ, coupling, tau = coupling_params["g"], coupling_params["Δ"], coupling_params["coupling"], sim_params["tau"]
    op1, op2 = parse_coupling(coupling)

    gates = ITensor[]
    for ind in eachindex(sites_sys)
        s1, b1 = sites_sys[ind], sites_bath[ind]
        hs = if problem == "Ising"
            J, h = ham_params
            ind < N ? J * op("Z", s1) * op("Z", sites_sys[ind+1]) + h * op("X", s1) * op("I", sites_sys[ind+1]) : h * op("X", s1)
        else # niIsing
            J, hx, hz = ham_params
            ind < N ? J * op("Z", s1) * op("Z", sites_sys[ind+1]) + hx * op("X", s1) * op("I", sites_sys[ind+1]) + hz * op("Z", s1) * op("I", sites_sys[ind+1]) :
                      hx * op("X", s1) + hz * op("Z", s1)
        end
        hsb = g * op(op1, s1) * op(op2, b1) - Δ / 2 * op("I", s1) * op("Z", b1)
        push!(gates, exp(-1.0im * tau / 2 * hs), exp(-1.0im * tau / 2 * hsb))
    end
    append!(gates, reverse(gates))
end

function build_trotter_circuit_ising(sites_sys, sites_bath, ham_params, coupling_params, sim_params)
    build_trotter_circuit(sites_sys, sites_bath, ham_params, coupling_params, sim_params, "Ising")
end

function build_trotter_circuit_niising(sites_sys, sites_bath, ham_params, coupling_params, sim_params)
    build_trotter_circuit(sites_sys, sites_bath, ham_params, coupling_params, sim_params, "niIsing")
end

function setup_init_state_mpo(sites; init_type="identity", theta=0.0)
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]
    
    if init_type == "identity"
        # Maximally mixed state (identity matrix)
        ρ_s = MPO(sites_sys, "Id")
        ρ_s = ρ_s ./ √2
    elseif init_type == "theta"
        # Create state based on theta angle (in units of pi)
        ψ_s = setup_init_state_mps(sites; init_type="theta", theta=theta)
        ρ_s = outer(ψ_s', ψ_s)
    else
        # Product state - create from MPS outer product
        ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        ρ_s = outer(ψ_s', ψ_s)
    end
    return ρ_s
end

function setup_problem_mpo(N, problem, ham_params, coupling_params, sim_params)
    sites = siteinds("S=1/2", 2N)
    sites_sys = sites[1:2:2N-1]
    sites_bath = sites[2:2:2N]

    H_sys, Δ_dmrg, e₀, ϕ₀ = setup_system(N, problem, sites_sys, ham_params)

    Δ = haskey(coupling_params, "Δ") ? coupling_params["Δ"] : Δ_dmrg
    coupling_params["Δ"] = Δ

    build_trotter_circuit_fn = problem == "Ising" ? build_trotter_circuit_ising : build_trotter_circuit_niising
    gates = build_trotter_circuit_fn(sites_sys, sites_bath, ham_params, coupling_params, sim_params)
    return sites, H_sys, ϕ₀, e₀, gates
end

function apply_cooling_step(ρ_s, sites, gates, noise_layer, trotter_steps, cutoff, Dmax, pe)
    ρ_sb = appendzeros_MPO(ρ_s, sites)
    for _ in 1:trotter_steps
        ρ_sb = apply(gates, ρ_sb; apply_dag=true, cutoff=cutoff, maxdim=Dmax, move_sites_back=true)
    end
    if pe > 0
        ρ_sb = apply(noise_layer, ρ_sb; apply_dag=true, cutoff=cutoff, maxdim=Dmax, move_sites_back=true)
    end
    return ρ_sb
end

function run_cooling_mpo(sites, H_sys, ϕ₀, gates, ρ_s, coupling_params, sim_params)
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]
    steps, trotter_steps = coupling_params["steps"], sim_params["trotter_steps"]
    cutoff, Dmax, pe = sim_params["cutoff"], sim_params["Dmax"], sim_params["pe"]

    noise_layer = pe > 0 ? [depolarizing_noise(sites[i], pe) for i = 1:2N] : nothing

    E_list = zeros(Float64, steps + 1)
    GS_overlap_list = zeros(Float64, steps + 1)

    E_list[1] = real(inner(H_sys, ρ_s) / tr(ρ_s))
    GS_overlap_list[1] = real(inner(ϕ₀', ρ_s, ϕ₀))

    println("Cooling starts")
    println("Step 1: energy/N=$(E_list[1]/N), overlap=$(GS_overlap_list[1])")

    for i in 2:steps+1
        ρ_sb = apply_cooling_step(ρ_s, sites, gates, noise_layer, trotter_steps, cutoff, Dmax, pe)
        ρ_s = partial_trace_bath(ρ_sb, sites, sites_sys)
        ρ_s = ρ_s / tr(ρ_s)

        E_list[i] = real(inner(H_sys, ρ_s))
        GS_overlap_list[i] = real(inner(ϕ₀', ρ_s, ϕ₀))
        truncate!(ρ_s, cutoff=cutoff, maxdim=Dmax)

        println("Step $i: energy/N = $(E_list[i]/N), gs_overlap = $(GS_overlap_list[i]), Dmax=$(maxlinkdim(ρ_s))")
    end
    return Dict(
        "E_list" => E_list,
        "GS_overlap_list" => GS_overlap_list,
        "final_state" => ρ_s
    )
end


nothing
