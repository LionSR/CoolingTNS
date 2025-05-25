using ITensors
# Removed legacy trotter circuit functions - use hamiltonian_dispatch.jl instead

# Removed - use generic setup_init_state_mps and convert to MPO if needed

# Multiple dispatch version for typed parameters
function setup_problem_mpo(N, problem, ham_params, coupling_params::CouplingParameters, sim_params::TensorNetworkParameters)
    sites = siteinds("S=1/2", 2N)
    sites_sys = sites[1:2:2N-1]
    sites_bath = sites[2:2:2N]

    # Create HamiltonianParameters struct from legacy parameters
    ham_param_struct = create_hamiltonian_params(problem, ham_params...)
    
    # Use new dispatch system
    H_sys, Δ_dmrg, e₀, ϕ₀ = setup_system(N, ham_param_struct, TNBackend(), sites_sys)

    # Create updated coupling parameters with computed delta
    Δ = hasfield(typeof(coupling_params), :delta) && coupling_params.delta !== nothing ? coupling_params.delta : Δ_dmrg
    updated_coupling_params = typeof(coupling_params)(coupling_params.coupling, coupling_params.g, coupling_params.steps, coupling_params.te, Δ)

    # Create HamiltonianParameters struct
    ham_param_struct = create_hamiltonian_params(problem, ham_params...)
    backend = MPOBackend()
    gates = build_trotter_circuit(ham_param_struct, backend, sites_sys, sites_bath, to_dict(updated_coupling_params), to_dict(sim_params))
    return sites, H_sys, ϕ₀, e₀, gates
end

# Removed backward compatibility - use typed parameters only

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

# Multiple dispatch version for typed parameters
function run_cooling_mpo(sites, H_sys, ϕ₀, gates, ρ_s, coupling_params::CouplingParameters, sim_params::TensorNetworkParameters)
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]
    steps = coupling_params.steps
    trotter_steps = sim_params.trotter_steps
    cutoff, Dmax, pe = sim_params.cutoff, sim_params.Dmax, sim_params.pe

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
    
    return TensorNetworkResults(
        E_list,
        GS_overlap_list,
        Float64[],  # No bath magnetization for MPO
        ρ_s
    )
end

# Removed backward compatibility - use typed parameters only


nothing
