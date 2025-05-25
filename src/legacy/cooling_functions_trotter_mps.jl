using ITensors
using ITensorMPS
include("parameter_types.jl")
include("hamiltonian_dispatch.jl")
include("setup_system_dispatch.jl")

# Multiple dispatch version for typed parameters
function setup_problem_trotter_mps(N, problem, ham_params, coupling_params::CouplingParameters, sim_params::TensorNetworkParameters)
    sites = siteinds("S=1/2", 2N)
    sites_sys = sites[1:2:2N-1]
    sites_bath = sites[2:2:2N]

    # Create HamiltonianParameters struct from legacy parameters
    ham_param_struct = create_hamiltonian_params(problem, ham_params...)
    
    # Use new dispatch system
    H_sys, Δ_dmrg, e₀, ϕ₀ = setup_system(N, ham_param_struct, TNBackend(), sites_sys)

    # Create updated coupling parameters with computed delta
    Δ = coupling_params.delta !== nothing ? coupling_params.delta : Δ_dmrg
    updated_coupling_params = CouplingParameters(coupling_params.coupling, coupling_params.g, coupling_params.steps, coupling_params.te, Δ)
    
    backend = TrotterMPSBackend()
    gates = build_trotter_circuit_bath_coupling(ham_param_struct, backend, sites_sys, sites_bath, to_dict(updated_coupling_params), to_dict(sim_params))
    
    # Create the total Hamiltonian using dispatch
    H_total = construct_system_bath_hamiltonian(ham_param_struct, backend, sites, to_dict(updated_coupling_params))
    
    return sites, H_sys, H_total, ϕ₀, e₀, gates, ham_param_struct
end

# Removed backward compatibility - use typed parameters only

# Removed legacy functions - use hamiltonian_dispatch.jl instead

# Multiple dispatch version for typed parameters
function run_cooling_trotter_mps(sites, H_sys, H_total, ϕ₀, gates, ψ_s, coupling_params::CouplingParameters, sim_params::TensorNetworkParameters, ham_param_struct)
    steps, te = coupling_params.steps, coupling_params.te
    cutoff, Dmax, tau, pe = sim_params.cutoff, sim_params.Dmax, sim_params.tau, sim_params.pe
    N = length(sites) ÷ 2

    E_list = zeros(Float64, steps + 1)
    E_total_list = zeros(Float64, steps + 1)
    GS_overlap_list = zeros(Float64, steps + 1)
    nb_list = zeros(Float64, steps + 1)

    E_list[1] = real(inner(ψ_s', H_sys, ψ_s))
    E_total_list[1] = real(inner(appendzeros_MPS(ψ_s, sites)', H_total, appendzeros_MPS(ψ_s, sites)))
    GS_overlap_list[1] = abs2(inner(ψ_s, ϕ₀))

    println("Cooling starts")
    println("Step 1: energy/N=$(E_list[1]/N), total energy=$(E_total_list[1]), overlap=$(GS_overlap_list[1])")

    for step = 2:steps+1
        ψ_sb = appendzeros_MPS(ψ_s, sites)
        
        # Use new evolve_state dispatch with Trotter evolution
        unified_sim_params = UnifiedSimulationParameters{MonteCarloWavefunction, TrotterEvolution}(
            MonteCarloWavefunction(), TrotterEvolution(), Dmax, cutoff, tau, pe, nothing
        )
        ψ_sb = evolve_state(ham_param_struct, unified_sim_params, TrotterMPSBackend(), H_total, ψ_sb, te, sites; gates=gates)    
        
        if pe > 0
            ψ_sb = apply_depolarizing_noise(ψ_sb, sites, pe)
            orthogonalize!(ψ_sb, 2)
        end

        v_b, ψ_s = sample_bath(ψ_sb)
        truncate!(ψ_s; cutoff)
        normalize!(ψ_s)

        E_list[step] = real(inner(ψ_s', H_sys, ψ_s))
        E_total_list[step] = real(inner(ψ_sb', H_total, ψ_sb))
        GS_overlap_list[step] = abs2(inner(ψ_s, ϕ₀))
        nb_list[step] = mean(v_b .- 1)

        println("Step $step: energy/N=$(E_list[step]/N), total energy=$(E_total_list[step]), overlap=$(GS_overlap_list[step]), DmaxSB=$(maxlinkdim(ψ_sb)), DmaxS=$(maxlinkdim(ψ_s)), <nb>=$(nb_list[step])")
    end

    println("After cooling: energy/N=$(E_list[end]/N), total energy=$(E_total_list[end]), overlap=$(GS_overlap_list[end])")

    # Create extended TensorNetworkResults with total energy
    result = TensorNetworkResults(
        E_list,
        GS_overlap_list,
        nb_list,
        ψ_s
    )
    
    # Add E_total_list as additional data
    return (result=result, E_total_list=E_total_list)
end

# Removed backward compatibility - use typed parameters only
