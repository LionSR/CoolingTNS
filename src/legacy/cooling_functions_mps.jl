using ITensors
using ITensorMPS
using Statistics

function setup_init_state_mps(sites; init_type="product", theta=0.0)
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]

    if init_type == "identity"
        # Create maximally mixed state (identity/N^N) by superposition
        # Start with equal superposition of all computational basis states
        ψ_s = randomMPS(sites_sys, linkdims=1)
        normalize!(ψ_s)
    elseif init_type == "theta"
        # Create state based on theta angle (in units of pi)
        theta_rad = theta * π
        if abs(theta + 0.5) < 1e-10  # All down
            ψ_s = MPS(sites_sys, "Dn")
        elseif abs(theta - 0.5) < 1e-10  # All up
            ψ_s = MPS(sites_sys, "Up")
        elseif abs(theta) < 1e-10  # X+ state
            ψ_s = MPS(sites_sys, "X+")
        else
            # General theta not implemented for MPS yet
            @warn "General theta states not implemented for MPS, using default alternating"
            ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        end
    else
        # Default product state
        # ψ_s = randomMPS(sites_sys, linkdims=1)
        # ψ_s = MPS(sites_sys, "Dn")
        # ψ_s = MPS(sites_sys, "X+")
        # ψ_s = MPS(sites_sys, "X-")
        ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
    end
    return ψ_s
end


# Multiple dispatch versions for typed parameters
function setup_problem_mps(N, problem, ham_params, coupling_params::CouplingParameters, sim_params::SimulationParameters)
    sites = siteinds("S=1/2", 2N)
    sites_sys = sites[1:2:2N-1]

    # Create HamiltonianParameters struct from legacy parameters
    ham_param_struct = create_hamiltonian_params(problem, ham_params...)
    
    # Use new dispatch system
    H_sys, Δ_dmrg, e₀, ϕ₀ = setup_system(N, ham_param_struct, TNBackend(), sites_sys)

    # Create updated coupling parameters with computed delta
    Δ = hasfield(typeof(coupling_params), :delta) && coupling_params.delta !== nothing ? coupling_params.delta : Δ_dmrg
    updated_coupling_params = typeof(coupling_params)(coupling_params.coupling, coupling_params.g, coupling_params.steps, coupling_params.te, Δ)

    # HamiltonianParameters struct already created above
    backend = MPSBackend()
    H_sys_bath = construct_system_bath_hamiltonian(ham_param_struct, backend, sites, to_dict(updated_coupling_params))

    return sites, H_sys, ϕ₀, e₀, H_sys_bath
end

# Removed backward compatibility - use typed parameters only


function evolve_state(H, ψ, t; Dmax, cutoff, tau)
    ψ_evolved = tdvp(H, -im * t, ψ; time_step=-1im * tau, reverse_step=false, normalize=true, maxdim=Dmax, cutoff=cutoff, outputlevel=0)
    normalize!(ψ_evolved)
    orthogonalize!(ψ_evolved, 2)
    return ψ_evolved
end


# Multiple dispatch version for typed parameters
function run_cooling_mps(sites, H_sys, ϕ₀, H_sys_bath, ψ_s, coupling_params::CouplingParameters, sim_params::TensorNetworkParameters)
    steps, te = coupling_params.steps, coupling_params.te
    cutoff, Dmax, tau, pe = sim_params.cutoff, sim_params.Dmax, sim_params.tau, sim_params.pe
    N = length(sites) ÷ 2

    E_list = zeros(Float64, steps + 1)
    GS_overlap_list = zeros(Float64, steps + 1)
    nb_list = zeros(Float64, steps + 1)

    E_list[1] = real(inner(ψ_s', H_sys, ψ_s))
    GS_overlap_list[1] = abs2(inner(ψ_s, ϕ₀))

    println("Cooling starts")
    println("Step 1: energy/N=$(E_list[1]/N), overlap=$(GS_overlap_list[1])")

    for step in 2:steps+1
        ψ_sb = appendzeros_MPS(ψ_s, sites)
        ψ_sb_evolved = evolve_state(H_sys_bath, ψ_sb, te; Dmax, cutoff, tau)
        
        if pe > 0
            ψ_sb_evolved = apply_depolarizing_noise(ψ_sb_evolved, sites, pe)
            orthogonalize!(ψ_sb_evolved, 2)
        end

        v_b, ψ_s = sample_bath(ψ_sb_evolved)
        truncate!(ψ_s; cutoff)
        normalize!(ψ_s)

        E_list[step] = real(inner(ψ_s', H_sys, ψ_s))
        GS_overlap_list[step] = abs2(inner(ψ_s, ϕ₀))
        nb_list[step] = mean(v_b .- 1)

        println("Step $step: energy/N=$(E_list[step]/N), overlap=$(GS_overlap_list[step]), DmaxSB=$(maxlinkdim(ψ_sb_evolved)), DmaxS=$(maxlinkdim(ψ_s)), <nb>=$(nb_list[step])")
    end

    println("After cooling: energy/N=$(E_list[end]/N), overlap=$(GS_overlap_list[end])")

    return TensorNetworkResults(
        E_list,
        GS_overlap_list,
        nb_list,
        ψ_s
    )
end

# Removed backward compatibility - use typed parameters only

