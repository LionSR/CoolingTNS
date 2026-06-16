# ============================================================================
# Problem Setup with Substance Implementations
# ============================================================================

"""
    setup_problem(backend::CoolingBackend, ham_params::HamiltonianParameters, coupling_params, sim_params)

Generic interface for problem setup - substance in each dispatch method.
"""
function setup_problem(backend::CoolingBackend, ham_params::HamiltonianParameters, coupling_params, sim_params)
    error("setup_problem not implemented for ham_model=$(typeof(ham_params.model)) and backend=$(typeof(backend))")
end

# ---------------------------------------------------------------------------
# Multi-frequency (multi-Δ) cooling setup
# ---------------------------------------------------------------------------

function setup_problem(
    backend::EDBackend,
    ham_params::HamiltonianParameters,
    coupling_params::MultiFrequencyCouplingParameters,
    _sim_params,
)
    H_sys, Δ_ed, e₀, ϕ₀ = setup_system(ham_params, backend)

    # For multi-frequency protocols, Δ changes every step, so we do not prebuild H_sys_bath.
    return CoolingProblem(
        backend,
        H_sys,
        nothing,
        ϕ₀,
        e₀,
        (
            coupling_params=coupling_params,
            ham_params=ham_params,
            coupling=coupling_params.coupling,
            g=coupling_params.g,
            gap=Δ_ed,
            # Cache step Hamiltonians H_SB(Δ) for multi-frequency / randomized-time ED runs.
            # Keyed by the bath detuning Δ (Float64).
            H_cache=Dict{Float64, Any}(),
        ),
    )
end

function setup_problem(
    backend::TNBackend,
    ham_params::HamiltonianParameters,
    coupling_params::MultiFrequencyCouplingParameters,
    sim_params::UnifiedSimulationParameters,
)
    supported =
        (sim_params.sim_method isa MonteCarloWavefunction && sim_params.evolution_method isa ContinuousEvolution) ||
        (sim_params.sim_method isa MonteCarloWavefunction && sim_params.evolution_method isa TrotterEvolution) ||
        (sim_params.sim_method isa DensityMatrix && sim_params.evolution_method isa TrotterEvolution)

    if !supported
        error(
            "Multi-frequency cooling for TN is currently implemented for either " *
            "(1) MonteCarloWavefunction + ContinuousEvolution (MPS + TDVP), " *
            "(2) MonteCarloWavefunction + TrotterEvolution (MPS + Trotter), or " *
            "(3) DensityMatrix + TrotterEvolution (MPO + Trotter). " *
            "Got sim_method=$(typeof(sim_params.sim_method)), evolution_method=$(typeof(sim_params.evolution_method)).",
        )
    end

    N = ham_params.N
    sites = siteinds("S=1/2", interleaved_total_sites(N))
    sites_sys = interleaved_system_indices(sites, N)

    H_sys, Δ_dmrg, e₀, ϕ₀ = setup_system(ham_params, backend, sites_sys)

    extra =
        if sim_params.evolution_method isa TrotterEvolution
            (
                coupling_params=coupling_params,
                coupling=coupling_params.coupling,
                g=coupling_params.g,
                sites=sites,
                gap=Δ_dmrg,
                ham_params=ham_params,
                gates_cache=Dict{Float64, Any}(),
            )
        else
            (
                coupling_params=coupling_params,
                coupling=coupling_params.coupling,
                g=coupling_params.g,
                sites=sites,
                gap=Δ_dmrg,
                ham_params=ham_params,
                # Cache step Hamiltonians H_SB(Δ) for multi-frequency MCWF+TDVP runs.
                # Keyed by the bath detuning Δ (Float64).
                H_cache=Dict{Float64, Any}(),
            )
        end

    return CoolingProblem(backend, H_sys, nothing, ϕ₀, e₀, extra)
end

# ED Backend - Direct implementation with substance
function setup_problem(backend::EDBackend, ham_params::HamiltonianParameters, coupling_params, sim_params)
    # Use unified setup_system dispatch for both Hamiltonian and ground state
    H_sys, Δ_ed, e₀, ϕ₀ = setup_system(ham_params, backend)
    
    # Set resonant cooling if Δ not specified
    updated_coupling_params = if coupling_params.delta === nothing
        # Use computed gap from setup_system
        BasicCouplingParameters(coupling_params.coupling, coupling_params.g, coupling_params.steps, coupling_params.te, Δ_ed)
    else
        coupling_params
    end
    
    # Build full system+bath Hamiltonian using dispatch
    H_full = construct_system_bath_hamiltonian(ham_params, backend, 2*ham_params.N, updated_coupling_params)
    
    return CoolingProblem(backend, H_sys, H_full, ϕ₀, e₀,
                         (coupling_params=updated_coupling_params, ham_params=ham_params, 
                          coupling=updated_coupling_params.coupling, g=updated_coupling_params.g))
end

# TN Backend - Direct implementation with substance and shared helper
function setup_problem(backend::TNBackend, ham_params::HamiltonianParameters, coupling_params, sim_params::UnifiedSimulationParameters)
    # Common TN setup
    N = ham_params.N
    sites = siteinds("S=1/2", interleaved_total_sites(N))
    sites_sys = interleaved_system_indices(sites, N)
    sites_bath = interleaved_bath_indices(sites, N)
    
    # Get system Hamiltonian and ground state
    H_sys, Δ_dmrg, e₀, ϕ₀ = setup_system(ham_params, backend, sites_sys)
    
    # Update coupling parameters with computed delta
    updated_coupling_params = if coupling_params.delta === nothing
        BasicCouplingParameters(coupling_params.coupling, coupling_params.g, coupling_params.steps, coupling_params.te, Δ_dmrg)
    else
        coupling_params
    end
    
    # Dispatch based on simulation method and evolution method
    return setup_tn_specific(backend, sim_params.sim_method, sim_params.evolution_method, 
                            ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, updated_coupling_params, sim_params)
end

# Monte Carlo + Continuous Evolution - Direct substance
function setup_tn_specific(backend::TNBackend, ::MonteCarloWavefunction, ::ContinuousEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    H_sys_bath = construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
    return CoolingProblem(backend, H_sys, H_sys_bath, ϕ₀, e₀,
                         (H_sys_bath=H_sys_bath, coupling_params=coupling_params, 
                          coupling=coupling_params.coupling, g=coupling_params.g, sites=sites))
end

# Density Matrix + Trotter Evolution - Direct substance
function setup_tn_specific(backend::TNBackend, ::DensityMatrix, ::TrotterEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    # Use interleaved gates that act on adjacent sites in [s1,b1,s2,b2,...] layout
    interleaved_gates = build_trotter_circuit_interleaved(ham_params, backend, sites, coupling_params, sim_params)
    return CoolingProblem(backend, H_sys, nothing, ϕ₀, e₀,
                         (interleaved_gates=interleaved_gates, coupling_params=coupling_params,
                          coupling=coupling_params.coupling, g=coupling_params.g, sites=sites))
end

# Monte Carlo + Trotter Evolution - Direct substance
# Uses same interleaved gates as DM+Trotter for consistency
function setup_tn_specific(backend::TNBackend, ::MonteCarloWavefunction, ::TrotterEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    interleaved_gates = build_trotter_circuit_interleaved(ham_params, backend, sites, coupling_params, sim_params)
    H_total = construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
    return CoolingProblem(backend, H_sys, H_total, ϕ₀, e₀,
                         (interleaved_gates=interleaved_gates, H_sys_bath=H_total, ham_param_struct=ham_params, coupling_params=coupling_params,
                          coupling=coupling_params.coupling, g=coupling_params.g, sites=sites))
end

# Density Matrix + Continuous Evolution (less common) - Direct substance
function setup_tn_specific(backend::TNBackend, ::DensityMatrix, ::ContinuousEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    H_sys_bath = construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
    return CoolingProblem(backend, H_sys, H_sys_bath, ϕ₀, e₀,
                         (H_sys_bath=H_sys_bath, coupling_params=coupling_params, 
                          coupling=coupling_params.coupling, g=coupling_params.g, sites=sites))
end
