# ============================================================================
# Problem Setup with Substance Implementations
# ============================================================================

"""
    setup_problem(backend::CoolingBackend, N, ham_params::HamiltonianParameters, coupling_params, sim_params)

Generic interface for problem setup - substance in each dispatch method.
"""
function setup_problem(backend::CoolingBackend, N, ham_params::HamiltonianParameters, coupling_params, sim_params)
    error("setup_problem not implemented for ham_model=$(typeof(ham_params.model)) and backend=$(typeof(backend))")
end

# ED Backend - Direct implementation with substance
function setup_problem(backend::EDBackend, N, ham_params::HamiltonianParameters, coupling_params, sim_params)
    # Use unified setup_system dispatch for both Hamiltonian and ground state
    H_sys, Δ_ed, e₀, ϕ₀ = setup_system(N, ham_params, backend)
    
    # Set resonant cooling if Δ not specified
    updated_coupling_params = if coupling_params.delta === nothing
        # Use computed gap from setup_system
        BasicCouplingParameters(coupling_params.coupling, coupling_params.g, coupling_params.steps, coupling_params.te, Δ_ed)
    else
        coupling_params
    end
    
    # Build full system+bath Hamiltonian using dispatch
    H_full = construct_system_bath_hamiltonian(ham_params, backend, 2N, updated_coupling_params)
    
    return CoolingProblem(backend, H_sys, H_full, ϕ₀, e₀, nothing, (coupling_params=updated_coupling_params,))
end

# TN Backend - Direct implementation with substance and shared helper
function setup_problem(backend::TNBackend, N, ham_params::HamiltonianParameters, coupling_params, sim_params::UnifiedSimulationParameters)
    # Common TN setup
    sites = siteinds("S=1/2", 2N)
    sites_sys = sites[1:2:2N-1]
    sites_bath = sites[2:2:2N]
    
    # Get system Hamiltonian and ground state
    H_sys, Δ_dmrg, e₀, ϕ₀ = setup_system(N, ham_params, backend, sites_sys)
    
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

# Monte Carlo + Continuous Evolution (MPS-like) - Direct substance
function setup_tn_specific(backend::TNBackend, ::MonteCarloWavefunction, ::ContinuousEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    H_sys_bath = construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
    return CoolingProblem(backend, H_sys, H_sys_bath, ϕ₀, e₀, sites, (H_sys_bath=H_sys_bath, coupling_params=coupling_params))
end

# Density Matrix + Trotter Evolution (MPO-like) - Direct substance
function setup_tn_specific(backend::TNBackend, ::DensityMatrix, ::TrotterEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    gates = build_trotter_circuit(ham_params, backend, sites_sys, sites_bath, coupling_params, sim_params)
    return CoolingProblem(backend, H_sys, nothing, ϕ₀, e₀, sites, (gates=gates, coupling_params=coupling_params))
end

# Monte Carlo + Trotter Evolution (TrotterMPS-like) - Direct substance
function setup_tn_specific(backend::TNBackend, ::MonteCarloWavefunction, ::TrotterEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    gates = build_trotter_circuit_bath_coupling(ham_params, backend, sites_sys, sites_bath, coupling_params, sim_params)
    H_total = construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
    return CoolingProblem(backend, H_sys, H_total, ϕ₀, e₀, sites, 
                         (gates=gates, H_sys_bath=H_total, ham_param_struct=ham_params, coupling_params=coupling_params))
end

# Density Matrix + Continuous Evolution (less common) - Direct substance
function setup_tn_specific(backend::TNBackend, ::DensityMatrix, ::ContinuousEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    H_sys_bath = construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
    return CoolingProblem(backend, H_sys, H_sys_bath, ϕ₀, e₀, sites, (H_sys_bath=H_sys_bath, coupling_params=coupling_params))
end