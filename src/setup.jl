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

_assert_supported_tn_multifrequency(sim_params::UnifiedSimulationParameters) =
    _assert_supported_tn_multifrequency(sim_params.sim_method, sim_params.evolution_method)

_assert_supported_tn_multifrequency(::MonteCarloWavefunction, ::ContinuousEvolution) = nothing
_assert_supported_tn_multifrequency(::MonteCarloWavefunction, ::TrotterEvolution) = nothing
_assert_supported_tn_multifrequency(::DensityMatrix, ::TrotterEvolution) = nothing

function _assert_supported_tn_multifrequency(
    sim_method::SimulationMethod,
    evolution_method::EvolutionMethod,
)
    error(
        "Multi-frequency cooling for TN is currently implemented for either " *
        "(1) MonteCarloWavefunction + ContinuousEvolution (MPS + TDVP), " *
        "(2) MonteCarloWavefunction + TrotterEvolution (MPS + Trotter), or " *
        "(3) DensityMatrix + TrotterEvolution (MPO + Trotter). " *
        "Got sim_method=$(typeof(sim_method)), evolution_method=$(typeof(evolution_method)).",
    )
end

function _tn_multifrequency_extra(
    ::TrotterEvolution,
    ham_params::HamiltonianParameters,
    coupling_params::MultiFrequencyCouplingParameters,
    sites::Vector{<:Index},
    gap::Real,
)
    return (
        coupling_params=coupling_params,
        coupling=coupling_params.coupling,
        g=coupling_params.g,
        sites=sites,
        # Diagnostic/default-grid metadata only. The multi-frequency evolution
        # uses the per-step `coupling_params.delta_values`, not this field.
        gap=Float64(gap),
        ham_params=ham_params,
        gates_cache=Dict{Float64, Any}(),
        trotter_step_gates_cache=Dict{Any, Any}(),
    )
end

function _tn_multifrequency_extra(
    ::ContinuousEvolution,
    ham_params::HamiltonianParameters,
    coupling_params::MultiFrequencyCouplingParameters,
    sites::Vector{<:Index},
    gap::Real,
)
    return (
        coupling_params=coupling_params,
        coupling=coupling_params.coupling,
        g=coupling_params.g,
        sites=sites,
        # Diagnostic/default-grid metadata only. The multi-frequency evolution
        # uses the per-step `coupling_params.delta_values`, not this field.
        gap=Float64(gap),
        ham_params=ham_params,
        # Cache step Hamiltonians H_SB(Δ) for multi-frequency MCWF+TDVP runs.
        # Keyed by the bath detuning Δ (Float64).
        H_cache=Dict{Float64, Any}(),
    )
end

"""
    setup_tn_multifrequency_problem_from_system(
        backend, ham_params, coupling_params, sim_params, sites, H_sys, gap, e₀, ϕ₀
    )

Build a tensor-network multi-frequency cooling problem from an already computed
system solve. This preserves the same method-specific `extra` fields as
`setup_problem(::TNBackend, ::HamiltonianParameters, ::MultiFrequencyCouplingParameters, ...)`,
but avoids recomputing the system DMRG ground state when only the detuning grid
changes.
"""
function setup_tn_multifrequency_problem_from_system(
    backend::TNBackend,
    ham_params::HamiltonianParameters,
    coupling_params::MultiFrequencyCouplingParameters,
    sim_params::UnifiedSimulationParameters,
    sites::Vector{<:Index},
    H_sys,
    gap::Real,
    e₀::Real,
    ϕ₀,
)
    _assert_supported_tn_multifrequency(sim_params)
    expected_sites = interleaved_total_sites(ham_params.N)
    length(sites) == expected_sites || throw(ArgumentError(
        "TN multi-frequency setup requires interleaved system-bath sites with " *
        "length $expected_sites for N=$(ham_params.N); got length $(length(sites)).",
    ))
    extra = _tn_multifrequency_extra(
        sim_params.evolution_method,
        ham_params,
        coupling_params,
        sites,
        gap,
    )

    return CoolingProblem(backend, H_sys, nothing, ϕ₀, Float64(e₀), extra)
end

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
    N = ham_params.N
    sites = siteinds("S=1/2", interleaved_total_sites(N))
    sites_sys = interleaved_system_indices(sites, N)

    H_sys, Δ_dmrg, e₀, ϕ₀ = setup_system(ham_params, backend, sites_sys)

    return setup_tn_multifrequency_problem_from_system(
        backend,
        ham_params,
        coupling_params,
        sim_params,
        sites,
        H_sys,
        Δ_dmrg,
        e₀,
        ϕ₀,
    )
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
                          coupling=coupling_params.coupling, g=coupling_params.g, sites=sites,
                          trotter_step_gates_cache=Dict{Any, Any}()))
end

# Monte Carlo + Trotter Evolution - Direct substance
# Uses same interleaved gates as DM+Trotter for consistency
function setup_tn_specific(backend::TNBackend, ::MonteCarloWavefunction, ::TrotterEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    interleaved_gates = build_trotter_circuit_interleaved(ham_params, backend, sites, coupling_params, sim_params)
    H_total = construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
    return CoolingProblem(backend, H_sys, H_total, ϕ₀, e₀,
                         (interleaved_gates=interleaved_gates, H_sys_bath=H_total, ham_param_struct=ham_params, coupling_params=coupling_params,
                          coupling=coupling_params.coupling, g=coupling_params.g, sites=sites,
                          trotter_step_gates_cache=Dict{Any, Any}()))
end

# Density Matrix + Continuous Evolution (less common) - Direct substance
function setup_tn_specific(backend::TNBackend, ::DensityMatrix, ::ContinuousEvolution,
                          ham_params, sites, sites_sys, sites_bath, H_sys, e₀, ϕ₀, coupling_params, sim_params)
    H_sys_bath = construct_system_bath_hamiltonian(ham_params, backend, sites, coupling_params)
    return CoolingProblem(backend, H_sys, H_sys_bath, ϕ₀, e₀,
                         (H_sys_bath=H_sys_bath, coupling_params=coupling_params, 
                          coupling=coupling_params.coupling, g=coupling_params.g, sites=sites))
end
