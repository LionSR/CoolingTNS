"""
    evolution.jl

Time evolution methods using multiple dispatch on SimulationMethod, EvolutionMethod, and backend.
"""

using ITensors
using ITensorMPS

# ============================================================================
# Time Evolution Interface
# ============================================================================

"""
    evolve_state(ham_params, sim_params, backend, H_total, state, t, sites; kwargs...)

Generic evolution interface using triple dispatch on model, simulation method, and backend.
"""
function evolve_state(ham_params::HamiltonianParameters, sim_params::UnifiedSimulationParameters,
                     backend::CoolingBackend, H_total, ψ, t, sites::Union{Nothing, Vector{<:Index}}; kwargs...)
    error("evolve_state not implemented for model=$(typeof(ham_params.model)), " *
          "sim_method=$(typeof(sim_params.sim_method)), " *
          "evolution_method=$(typeof(sim_params.evolution_method)), " *
          "backend=$(typeof(backend))")
end

# ============================================================================
# Monte Carlo + Continuous Evolution + Tensor Networks
# ============================================================================

function evolve_state(::HamiltonianParameters, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, ContinuousEvolution},
                     ::TNBackend, H_total, ψ, t::Float64, ::Vector{<:Index}; kwargs...)
    Dmax, cutoff, tau = sim_params.Dmax, sim_params.cutoff, sim_params.tau
    @debug "evolve_state MC+Continuous: Dmax=$Dmax, tau=$tau, t=$t, nsite=2"

    # Use nsite=2 to allow bond dimension growth from product states
    ψ_evolved = tdvp(H_total, -im * t, ψ;
                     time_step=-im * tau, nsite=2, reverse_step=true, normalize=true,
                     maxdim=Dmax, cutoff=cutoff, outputlevel=0)
    normalize!(ψ_evolved)
    orthogonalize!(ψ_evolved, 2)
    return ψ_evolved
end

# ============================================================================
# Monte Carlo + Trotter Evolution + Tensor Networks
# ============================================================================

function evolve_state(ham_params::HamiltonianParameters, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, TrotterEvolution},
                     ::TNBackend, ::Any, ψ, t::Float64, ::Vector{<:Index}; gates=nothing, kwargs...)
    gates === nothing && error("Trotter evolution requires pre-computed gates")

    Dmax, cutoff, tau = sim_params.Dmax, sim_params.cutoff, sim_params.tau
    steps = max(1, Int(floor(t / tau)))
    ψ_evolved = copy(ψ)

    # All Hamiltonian terms (system + bath + coupling) are in the interleaved gates,
    # matching the MPO DM+Trotter decomposition for consistency.
    for _ in 1:steps
        ψ_evolved = apply(gates, ψ_evolved; cutoff=cutoff, maxdim=Dmax, move_sites_back=true)
        normalize!(ψ_evolved)
    end

    return ψ_evolved
end

# ============================================================================
# Density Matrix + Trotter Evolution + MPO
# ============================================================================

function evolve_state(::HamiltonianParameters, sim_params::UnifiedSimulationParameters{DensityMatrix, TrotterEvolution},
                     ::TNBackend, gates, ρ, t::Float64, ::Vector{<:Index}; kwargs...)
    Dmax = max(sim_params.Dmax, 4 * sim_params.Dmax)
    cutoff = sim_params.cutoff / 10
    steps = max(1, Int(floor(t / sim_params.tau)))

    for _ in 1:steps
        ρ = apply(gates, ρ; apply_dag=true, cutoff=cutoff, maxdim=Dmax, move_sites_back=true)
    end

    return ρ
end
