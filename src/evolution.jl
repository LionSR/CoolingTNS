"""
    evolution.jl

Time evolution methods using multiple dispatch on SimulationMethod, EvolutionMethod, and backend.
"""

using ITensors
using ITensorMPS

# ============================================================================
# Time Evolution  
# ============================================================================

"""
    evolve_state(ham_params::HamiltonianParameters, sim_params::UnifiedSimulationParameters, backend::CoolingBackend, H_total, ψ, t, sites; kwargs...)

Generic evolution interface using triple dispatch on model, simulation method, and backend.
"""
function evolve_state(ham_params::HamiltonianParameters, sim_params::UnifiedSimulationParameters, backend::CoolingBackend, H_total, ψ, t, sites::Union{Nothing, Vector{<:Index}}; kwargs...)
    error("evolve_state not implemented for model=$(typeof(ham_params.model)), sim_method=$(typeof(sim_params.sim_method)), evolution_method=$(typeof(sim_params.evolution_method)), backend=$(typeof(backend))")
end

# Monte Carlo + Continuous Evolution + Tensor Networks
function evolve_state(ham_params::HamiltonianParameters, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, ContinuousEvolution}, 
                     backend::TNBackend, H_total, ψ, t::Float64, sites::Vector{<:Index}; kwargs...)
    # Use TDVP for continuous evolution
    Dmax, cutoff, tau = sim_params.Dmax, sim_params.cutoff, sim_params.tau
    ψ_evolved = tdvp(H_total, -im * t, ψ; time_step=-1im * tau, reverse_step=false, normalize=true, maxdim=Dmax, cutoff=cutoff, outputlevel=0)
    normalize!(ψ_evolved)
    orthogonalize!(ψ_evolved, 2)
    return ψ_evolved
end

# Monte Carlo + Trotter Evolution + Tensor Networks  
function evolve_state(ham_params::HamiltonianParameters, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, TrotterEvolution}, 
                     backend::TNBackend, H_total, ψ, t::Float64, sites::Vector{<:Index}; gates=nothing, kwargs...)
    if gates === nothing
        error("Trotter evolution requires pre-computed gates")
    end
    
    Dmax, cutoff, tau = sim_params.Dmax, sim_params.cutoff, sim_params.tau
    steps = Int(t / tau)
    ψ_evolved = copy(ψ)
    
    # Create zero coupling Hamiltonian based on model type and backend
    H_sys_zero = construct_zero_coupling_hamiltonian(ham_params, backend, sites)
    
    for _ in 1:steps
        # Evolve with system-only Hamiltonian using TDVP
        ψ_evolved = tdvp(H_sys_zero, -im * tau, ψ_evolved; nsteps=1, reverse_step=false, normalize=true, maxdim=Dmax, cutoff=cutoff, outputlevel=0)
        
        # Apply the pre-computed gates
        ψ_evolved = apply(gates, ψ_evolved; cutoff=cutoff, maxdim=Dmax, move_sites_back=true)

        orthogonalize!(ψ_evolved, 2)
        normalize!(ψ_evolved)
    end
    
    return ψ_evolved
end

# Density Matrix + Trotter Evolution + MPO
function evolve_state(ham_params::HamiltonianParameters, sim_params::UnifiedSimulationParameters{DensityMatrix, TrotterEvolution}, 
                     backend::TNBackend, gates, ρ, t::Float64, sites::Vector{<:Index}; kwargs...)
    Dmax, cutoff, trotter_steps = sim_params.Dmax, sim_params.cutoff, sim_params.trotter_steps
    
    for _ in 1:trotter_steps
        ρ = apply(gates, ρ; apply_dag=true, cutoff=cutoff, maxdim=Dmax, move_sites_back=true)
    end
    
    return ρ
end