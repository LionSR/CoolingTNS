"""
    cooling_interface.jl

Unified interface for cooling simulations with clean dispatch architecture.
All functions contain substance implementations rather than empty wrappers.
"""

using ITensors
using Yao
using KrylovKit
using LinearAlgebra
include("parameter_types.jl")
include("hamiltonian_dispatch.jl")
include("trotter_dispatch.jl")
include("evolution_dispatch.jl")
include("initial_state_dispatch.jl")
include("setup_system_dispatch.jl")

# Import ED functions
include("cooling_functions_ed.jl")

# Default simulation methods for backends (can be overridden)
default_simulation_method(::EDBackend) = DensityMatrix()  # Can be Monte Carlo or Density Matrix
default_simulation_method(::TNBackend) = MonteCarloWavefunction()  # Most common for TN

# Default evolution methods for backends (can be overridden)
default_evolution_method(::EDBackend) = ContinuousEvolution()  # Matrix exponentiation
default_evolution_method(::TNBackend) = ContinuousEvolution()  # TDVP is default for TN

# Container for problem setup results
struct CoolingProblem{B<:CoolingBackend}
    backend::B
    H_sys::Any  # System Hamiltonian
    H_sys_bath::Any  # Full system+bath Hamiltonian (unified naming)
    ϕ₀::Any     # Ground state
    e₀::Float64 # Ground state energy
    sites::Any  # Site indices (for tensor network methods)
    extra::NamedTuple  # Backend-specific extras (gates, etc.)
end

# Container for quantum states  
struct QuantumState{B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    backend::B
    sim_method::S
    evolution_method::E
    state::Any  # The actual state (MPS, MPO, density matrix, etc.)
    sites::Any  # Site indices (if applicable)
end

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

# ============================================================================
# Initial State Setup with Substance Implementations
# ============================================================================

# ED Backend - Direct substance implementation
function setup_initial_state(problem::CoolingProblem{EDBackend}, sim_params::UnifiedSimulationParameters, init_type::String="product", theta::Float64=0.0)
    # Extract N from Hamiltonian size - for ED, assume it's 2^N dimensional  
    N = Int(log2(size(mat(problem.H_sys), 1))) ÷ 2  # Total qubits ÷ 2 for system qubits
    # TODO: this way of extracting N is very sus, we must have N somewhere!! if not we need to have it in one of the struct.
    state = setup_initial_state(sim_params, problem.backend, N; init_type=init_type, theta=theta)
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, state, N)
end

# TN Backend - Direct substance implementation
function setup_initial_state(problem::CoolingProblem{TNBackend}, sim_params::UnifiedSimulationParameters, init_type::String="product", theta::Float64=0.0)
    state = setup_initial_state(sim_params, problem.backend, problem.sites; init_type=init_type, theta=theta)
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, state, problem.sites)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    get_backend(method::String) -> CoolingBackend

Convert string method name to backend type.
"""
function get_backend(method::String)
    if method == "ED"
        return EDBackend()
    elseif method in ["MPS", "MPO", "TrotterMPS", "TN"]
        return TNBackend()
    else
        error("Unknown method: $method. Use 'ED' for exact diagonalization or 'TN'/'MPS'/'MPO'/'TrotterMPS' for tensor networks")
    end
end

"""
    create_sim_params(backend::CoolingBackend; sim_method=nothing, evolution_method=nothing, kwargs...)

Create UnifiedSimulationParameters with intelligent defaults based on backend.
"""
function create_sim_params(backend::CoolingBackend; 
                          sim_method=nothing, 
                          evolution_method=nothing, 
                          kwargs...)
    # Use defaults if not specified
    sim_method = isnothing(sim_method) ? default_simulation_method(backend) : sim_method
    evolution_method = isnothing(evolution_method) ? default_evolution_method(backend) : evolution_method
    
    # Use direct constructor with dispatch
    return UnifiedSimulationParameters(sim_method, evolution_method; kwargs...)
end

# Export everything
export CoolingBackend, EDBackend, TNBackend
export SimulationMethod, DensityMatrix, MonteCarloWavefunction
export EvolutionMethod, ContinuousEvolution, TrotterEvolution
export UnifiedSimulationParameters
export CoolingProblem, QuantumState
export setup_problem, setup_initial_state
export get_backend, create_sim_params