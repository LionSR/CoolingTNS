"""
    cooling_interface.jl

Defines abstract types and interfaces for cooling simulations,
providing a unified API across different backends (ED, MPS, MPO, TrotterMPS).
"""

using ITensors
using ITensorMPS

# Abstract types for simulation backends
abstract type CoolingBackend end
struct EDBackend <: CoolingBackend end
struct MPSBackend <: CoolingBackend end
struct MPOBackend <: CoolingBackend end
struct TrotterMPSBackend <: CoolingBackend end

# Abstract types for simulation methods
abstract type SimulationMethod end
struct DensityMatrix <: SimulationMethod end
struct MonteCarloWavefunction <: SimulationMethod end

# Map backends to their natural simulation methods
simulation_method(::EDBackend) = DensityMatrix()  # Can be overridden
simulation_method(::MPSBackend) = MonteCarloWavefunction()
simulation_method(::MPOBackend) = DensityMatrix()
simulation_method(::TrotterMPSBackend) = MonteCarloWavefunction()

# Container for problem setup results
struct CoolingProblem{B<:CoolingBackend}
    backend::B
    H_sys::Any  # System Hamiltonian
    H_full::Any  # Full system+bath Hamiltonian or evolution operator
    ϕ₀::Any     # Ground state
    e₀::Float64 # Ground state energy
    sites::Any  # Site indices (for tensor network methods)
    extra::Dict{String,<:Any}  # Backend-specific extras (gates, etc.)
end

# Container for quantum states
struct QuantumState{B<:CoolingBackend,M<:SimulationMethod}
    backend::B
    method::M
    state::Any  # The actual state (MPS, MPO, density matrix, etc.)
    sites::Any  # Site indices (if applicable)
end

"""
    setup_problem(backend::CoolingBackend, N, problem, ham_params, coupling_params, sim_params)

Unified interface for problem setup across all backends.
"""
function setup_problem(backend::CoolingBackend, N, problem, ham_params, coupling_params, sim_params)
    error("setup_problem not implemented for backend type $(typeof(backend))")
end

# Specific implementations for each backend
function setup_problem(backend::EDBackend, N, problem, ham_params, coupling_params, sim_params)
    H_sys, H_full, ϕ₀, e₀ = setup_problem_ed(N, problem, ham_params, coupling_params, sim_params)
    return CoolingProblem(backend, H_sys, H_full, ϕ₀, e₀, nothing, Dict{String,Any}())
end

function setup_problem(backend::MPSBackend, N, problem, ham_params, coupling_params, sim_params)
    sites, H_sys, ϕ₀, e₀, H_sys_bath = setup_problem_mps(N, problem, ham_params, coupling_params, sim_params)
    return CoolingProblem(backend, H_sys, H_sys_bath, ϕ₀, e₀, sites, Dict("H_sys_bath" => H_sys_bath))
end

function setup_problem(backend::MPOBackend, N, problem, ham_params, coupling_params, sim_params)
    sites, H_sys, ϕ₀, e₀, gates = setup_problem_mpo(N, problem, ham_params, coupling_params, sim_params)
    return CoolingProblem(backend, H_sys, gates, ϕ₀, e₀, sites, Dict("gates" => gates))
end

function setup_problem(backend::TrotterMPSBackend, N, problem, ham_params, coupling_params, sim_params)
    sites, H_sys, H_total, ϕ₀, e₀, gates = setup_problem_trotter_mps(N, problem, ham_params, coupling_params, sim_params)
    return CoolingProblem(backend, H_sys, H_total, ϕ₀, e₀, sites, Dict("gates" => gates, "H_total" => H_total))
end

"""
    setup_initial_state(problem::CoolingProblem, init_type, theta; method=nothing)

Unified interface for initial state setup.
"""
function setup_initial_state(problem::CoolingProblem, init_type::String, theta::Float64=0.0; method::Union{Nothing,SimulationMethod}=nothing)
    backend = problem.backend
    sim_method = isnothing(method) ? simulation_method(backend) : method
    
    if backend isa EDBackend
        N = problem.sites === nothing ? 0 : length(problem.sites) ÷ 2
        # For ED, we need to determine N from H_sys
        if N == 0
            # Extract N from Hamiltonian size (number of qubits)
            # H_sys is a Yao block, get number of qubits
            N = nqubits(problem.H_sys)
        end
        state = setup_init_state_ed(2N; init_type=init_type, theta=theta, method=sim_method)
        return QuantumState(backend, sim_method, state, nothing)
    elseif backend isa MPSBackend || backend isa TrotterMPSBackend
        state = setup_init_state_mps(problem.sites; init_type=init_type, theta=theta)
        return QuantumState(backend, sim_method, state, problem.sites)
    elseif backend isa MPOBackend
        state = setup_init_state_mpo(problem.sites; init_type=init_type, theta=theta)
        return QuantumState(backend, sim_method, state, problem.sites)
    else
        error("setup_initial_state not implemented for backend type $(typeof(backend))")
    end
end

"""
    run_cooling(problem::CoolingProblem, initial_state::QuantumState, coupling_params, sim_params, ham_params=nothing)

Unified interface for running cooling simulations.
"""
function run_cooling(problem::CoolingProblem, initial_state::QuantumState, coupling_params, sim_params, ham_params=nothing)
    backend = problem.backend
    
    if backend isa EDBackend
        # ED uses the state wrapper directly
        return run_cooling_ed(
            problem.H_sys,
            problem.H_full,
            problem.ϕ₀,
            initial_state.state,
            coupling_params,
            sim_params
        )
    elseif backend isa MPSBackend
        return run_cooling_mps(
            problem.sites,
            problem.H_sys,
            problem.ϕ₀,
            problem.extra["H_sys_bath"],
            initial_state.state,
            coupling_params,
            sim_params
        )
    elseif backend isa MPOBackend
        return run_cooling_mpo(
            problem.sites,
            problem.H_sys,
            problem.ϕ₀,
            problem.extra["gates"],
            initial_state.state,
            coupling_params,
            sim_params
        )
    elseif backend isa TrotterMPSBackend
        return run_cooling_trotter_mps(
            problem.sites,
            problem.H_sys,
            problem.extra["H_total"],
            problem.ϕ₀,
            problem.extra["gates"],
            initial_state.state,
            coupling_params,
            sim_params,
            ham_params
        )
    else
        error("run_cooling not implemented for backend type $(typeof(backend))")
    end
end

"""
    get_backend(method::String) -> CoolingBackend

Convert string method name to backend type.
"""
function get_backend(method::String)
    if method == "ED"
        return EDBackend()
    elseif method == "MPS"
        return MPSBackend()
    elseif method == "MPO"
        return MPOBackend()
    elseif method == "TrotterMPS"
        return TrotterMPSBackend()
    else
        error("Unknown method: $method")
    end
end

"""
    get_simulation_method(backend::CoolingBackend, ed_method::String="") -> SimulationMethod

Determine simulation method based on backend and optional ED method specification.
"""
function get_simulation_method(backend::CoolingBackend, ed_method::String="")
    if backend isa EDBackend && !isempty(ed_method)
        if ed_method == "density_matrix"
            return DensityMatrix()
        elseif ed_method == "monte_carlo"
            return MonteCarloWavefunction()
        else
            error("Unknown ED method: $ed_method")
        end
    else
        return simulation_method(backend)
    end
end

# Export everything
export CoolingBackend, EDBackend, MPSBackend, MPOBackend, TrotterMPSBackend
export SimulationMethod, DensityMatrix, MonteCarloWavefunction
export CoolingProblem, QuantumState
export setup_problem, setup_initial_state, run_cooling
export get_backend, get_simulation_method, simulation_method