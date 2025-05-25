"""
    cooling_types.jl

Core types for cooling simulations.
"""

# Container for problem setup results
struct CoolingProblem{B<:CoolingBackend}
    backend::B
    H_sys::Any  # System Hamiltonian
    H_sys_bath::Any  # Full system+bath Hamiltonian (unified naming)
    ϕ₀::Any     # Ground state
    e₀::Float64 # Ground state energy
    sites::Union{Nothing, Vector{<:Index}}  # Site indices (for tensor network methods)
    extra::NamedTuple  # Backend-specific extras (gates, etc.)
end

# Container for quantum states  
struct QuantumState{B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    backend::B
    sim_method::S
    evolution_method::E
    state::Any  # The actual state (MPS, MPO, density matrix, etc.)
end