"""
    cooling_types.jl

Core container types for cooling simulations.

These structs are intentionally lightweight: they primarily bundle together backend-specific
objects (Hamiltonians, states, cached gates, etc.) while keeping the *backend* and *method*
information in the type domain for multiple dispatch.
"""

"""Container for a fully-initialized cooling problem.

The concrete types of `H_sys`, `H_sys_bath`, and `ϕ₀` depend on the backend
(e.g. MPO/MPS for tensor networks vs. dense/sparse matrices for ED). We keep them as
type parameters rather than `Any` for type stability.
"""
struct CoolingProblem{B<:CoolingBackend, Hsys, HsysBath, Phi0, Extra<:NamedTuple}
    backend::B
    H_sys::Hsys                 # System Hamiltonian
    H_sys_bath::HsysBath        # Full system+bath Hamiltonian (or `nothing` if unused)
    ϕ₀::Phi0                    # Ground state
    e₀::Float64                 # Ground state energy
    extra::Extra                # Backend-specific extras (sites, cached gates, etc.)
end

"""Container for a quantum state with method+backend metadata.

`state` is backend/method dependent (MPS, MPO, EDStateVector, EDDensityMatrix, ...).
"""
struct QuantumState{B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod, StateT}
    backend::B
    sim_method::S
    evolution_method::E
    state::StateT
end
