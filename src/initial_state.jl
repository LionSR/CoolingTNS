"""
    initial_state.jl

Unified initial state setup using multiple dispatch on SimulationMethod and backend.
"""

using ITensors
using ITensorMPS
using LinearAlgebra

# ============================================================================
# Main Initial State Interface
# ============================================================================

"""
    setup_initial_state(problem::CoolingProblem, sim_params::UnifiedSimulationParameters, init_type::String, theta::Float64)

Direct dispatch implementation for initial state setup.

For `init_type == "theta"`, `theta` is the dimensionless code parameter
`theta_code`. The corresponding physical product-state angle is
`initial_product_angle(theta_code)`. For `init_type == "ground"`, the initial
state is the system ground state already stored in `problem`.
"""

# Generic fallback
function setup_initial_state(problem::CoolingProblem{B}, sim_params::UnifiedSimulationParameters{S,E},
                           init_type::String, theta::Float64) where {B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    error("setup_initial_state not implemented for sim_method=$S, backend=$B")
end

# ============================================================================
# Shared Initial-State Validation
# ============================================================================

"""
    _reject_identity_for_mcwf()

Throw the shared error for asking a wavefunction path to prepare the maximally
mixed state: it is a density matrix, not one Monte-Carlo wavefunction.
"""
function _reject_identity_for_mcwf()
    throw(ArgumentError(
        "init_type=\"identity\" denotes the maximally mixed density matrix " *
        "and is not a single MonteCarloWavefunction state. Use DensityMatrix() " *
        "or choose a pure initial state such as \"product\", \"theta\", or \"ground\"."
    ))
end

# ============================================================================
# ED Backend Helper: Create theta-parameterized state vector
# ============================================================================

"""
    initial_product_angle(theta_code::Real)

Return the physical per-site product-state angle `alpha` in
`cos(alpha)|0> + sin(alpha)|1>` for the code-level theta parameter.

The code convention is `theta_code = -1/2, 0, 1/2` for
`|0>`, `|+>`, and `|1>`, respectively.
"""
initial_product_angle(theta_code::Real) = (theta_code + 0.5) * π / 2

"""
    theta_code_from_initial_product_angle(alpha::Real)

Return the dimensionless code-level theta parameter used by
`setup_initial_state(..., "theta", theta_code)` for a physical product-state
per-site angle `alpha`.
"""
theta_code_from_initial_product_angle(alpha::Real) = 2 * alpha / π - 0.5

"""
    theta_site_amplitudes(theta_code::Real)

Return the single-site amplitudes `(amp0, amp1)` for the theta initial-state
convention.
"""
function theta_site_amplitudes(theta_code::Real)
    alpha = initial_product_angle(theta_code)
    return cos(alpha), sin(alpha)
end

"""
    _theta_product_mps(sites_sys, theta_code::Real) -> MPS

Create the tensor-network product state whose one-site amplitudes are given by
`theta_site_amplitudes(theta_code)`.
"""
function _theta_product_mps(sites_sys::Vector{<:Index}, theta_code::Real)
    amp0, amp1 = theta_site_amplitudes(theta_code)
    ψ = MPS(ComplexF64, sites_sys, "Up")

    for i in eachindex(sites_sys)
        T = ITensor(ComplexF64, sites_sys[i])
        T[sites_sys[i] => 1] = amp0
        T[sites_sys[i] => 2] = amp1

        for I in inds(ψ[i])
            if I != sites_sys[i]
                T *= ITensor(ComplexF64(1.0), I)
            end
        end

        ψ[i] = T
    end

    orthogonalize!(ψ, 1)
    return ψ
end

"""
    theta_product_state_ed(N::Int, theta_code::Real) -> EDStateVector

Create the ED product state whose one-site amplitudes are
`theta_site_amplitudes(theta_code)`; the exact-diagonalization twin of
[`_theta_product_mps`](@ref). The code convention is
`theta_code = -0.5, 0, 0.5` giving `|0>`, `|+>`, and `|1>` on each site.
"""
function theta_product_state_ed(N::Int, theta_code::Real)
    data = ones(ComplexF64, 2^N)
    amp0, amp1 = theta_site_amplitudes(theta_code)

    for idx in 0:(2^N - 1)
        amplitude = 1.0
        for i in 0:(N - 1)
            bit = (idx >> i) & 1
            amplitude *= (bit == 0) ? amp0 : amp1
        end
        data[idx + 1] = amplitude
    end
    return EDStateVector(data, N)
end

"""
    create_theta_state_ed(N::Int, init_type::String, theta::Float64) -> EDStateVector
    create_theta_state_ed(N::Int, kind::InitialStateKind, theta::Float64) -> EDStateVector

Create an ED state vector for an initial-state kind and the code-level theta
parameter. The string form resolves `init_type` through
[`initial_state_kind`](@ref); the kind methods carry the construction.

`IdentityInitialState` and `GroundInitialState` are rejected because this
constructor returns pure state vectors built from `N` and `theta` alone: the
identity initial state denotes the maximally mixed density matrix, and the
ground state needs a `CoolingProblem` reference.
"""
create_theta_state_ed(N::Int, init_type::String, theta::Float64)::EDStateVector =
    create_theta_state_ed(N, initial_state_kind(init_type), theta)

# Default product state - all zeros |00...0⟩
create_theta_state_ed(N::Int, ::ProductInitialState, ::Float64) = zero_state_ed(N)

create_theta_state_ed(N::Int, ::ThetaInitialState, theta::Float64) =
    theta_product_state_ed(N, theta)

create_theta_state_ed(::Int, ::IdentityInitialState, ::Float64) = throw(ArgumentError(
    "create_theta_state_ed constructs pure state vectors; " *
    "init_type=\"identity\" is a density matrix initial state."
))

create_theta_state_ed(::Int, ::GroundInitialState, ::Float64) = throw(ArgumentError(
    "create_theta_state_ed cannot construct init_type=\"ground\" without " *
    "a CoolingProblem ground-state reference. Use setup_initial_state instead."
))

# ============================================================================
# Monte Carlo Wavefunction States
# ============================================================================

"""
    initial_pure_state(kind::InitialStateKind, problem::CoolingProblem, theta::Float64)

Prepare the pure system state named by `kind` in the representation of
`problem`'s backend: an `MPS` for [`TNBackend`](@ref), an `EDStateVector` for
[`EDBackend`](@ref).  `theta` is the dimensionless code-level theta parameter
and is used only by `ThetaInitialState`.

`IdentityInitialState` has no pure representative and is rejected here, so the
Monte-Carlo entry point never has to test for it.
"""
initial_pure_state(::GroundInitialState, problem::CoolingProblem{TNBackend}, ::Float64) =
    deepcopy(problem.ϕ₀)

initial_pure_state(::ThetaInitialState, problem::CoolingProblem{TNBackend}, theta::Float64) =
    _theta_product_mps(siteinds(problem.ϕ₀), theta)

initial_pure_state(::ProductInitialState, problem::CoolingProblem{TNBackend}, ::Float64) =
    MPS(siteinds(problem.ϕ₀), "Up")

initial_pure_state(::IdentityInitialState, ::CoolingProblem{TNBackend}, ::Float64) =
    _reject_identity_for_mcwf()

initial_pure_state(::GroundInitialState, problem::CoolingProblem{EDBackend}, ::Float64) =
    EDStateVector(problem.ϕ₀.data, problem.ϕ₀.n_qubits)

initial_pure_state(kind::InitialStateKind, problem::CoolingProblem{EDBackend}, theta::Float64) =
    create_theta_state_ed(problem.extra.ham_params.N, kind, theta)

initial_pure_state(::IdentityInitialState, ::CoolingProblem{EDBackend}, ::Float64) =
    _reject_identity_for_mcwf()

"""
    setup_initial_state(problem, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction},
                        init_type::String, theta::Float64)

Monte-Carlo wavefunction initial state: the pure state selected by `init_type`,
tagged with the backend and method types it will be evolved with.
"""
function setup_initial_state(problem::CoolingProblem, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    ψ_s = initial_pure_state(initial_state_kind(init_type), problem, theta)
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ψ_s)
end

# ============================================================================
# Density Matrix States
# ============================================================================

"""
    initial_density_state(kind::InitialStateKind, problem::CoolingProblem, theta::Float64)

Prepare the initial density matrix named by `kind` in the representation of
`problem`'s backend: an `MPO` for [`TNBackend`](@ref), an `EDDensityMatrix` for
[`EDBackend`](@ref).  Every pure kind is promoted from
[`initial_pure_state`](@ref) as `|ψ><ψ|`; `IdentityInitialState` is the
maximally mixed state, which has no pure representative to promote.
"""
function initial_density_state(::IdentityInitialState, problem::CoolingProblem{TNBackend}, ::Float64)
    sites_sys = siteinds(problem.ϕ₀)
    return MPO(sites_sys, "Id") / (2.0^length(sites_sys))
end

function initial_density_state(kind::InitialStateKind, problem::CoolingProblem{TNBackend}, theta::Float64)
    ψ_s = initial_pure_state(kind, problem, theta)
    return outer(ψ_s', ψ_s)
end

initial_density_state(::IdentityInitialState, problem::CoolingProblem{EDBackend}, ::Float64) =
    maximally_mixed_ed(problem.extra.ham_params.N)

# The stored ground state is already an `EDStateVector`, so it is converted
# directly rather than round-tripped through `initial_pure_state`, whose
# `EDStateVector` constructor would renormalize it.
initial_density_state(::GroundInitialState, problem::CoolingProblem{EDBackend}, ::Float64) =
    state_to_density_ed(problem.ϕ₀)

initial_density_state(kind::InitialStateKind, problem::CoolingProblem{EDBackend}, theta::Float64) =
    state_to_density_ed(initial_pure_state(kind, problem, theta))

"""
    setup_initial_state(problem, sim_params::UnifiedSimulationParameters{DensityMatrix},
                        init_type::String, theta::Float64)

Density-matrix initial state: the density matrix selected by `init_type`,
tagged with the backend and method types it will be evolved with.
"""
function setup_initial_state(problem::CoolingProblem, sim_params::UnifiedSimulationParameters{DensityMatrix, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    ρ_s = initial_density_state(initial_state_kind(init_type), problem, theta)
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ρ_s)
end
