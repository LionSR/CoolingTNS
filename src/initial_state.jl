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
`initial_product_angle(theta_code)`.
"""

# Generic fallback
function setup_initial_state(problem::CoolingProblem{B}, sim_params::UnifiedSimulationParameters{S,E},
                           init_type::String, theta::Float64) where {B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    error("setup_initial_state not implemented for sim_method=$S, backend=$B")
end

# ============================================================================
# Shared Initial-State Validation
# ============================================================================

function _reject_identity_for_mcwf(init_type::String)
    if init_type == "identity"
        throw(ArgumentError(
            "init_type=\"identity\" denotes the maximally mixed density matrix " *
            "and is not a single MonteCarloWavefunction state. Use DensityMatrix() " *
            "or choose a pure initial state such as \"product\" or \"theta\"."
        ))
    end
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
    create_theta_state_ed(N::Int, init_type::String, theta::Float64) -> EDStateVector

Create an ED state vector based on `init_type` and the code-level theta
parameter. For `init_type == "theta"`, the code convention is
`theta = -0.5, 0, 0.5` giving `|0>`, `|+>`, and `|1>` on each site.
The value `init_type == "identity"` is rejected because this constructor
returns pure state vectors, while the identity initial state denotes the
maximally mixed density matrix.
"""
function create_theta_state_ed(N::Int, init_type::String, theta::Float64)::EDStateVector
    if init_type == "identity"
        throw(ArgumentError(
            "create_theta_state_ed constructs pure state vectors; " *
            "init_type=\"identity\" is a density matrix initial state."
        ))
    end

    if init_type != "theta"
        # Default product state - all zeros |00...0⟩
        return zero_state_ed(N)
    end

    # Code convention: theta=-0.5,0,0.5 gives |0>, |+>, |1>, respectively.
    data = ones(ComplexF64, 2^N)
    amp0, amp1 = theta_site_amplitudes(theta)

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

# ============================================================================
# Monte Carlo Wavefunction States
# ============================================================================

# Monte Carlo + TN
function setup_initial_state(problem::CoolingProblem{TNBackend}, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    _reject_identity_for_mcwf(init_type)

    ϕ₀ = problem.ϕ₀
    sites_sys = siteinds(ϕ₀)

    if init_type == "theta"
        ψ_s = _theta_product_mps(sites_sys, theta)
    else
        ψ_s = MPS(sites_sys, "Up")
    end
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ψ_s)
end

# Monte Carlo + ED
function setup_initial_state(problem::CoolingProblem{EDBackend}, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    _reject_identity_for_mcwf(init_type)

    N = problem.extra.ham_params.N
    state = create_theta_state_ed(N, init_type, theta)
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, state)
end

# ============================================================================
# Density Matrix States
# ============================================================================

# Density Matrix + TN
function setup_initial_state(problem::CoolingProblem{TNBackend}, sim_params::UnifiedSimulationParameters{DensityMatrix, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    ϕ₀ = problem.ϕ₀
    sites_sys = siteinds(ϕ₀)

    if init_type == "identity"
        ρ_s = MPO(sites_sys, "Id")
        ρ_s = ρ_s / (2.0^length(sites_sys))
    elseif init_type == "theta"
        ψ_s = _theta_product_mps(sites_sys, theta)
        ρ_s = outer(ψ_s', ψ_s)
    else
        ψ_s = MPS(sites_sys, "Up")
        ρ_s = outer(ψ_s', ψ_s)
    end
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ρ_s)
end

# Density Matrix + ED
function setup_initial_state(problem::CoolingProblem{EDBackend}, sim_params::UnifiedSimulationParameters{DensityMatrix, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    N = problem.extra.ham_params.N

    if init_type == "identity"
        ρ = maximally_mixed_ed(N)
    else
        ψ = create_theta_state_ed(N, init_type, theta)
        ρ = state_to_density_ed(ψ)
    end
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ρ)
end
