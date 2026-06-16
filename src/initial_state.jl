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
"""

# Generic fallback
function setup_initial_state(problem::CoolingProblem{B}, sim_params::UnifiedSimulationParameters{S,E},
                           init_type::String, theta::Float64) where {B<:CoolingBackend, S<:SimulationMethod, E<:EvolutionMethod}
    error("setup_initial_state not implemented for sim_method=$S, backend=$B")
end

# ============================================================================
# ED Backend Helper: Create theta-parameterized state vector
# ============================================================================

"""
    _theta_site_amplitudes(theta::Float64) -> Tuple{Float64, Float64}

Return the one-site amplitudes for the real product state
`cos(α)|0⟩ + sin(α)|1⟩`, with `α = (theta + 1/2)π/2`.
"""
function _theta_site_amplitudes(theta::Float64)
    α = (theta + 0.5) * π / 2
    return cos(α), sin(α)
end

"""
    _theta_product_mps(sites_sys, theta::Float64) -> MPS

Create the tensor-network product state whose one-site amplitudes are given by
`_theta_site_amplitudes(theta)`.
"""
function _theta_product_mps(sites_sys::Vector{<:Index}, theta::Float64)
    amp0, amp1 = _theta_site_amplitudes(theta)
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

Create an ED state vector based on `init_type` and the theta product-state
parameter. For `init_type == "theta"`, the convention is
`theta = -0.5, 0, 0.5` giving `|0⟩`, `|+⟩`, and `|1⟩` on each site.
"""
function create_theta_state_ed(N::Int, init_type::String, theta::Float64)::EDStateVector
    if init_type == "identity"
        # Equal superposition state (uniform distribution)
        data = ComplexF64.(ones(2^N) / sqrt(2^N))
        return EDStateVector(data, N)
    end

    if init_type != "theta"
        # Default product state - all zeros |00...0⟩
        return zero_state_ed(N)
    end

    # Code convention: theta=-0.5,0,0.5 gives |0>, |+>, |1>, respectively.
    data = ones(ComplexF64, 2^N)
    amp0, amp1 = _theta_site_amplitudes(theta)

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
    ϕ₀ = problem.ϕ₀
    sites_sys = siteinds(ϕ₀)

    if init_type == "identity"
        ψ_s = randomMPS(sites_sys, linkdims=1)
        normalize!(ψ_s)
    elseif init_type == "theta"
        ψ_s = _theta_product_mps(sites_sys, theta)
    else
        ψ_s = MPS(sites_sys, "Up")
    end
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ψ_s)
end

# Monte Carlo + ED
function setup_initial_state(problem::CoolingProblem{EDBackend}, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
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
