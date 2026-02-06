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
    create_theta_state_ed(N::Int, init_type::String, theta::Float64) -> EDStateVector

Create an ED state vector based on init_type and theta parameter.
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

    # Theta-parameterized state
    theta_rad = theta * π

    # Special cases for efficiency
    if abs(theta + 0.5) < 1e-10  # All down (theta = -0.5)
        return zero_state_ed(N)
    elseif abs(theta - 0.5) < 1e-10  # All up (theta = 0.5)
        config = (1 << N) - 1  # All bits set to 1
        return product_state_ed(N, config)
    elseif abs(theta) < 1e-10  # X+ state (theta = 0)
        data = ComplexF64.(ones(2^N) / sqrt(2^N))
        return EDStateVector(data, N)
    end

    # General theta state: |θ⟩ = cos(θπ/2)|0⟩ + sin(θπ/2)|1⟩ for each qubit
    data = ones(ComplexF64, 2^N)
    cos_half = cos(theta_rad / 2)
    sin_half = sin(theta_rad / 2)

    for idx in 0:(2^N - 1)
        amplitude = 1.0
        for i in 0:(N - 1)
            bit = (idx >> i) & 1
            amplitude *= (bit == 0) ? cos_half : sin_half
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
    N = length(sites_sys)

    if init_type == "identity"
        ψ_s = randomMPS(sites_sys, linkdims=1)
        normalize!(ψ_s)
    elseif init_type == "theta"
        theta_rad = theta * π
        if abs(theta + 0.5) < 1e-10
            ψ_s = MPS(sites_sys, "Dn")
        elseif abs(theta - 0.5) < 1e-10
            ψ_s = MPS(sites_sys, "Up")
        elseif abs(theta) < 1e-10
            ψ_s = MPS(sites_sys, "X+")
        else
            @warn "General theta states not implemented for tensor networks, using default alternating"
            ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        end
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
    N = length(sites_sys)

    if init_type == "identity"
        ρ_s = MPO(sites_sys, "Id")
        ρ_s = ρ_s / (2.0^N)
    elseif init_type == "theta"
        theta_rad = theta * π
        if abs(theta + 0.5) < 1e-10
            ψ_s = MPS(sites_sys, "Dn")
        elseif abs(theta - 0.5) < 1e-10
            ψ_s = MPS(sites_sys, "Up")
        elseif abs(theta) < 1e-10
            ψ_s = MPS(sites_sys, "X+")
        else
            @warn "General theta states not implemented for tensor networks, using default alternating"
            ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        end
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
