"""
    initial_state.jl

Unified initial state setup using multiple dispatch on SimulationMethod and backend.
"""

using ITensors
using ITensorMPS
using Yao
using LinearAlgebra

# ============================================================================
# Helper Functions
# ============================================================================

"""Create completely mixed state (maximally mixed density matrix)"""
function completely_mixed_state(N::Int)
    # ρ = I / 2^N
    dim = 2^N
    return Matrix(I, dim, dim) / dim
end

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
# Monte Carlo Wavefunction States
# ============================================================================

# Monte Carlo + TN
function setup_initial_state(problem::CoolingProblem{TNBackend}, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    # Get sites from the ground state MPS
    ϕ₀ = problem.ϕ₀
    sites_sys = siteinds(ϕ₀)  # ϕ₀ already has only system sites
    N = length(sites_sys)

    if init_type == "identity"
        # Create maximally mixed state (equal superposition)
        ψ_s = randomMPS(sites_sys, linkdims=1)
        normalize!(ψ_s)
    elseif init_type == "theta"
        # Create state based on theta angle (in units of pi)
        theta_rad = theta * π
        if abs(theta + 0.5) < 1e-10  # All down
            ψ_s = MPS(sites_sys, "Dn")
        elseif abs(theta - 0.5) < 1e-10  # All up
            ψ_s = MPS(sites_sys, "Up")
        elseif abs(theta) < 1e-10  # X+ state
            ψ_s = MPS(sites_sys, "X+")
        else
            # General theta not implemented for tensor networks yet
            @warn "General theta states not implemented for tensor networks, using default alternating"
            ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        end
    else
        # Default product state
        ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
    end
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ψ_s)
end


# Monte Carlo + ED
function setup_initial_state(problem::CoolingProblem{EDBackend}, sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    
    N = problem.extra.ham_params.N
    N_sys = N  # System qubits
    
    if init_type == "identity"
        # Create equal superposition state (maximally mixed when traced)
        # For Monte Carlo, we only need system state
        reg = uniform_state(N_sys)
    elseif init_type == "theta"
        # Create theta-parameterized product state
        theta_rad = theta * π
        if abs(theta + 0.5) < 1e-10  # All down
            reg = zero_state(N_sys)
        elseif abs(theta - 0.5) < 1e-10  # All up  
            # All up state
            config = (1 << N_sys) - 1  # All bits set to 1
            reg = product_state(N_sys, config)
        elseif abs(theta) < 1e-10  # X+ state
            reg = uniform_state(N_sys)  # Simplified for now
        else
            # General theta state: |θ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            angles = fill(theta_rad, N_sys)
            reg = zero_state(N_sys)
            for i in 1:N_sys
                reg |> put(i => Ry(angles[i]))
            end
        end
    else
        # Default product state (alternating up/down for system)
        pattern = ""
        for i in 1:N_sys
            pattern *= isodd(i) ? "1" : "0"  # Alternating pattern
        end
        # Convert pattern string to bit configuration
        config = parse(Int, pattern, base=2)
        reg = product_state(N_sys, config)
    end
    
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, reg)
end

# ============================================================================
# Density Matrix States
# ============================================================================

# Density Matrix + TN
function setup_initial_state(problem::CoolingProblem{TNBackend}, sim_params::UnifiedSimulationParameters{DensityMatrix, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    # Get sites from ground state or H_sys
    ϕ₀ = problem.ϕ₀
    sites_sys = siteinds(ϕ₀)  # ϕ₀ already has only system sites
    N = length(sites_sys)
    
    if init_type == "identity"
        # Maximally mixed state (identity matrix)
        ρ_s = MPO(sites_sys, "Id")
        ρ_s = ρ_s ./ √2
    elseif init_type == "theta"
        # Create state based on theta angle (in units of pi)
        # First create MPS state with theta
        theta_rad = theta * π
        if abs(theta + 0.5) < 1e-10  # All down
            ψ_s = MPS(sites_sys, "Dn")
        elseif abs(theta - 0.5) < 1e-10  # All up
            ψ_s = MPS(sites_sys, "Up")
        elseif abs(theta) < 1e-10  # X+ state
            ψ_s = MPS(sites_sys, "X+")
        else
            # General theta not implemented for tensor networks yet
            @warn "General theta states not implemented for tensor networks, using default alternating"
            ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        end
        ρ_s = outer(ψ_s', ψ_s)
    else
        # Product state - create from MPS outer product
        ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        ρ_s = outer(ψ_s', ψ_s)
    end
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ρ_s)
end



# Density Matrix + ED  
function setup_initial_state(problem::CoolingProblem{EDBackend}, sim_params::UnifiedSimulationParameters{DensityMatrix, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    
    N = problem.extra.ham_params.N
    N_sys = N  # We only need system state for initial state
    if init_type == "identity"
        # Maximally mixed state: ρ = I/2^N
        return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, 
                           completely_mixed_state(N_sys))
    else
        # Create pure state first, then convert to density matrix
        
        if init_type == "theta"
            # Create theta-parameterized product state
            theta_rad = theta * π
            if abs(theta + 0.5) < 1e-10  # All down
                ψ = zero_state(N_sys)
            elseif abs(theta - 0.5) < 1e-10  # All up  
                # All up state
                config = (1 << N_sys) - 1  # All bits set to 1
                ψ = product_state(N_sys, config)
            elseif abs(theta) < 1e-10  # X+ state
                ψ = uniform_state(N_sys)  # Simplified for now
            else
                # General theta state: |θ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                angles = fill(theta_rad, N_sys)
                ψ = zero_state(N_sys)
                for i in 1:N_sys
                    ψ |> put(i => Ry(angles[i]))
                end
            end
        else
            # Default product state (alternating up/down for system, zero for bath)
            pattern = ""
            for i in 1:N_sys
                pattern *= isodd(i) ? "1" : "0"  # Alternating pattern
            end
            # Convert pattern string to bit configuration
            config = parse(Int, pattern, base=2)
            ψ = product_state(N_sys, config)
        end
        # Convert pure state to density matrix
        ρ = Matrix(ψ.state * ψ.state')
        return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ρ)
    end
end

