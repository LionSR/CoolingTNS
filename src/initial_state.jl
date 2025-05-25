"""
    initial_state.jl

Unified initial state setup using multiple dispatch on SimulationMethod and backend.
"""

using ITensors
using ITensorMPS
using LinearAlgebra

# Include clean ED backend if available
if !@isdefined(EDStateVector)
    include("ed_backend.jl")
end

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
        # Create equal superposition state (uniform distribution)
        data = ones(Float64, 2^N_sys) / sqrt(2^N_sys)
        state = EDStateVector(data, N_sys)
    elseif init_type == "theta"
        # Create theta-parameterized product state
        theta_rad = theta * π
        if abs(theta + 0.5) < 1e-10  # All down
            state = zero_state_ed(N_sys)
        elseif abs(theta - 0.5) < 1e-10  # All up  
            # All up state
            config = (1 << N_sys) - 1  # All bits set to 1
            state = product_state_ed(N_sys, config)
        elseif abs(theta) < 1e-10  # X+ state (uniform superposition)
            data = ones(Float64, 2^N_sys) / sqrt(2^N_sys)
            state = EDStateVector(data, N_sys)
        else
            # General theta state: |θ⟩ = cos(θπ/2)|0⟩ + sin(θπ/2)|1⟩ for each qubit
            data = ones(Float64, 2^N_sys)
            for idx in 0:(2^N_sys-1)
                amplitude = 1.0
                for i in 0:(N_sys-1)
                    bit = (idx >> i) & 1
                    if bit == 0
                        amplitude *= cos(theta_rad / 2)
                    else
                        amplitude *= sin(theta_rad / 2)
                    end
                end
                data[idx+1] = amplitude
            end
            state = EDStateVector(data, N_sys)
        end
    else
        # Default product state (alternating up/down for system)
        config = 0
        for i in 0:(N_sys-1)
            if isodd(i+1)  # Julia is 1-indexed
                config |= (1 << i)
            end
        end
        state = product_state_ed(N_sys, config)
    end
    
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, state)
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
        ρ = maximally_mixed_ed(N_sys)
        return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ρ)
    else
        # Create pure state first using clean ED backend, then convert to density matrix
        if init_type == "theta"
            # Create theta-parameterized product state
            theta_rad = theta * π
            if abs(theta + 0.5) < 1e-10  # All down
                ψ = zero_state_ed(N_sys)
            elseif abs(theta - 0.5) < 1e-10  # All up  
                # All up state
                config = (1 << N_sys) - 1  # All bits set to 1
                ψ = product_state_ed(N_sys, config)
            elseif abs(theta) < 1e-10  # X+ state
                data = ones(Float64, 2^N_sys) / sqrt(2^N_sys)
                ψ = EDStateVector(data, N_sys)
            else
                # General theta state: |θ⟩ = cos(θπ/2)|0⟩ + sin(θπ/2)|1⟩ for each qubit
                data = ones(Float64, 2^N_sys)
                for idx in 0:(2^N_sys-1)
                    amplitude = 1.0
                    for i in 0:(N_sys-1)
                        bit = (idx >> i) & 1
                        if bit == 0
                            amplitude *= cos(theta_rad / 2)
                        else
                            amplitude *= sin(theta_rad / 2)
                        end
                    end
                    data[idx+1] = amplitude
                end
                ψ = EDStateVector(data, N_sys)
            end
        else
            # Default product state (alternating up/down)
            config = 0
            for i in 0:(N_sys-1)
                if isodd(i+1)  # Julia is 1-indexed
                    config |= (1 << i)
                end
            end
            ψ = product_state_ed(N_sys, config)
        end
        # Convert pure state to density matrix
        ρ = state_to_density_ed(ψ)
        return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, ρ)
    end
end

