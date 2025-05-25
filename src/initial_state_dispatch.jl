"""
    initial_state_dispatch.jl

Unified initial state setup using multiple dispatch on SimulationMethod and backend.
"""

using ITensors
using ITensorMPS
using Yao

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
    sites = problem.sites
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]

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
    total_qubits = 2N
    N_sys = total_qubits ÷ 2  # System qubits
    
    if init_type == "identity"
        # Create equal superposition state (maximally mixed when traced)
        # TODO: this is wrong we should just sample any of the superpositions instead.
        reg = uniform_state(total_qubits)
    elseif init_type == "theta"
        # Create theta-parameterized product state
        theta_rad = theta * π
        if abs(theta + 0.5) < 1e-10  # All down
            reg = zero_state(total_qubits)
        elseif abs(theta - 0.5) < 1e-10  # All up  
            reg = product_state(bit"1"^total_qubits)
        elseif abs(theta) < 1e-10  # X+ state
            reg = uniform_state(total_qubits)  # Simplified for now
        else
            # General theta state: |θ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            angles = fill(theta_rad, N_sys)
            # Apply rotation to system qubits only
            reg = zero_state(total_qubits)
            for i in 1:N_sys
                sys_qubit = 2*i - 1  # System qubits at odd positions
                reg |> put(sys_qubit => Ry(angles[i]))
            end
        end
    else
        # Default product state (alternating up/down for system, zero for bath)
        # Be default let us use all up
        pattern = ""
        for i in 1:total_qubits
            if isodd(i)  # System qubit
                pattern *= isodd((i+1)÷2) ? "1" : "0"  # Alternating pattern
            else  # Bath qubit  
                pattern *= "0"  # Initialize bath in ground state
            end
        end
        reg = product_state(bit_str(pattern))
    end
    
    return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, reg)
end

# ============================================================================
# Density Matrix States
# ============================================================================

# Density Matrix + TN
function setup_initial_state(problem::CoolingProblem{TNBackend}, sim_params::UnifiedSimulationParameters{DensityMatrix, E},
                           init_type::String, theta::Float64) where E<:EvolutionMethod
    sites = problem.sites
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]
    
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
    total_qubits = 2N
    if init_type == "identity"
        # Maximally mixed state: ρ = I/2^N
        return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, 
                           completely_mixed_state(total_qubits))
    else
        # Create pure state first, then convert to density matrix
        N_sys = total_qubits ÷ 2  # System qubits
        
        if init_type == "theta"
            # Create theta-parameterized product state
            theta_rad = theta * π
            if abs(theta + 0.5) < 1e-10  # All down
                ψ = zero_state(total_qubits)
            elseif abs(theta - 0.5) < 1e-10  # All up  
                ψ = product_state(bit"1"^total_qubits)
            elseif abs(theta) < 1e-10  # X+ state
                ψ = uniform_state(total_qubits)  # Simplified for now
            else
                # General theta state: |θ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                angles = fill(theta_rad, N_sys)
                # Apply rotation to system qubits only
                ψ = zero_state(total_qubits)
                for i in 1:N_sys
                    sys_qubit = 2*i - 1  # System qubits at odd positions
                    ψ |> put(sys_qubit => Ry(angles[i]))
                end
            end
        else
            # Default product state (alternating up/down for system, zero for bath)
            pattern = ""
            for i in 1:total_qubits
                if isodd(i)  # System qubit
                    pattern *= isodd((i+1)÷2) ? "1" : "0"  # Alternating pattern
                else  # Bath qubit  
                    pattern *= "0"  # Initialize bath in ground state
                end
            end
            # Convert pattern string to bit configuration
            config = parse(Int, pattern, base=2)
            ψ = product_state(ComplexF64, config, N_total)
        end
        return QuantumState(problem.backend, sim_params.sim_method, sim_params.evolution_method, 
                           DensityMatrix(ψ))
    end
end

