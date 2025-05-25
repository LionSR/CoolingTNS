"""
    initial_state_dispatch.jl

Unified initial state setup using multiple dispatch on SimulationMethod and backend.
"""

using ITensors
using ITensorMPS
using Yao
include("parameter_types.jl")

# ============================================================================
# Generic Initial State Interface
# ============================================================================

"""
    setup_initial_state(sim_params::UnifiedSimulationParameters, backend::CoolingBackend, sites_or_N; init_type="product", theta=0.0)

Generic interface for initial state setup using double dispatch:
- SimulationMethod: DensityMatrix vs MonteCarloWavefunction  
- Backend: EDBackend vs TNBackend
"""
function setup_initial_state(sim_params::UnifiedSimulationParameters, backend::CoolingBackend, sites_or_N; init_type="product", theta=0.0)
    error("setup_initial_state not implemented for sim_method=$(typeof(sim_params.sim_method)), backend=$(typeof(backend))")
end

# ============================================================================
# Monte Carlo Wavefunction States
# ============================================================================

# Monte Carlo + TN
function setup_initial_state(sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E}, 
                            backend::TNBackend, sites; init_type="product", theta=0.0) where E<:EvolutionMethod
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
            # General theta not implemented for MPS yet
            @warn "General theta states not implemented for MPS, using default alternating"
            ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        end
    else
        # Default product state
        ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
    end
    return ψ_s
end


# Monte Carlo + Any + ED
function setup_initial_state(sim_params::UnifiedSimulationParameters{MonteCarloWavefunction, E}, 
                            backend::EDBackend, N::Int; init_type="product", theta=0.0) where E<:EvolutionMethod
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
    
    return reg
end

# ============================================================================
# Density Matrix States
# ============================================================================

# Density Matrix + TN
function setup_initial_state(sim_params::UnifiedSimulationParameters{DensityMatrix, E}, 
                            backend::TNBackend, sites; init_type="identity", theta=0.0) where E<:EvolutionMethod
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
            # General theta not implemented for MPS yet
            @warn "General theta states not implemented for MPS, using default alternating"
            ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        end
        ρ_s = outer(ψ_s', ψ_s)
    else
        # Product state - create from MPS outer product
        ψ_s = MPS(sites_sys, [isodd(n) ? "Up" : "Dn" for n in 1:N])
        ρ_s = outer(ψ_s', ψ_s)
    end
    return ρ_s
end



# Density Matrix + Any + ED  
function setup_initial_state(sim_params::UnifiedSimulationParameters{DensityMatrix, E}, 
                            backend::EDBackend, N::Int; init_type="identity", theta=0.0) where E<:EvolutionMethod
    total_qubits = 2N
    if init_type == "identity"
        # Maximally mixed state: ρ = I/2^N
        return DensityMatrix(uniform_state(total_qubits))
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
            ψ = product_state(bit_str(pattern))
        end
        return DensityMatrix(ψ)
    end
end

