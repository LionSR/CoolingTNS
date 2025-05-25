"""
    initial_state_ed_clean.jl

Initial state preparation for ED backend without Yao dependencies.
"""

include("ed_backend.jl")

# ============================================================================
# Initial State Setup for ED Backend
# ============================================================================

"""
    setup_initial_state(::EDBackend, init_state_type::String, N::Int, extra_params=nothing)

Setup initial state for ED backend based on type string.
"""
function setup_initial_state(::EDBackend, init_state_type::String, N::Int, extra_params=nothing)
    if init_state_type == "up"
        # All spins up: |111...⟩
        config = 2^N - 1
        return product_state_ed(N, config)
        
    elseif init_state_type == "down"
        # All spins down: |000...⟩
        return zero_state_ed(N)
        
    elseif init_state_type == "random"
        # Random state
        return random_state_ed(N)
        
    elseif init_state_type == "theta"
        # Parameterized state: tensor product of cos(θπ/2)|0⟩ + sin(θπ/2)|1⟩
        theta = extra_params !== nothing ? extra_params : 0.5
        
        # Build state as tensor product
        data = ones(Float64, 2^N)
        for idx in 0:(2^N-1)
            amplitude = 1.0
            for i in 0:(N-1)
                bit = (idx >> i) & 1
                if bit == 0
                    amplitude *= cos(theta * π / 2)
                else
                    amplitude *= sin(theta * π / 2)
                end
            end
            data[idx+1] = amplitude
        end
        return EDStateVector(data, N)
        
    elseif init_state_type == "identity"
        # Maximally mixed state (for density matrix simulations)
        return maximally_mixed_ed(N)
        
    else
        error("Unknown initial state type: $init_state_type")
    end
end

"""
    setup_system_initial_state(::EDBackend, ::DensityMatrix, init_state_type::String, N_sys::Int; kwargs...)

Setup initial density matrix for ED backend.
"""
function setup_system_initial_state(::EDBackend, ::DensityMatrix, init_state_type::String, N_sys::Int; kwargs...)
    if init_state_type == "identity"
        return maximally_mixed_ed(N_sys)
    else
        # Create pure state and convert to density matrix
        ψ = setup_initial_state(EDBackend(), init_state_type, N_sys, get(kwargs, :theta, nothing))
        return state_to_density_ed(ψ)
    end
end

"""
    setup_system_initial_state(::EDBackend, ::MonteCarloWavefunction, init_state_type::String, N_sys::Int; kwargs...)

Setup initial state vector for ED backend Monte Carlo.
"""
function setup_system_initial_state(::EDBackend, ::MonteCarloWavefunction, init_state_type::String, N_sys::Int; kwargs...)
    theta = get(kwargs, :theta, nothing)
    return setup_initial_state(EDBackend(), init_state_type, N_sys, theta)
end