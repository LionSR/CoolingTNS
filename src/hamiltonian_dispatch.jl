"""
    hamiltonian_dispatch.jl

Main dispatcher that includes all Hamiltonian-related functionality.
This file serves as a convenient single import for all Hamiltonian operations.
"""

# Include parameter types for backend definitions
include("parameter_types.jl")

# Include all modular dispatch files
include("system_hamiltonian_dispatch.jl")
include("system_bath_hamiltonian_dispatch.jl")
include("ground_state_dispatch.jl")
include("coupling_utils.jl")



# ============================================================================
# Utility Functions
# ============================================================================

"""
    get_model_from_string(problem::String) -> HamiltonianModel

Convert problem string to HamiltonianModel type.
"""
function get_model_from_string(problem::String)
    if problem == "Ising"
        return IsingModel()
    elseif problem == "niIsing"
        return NiIsingModel()
    elseif problem == "Rydberg"
        return RydbergModel()
    else
        error("Unknown problem type: $problem")
    end
end

"""
    create_hamiltonian_params(problem::String, params...)

Create HamiltonianParameters from problem string and parameters.
"""
function create_hamiltonian_params(problem::String, params...)
    if problem == "Ising"
        return IsingParameters(params...)
    elseif problem == "niIsing"
        return NiIsingParameters(params...)
    elseif problem == "Rydberg"
        return RydbergParameters(params...)
    else
        error("Unknown problem type: $problem")
    end
end