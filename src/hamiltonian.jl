"""
    hamiltonian.jl

Main dispatcher that includes all Hamiltonian-related functionality.
This file serves as a convenient single import for all Hamiltonian operations.
"""

# parameter_types.jl already included by parent modules


# ============================================================================
# Utility Functions
# ============================================================================

# Model string to type mapping for dispatch-based construction
const MODEL_TYPE_MAP = Dict(
    "Ising" => IsingModel,
    "niIsing" => NiIsingModel,
    "Rydberg" => RydbergModel
)

const MODEL_PARAM_MAP = Dict(
    "Ising" => IsingParameters,
    "niIsing" => NiIsingParameters,
    "Rydberg" => RydbergParameters
)

"""
    get_model_from_string(problem::String) -> HamiltonianModel

Convert problem string to HamiltonianModel type.
"""
function get_model_from_string(problem::String)
    haskey(MODEL_TYPE_MAP, problem) || error("Unknown problem type: $problem")
    return MODEL_TYPE_MAP[problem]()
end

"""
    create_hamiltonian_params(problem::String, params...)

Create HamiltonianParameters from problem string and parameters.
"""
function create_hamiltonian_params(problem::String, params...)
    haskey(MODEL_PARAM_MAP, problem) || error("Unknown problem type: $problem")
    return MODEL_PARAM_MAP[problem](params...)
end