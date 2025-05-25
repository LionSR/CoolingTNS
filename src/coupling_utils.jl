"""
    coupling_utils.jl

Utilities for parsing and handling system-bath coupling operators.
"""

"""
    parse_coupling(coupling::String) -> (String, String)

Parse coupling string (e.g., "XX", "YZ") into individual operator strings.
"""
function parse_coupling(coupling::String)
    if length(coupling) != 2
        error("Coupling must be a two-character string like 'XX', 'YZ', etc.")
    end
    return string(coupling[1]), string(coupling[2])
end

"""
    get_pauli_operators(backend::CoolingBackend)

Get appropriate Pauli operator representations for the given backend.
"""
function get_pauli_operators(::TNBackend)
    # For ITensors, operators are strings
    return Dict("X" => "X", "Y" => "Y", "Z" => "Z")
end

function get_pauli_operators(::EDBackend)
    # For Yao, operators are matrix objects
    return Dict("X" => X, "Y" => Y, "Z" => Z)
end

"""
    validate_coupling(coupling::String)

Validate that coupling string contains only valid Pauli operators.
"""
function validate_coupling(coupling::String)
    valid_ops = ["X", "Y", "Z"]
    op1, op2 = parse_coupling(coupling)
    
    if !(op1 in valid_ops) || !(op2 in valid_ops)
        error("Invalid coupling operators. Must be combinations of X, Y, Z. Got: $coupling")
    end
    
    return true
end