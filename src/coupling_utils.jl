"""
    coupling_utils.jl

Utilities for parsing and handling system-bath coupling operators.
"""

const _VALID_COUPLING_OPS = ('X', 'Y', 'Z')

"""
    parse_coupling(coupling::String) -> (String, String)

Parse a two-character coupling string (e.g. `"XX"`, `"YZ"`) into individual
operator labels.

Throws an `ArgumentError` if `coupling` is not exactly two characters long or
contains characters other than `X`, `Y`, or `Z`.
"""
function parse_coupling(coupling::String)
    if length(coupling) != 2
        throw(ArgumentError(
            "Coupling must be a two-character string like \"XX\" or \"YZ\"; got \"$coupling\"."
        ))
    end

    op1 = coupling[1]
    op2 = coupling[2]

    if !(op1 in _VALID_COUPLING_OPS) || !(op2 in _VALID_COUPLING_OPS)
        throw(ArgumentError("Coupling operators must be X, Y, or Z; got \"$coupling\"."))
    end

    return string(op1), string(op2)
end
