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

"""
    coupling_operator_terms(coupling::String) -> Tuple

Expand a two-character coupling label into the local Hamiltonian terms used by
all backends.

Identical labels represent a single product operator, for example `"XX"` gives
``X_S X_B``. Mixed labels represent the symmetric Hermitian convention, for
example `"XY"` gives ``X_S Y_B + Y_S X_B``.
"""
function coupling_operator_terms(coupling::String)
    op1, op2 = parse_coupling(coupling)
    op1 == op2 && return ((op1, op2),)
    return ((op1, op2), (op2, op1))
end
