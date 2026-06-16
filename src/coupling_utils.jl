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

"""
    get_bath_operator(coupling::String) -> String

Return the Pauli operator used in the local bath Hamiltonian.

The bath field is chosen from the Pauli operators appearing on the bath leg of
`coupling_operator_terms(coupling)`. For a one-Pauli bath set we keep the
historical convention: bath-side `X` or `Y` uses a `Z` field, while bath-side
`Z` uses an `X` field. For a mixed symmetric coupling, the field is the unique
Pauli operator absent from the bath-side set. Thus `XY`/`YX` use `Z`,
`YZ`/`ZY` use `X`, and `XZ`/`ZX` use `Y`.
"""
function get_bath_operator(coupling::String)
    bath_labels = unique(last.(coupling_operator_terms(coupling)))

    if length(bath_labels) == 1
        return only(bath_labels) == "Z" ? "X" : "Z"
    end

    has_bath_x = "X" in bath_labels
    has_bath_y = "Y" in bath_labels

    !has_bath_x && return "X"
    !has_bath_y && return "Y"
    return "Z"
end

"""
    bath_ground_state_amplitudes(coupling::String) -> (String, Vector{ComplexF64})

Return the one-qubit bath ground state selected by `get_bath_operator`.

For positive detuning, the bath ground state is the eigenvalue `-1` state of the
bath Hamiltonian Pauli operator.
"""
function bath_ground_state_amplitudes(coupling::String)
    bath_op = get_bath_operator(coupling)
    if bath_op == "X"
        return "X-", ComplexF64[1 / sqrt(2), -1 / sqrt(2)]
    elseif bath_op == "Y"
        return "Y-", ComplexF64[1 / sqrt(2), -im / sqrt(2)]
    end

    return "Dn", ComplexF64[0, 1]
end
