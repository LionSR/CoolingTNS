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
    get_bath_operator(coupling::String) -> String

Choose the bath Hamiltonian Pauli operator for a system-bath coupling label.

The bath field must not commute with the Pauli operator acting on the bath leg
of the coupling. With the convention `"AB" = A_S B_B`, this helper inspects
only `B`. We choose `Z` for bath-side `X` or `Y` couplings and `X` for
bath-side `Z` couplings.
"""
function get_bath_operator(coupling::String)
    _, bath_coupling_op = parse_coupling(coupling)
    return bath_coupling_op == "Z" ? "X" : "Z"
end

"""
    bath_ground_state_amplitudes(coupling::String) -> (String, Vector{ComplexF64})

Return the one-qubit bath ground state selected by `get_bath_operator`.

For positive detuning, the bath ground state is the eigenvalue `-1` state of the
bath Hamiltonian Pauli operator. Thus a `Z` bath field gives `|Dn⟩`, while an
`X` bath field gives `|X-⟩ = (|Up⟩ - |Dn⟩)/√2`.
"""
function bath_ground_state_amplitudes(coupling::String)
    bath_op = get_bath_operator(coupling)
    if bath_op == "X"
        return "X-", ComplexF64[1 / sqrt(2), -1 / sqrt(2)]
    elseif bath_op == "Z"
        return "Dn", ComplexF64[0, 1]
    end
    error("Unsupported bath Hamiltonian operator: $bath_op")
end
