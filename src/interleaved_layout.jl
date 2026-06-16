"""
    interleaved_layout.jl

Central convention for the interleaved system-bath layout

    (s_1, b_1, s_2, b_2, ..., s_N, b_N).

All functions here use one-indexed site labels, matching Julia arrays,
ITensors site lists, and the ED Pauli helper interface.
"""

_checked_nonnegative_N(N::Integer) =
    N >= 0 ? Int(N) : throw(ArgumentError("N must be nonnegative, got $N"))

_checked_positive_site(i::Integer) =
    i >= 1 ? Int(i) : throw(ArgumentError("site index must be positive, got $i"))

"""Total number of sites in an interleaved system+bath chain."""
interleaved_total_sites(N::Integer) = 2 * _checked_nonnegative_N(N)

"""Position of system spin `i` in the interleaved chain."""
interleaved_system_site(i::Integer) = 2 * _checked_positive_site(i) - 1

"""Position of bath spin `i` in the interleaved chain."""
interleaved_bath_site(i::Integer) = 2 * _checked_positive_site(i)

"""All system-site positions for an `N`-spin interleaved chain."""
function interleaved_system_sites(N::Integer)
    N_int = _checked_nonnegative_N(N)
    return [interleaved_system_site(i) for i in 1:N_int]
end

"""All bath-site positions for an `N`-spin interleaved chain."""
function interleaved_bath_sites(N::Integer)
    N_int = _checked_nonnegative_N(N)
    return [interleaved_bath_site(i) for i in 1:N_int]
end

"""System entries from a vector ordered as `(s_1,b_1,...,s_N,b_N)`."""
interleaved_system_indices(sites::AbstractVector, N::Integer) =
    sites[interleaved_system_sites(N)]

"""Bath entries from a vector ordered as `(s_1,b_1,...,s_N,b_N)`."""
interleaved_bath_indices(sites::AbstractVector, N::Integer) =
    sites[interleaved_bath_sites(N)]

"""Zero-indexed ED basis bit position corresponding to a one-indexed site."""
interleaved_bit_position(site::Integer) = _checked_positive_site(site) - 1

"""Zero-indexed ED basis bit position of system spin `i`."""
interleaved_system_bit(i::Integer) = interleaved_bit_position(interleaved_system_site(i))

"""Zero-indexed ED basis bit position of bath spin `i`."""
interleaved_bath_bit(i::Integer) = interleaved_bit_position(interleaved_bath_site(i))

"""All zero-indexed ED basis bit positions for system spins."""
function interleaved_system_bits(N::Integer)
    N_int = _checked_nonnegative_N(N)
    return [interleaved_system_bit(i) for i in 1:N_int]
end

"""All zero-indexed ED basis bit positions for bath spins."""
function interleaved_bath_bits(N::Integer)
    N_int = _checked_nonnegative_N(N)
    return [interleaved_bath_bit(i) for i in 1:N_int]
end

"""
    interleaved_basis_state(system_state, bath_state, N) -> Int

Map integer basis labels for an `N`-spin system and an `N`-spin bath into the
integer basis label of the interleaved ED Hilbert space.
"""
function interleaved_basis_state(system_state::Integer, bath_state::Integer, N::Integer)
    N_int = _checked_nonnegative_N(N)
    max_state = 1 << N_int
    0 <= system_state < max_state ||
        throw(ArgumentError("system_state must be in 0:$(max_state - 1), got $system_state"))
    0 <= bath_state < max_state ||
        throw(ArgumentError("bath_state must be in 0:$(max_state - 1), got $bath_state"))

    full_state = 0
    for i in 1:N_int
        source_bit = i - 1
        full_state |= ((system_state >> source_bit) & 1) << interleaved_system_bit(i)
        full_state |= ((bath_state >> source_bit) & 1) << interleaved_bath_bit(i)
    end
    return full_state
end

"""Map a system basis label into the interleaved ED Hilbert space with bath state zero."""
interleaved_system_basis_state(system_state::Integer, N::Integer) =
    interleaved_basis_state(system_state, 0, N)
