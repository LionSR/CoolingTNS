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
