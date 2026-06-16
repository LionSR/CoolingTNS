"""
Plotting-facing dispersion and k-space utilities for the transverse-field
Ising model.

The physical source of truth is `mode_analysis.jl`, which fixes the map from
the code Hamiltonian `J Σ σ_z σ_z + h Σ σ_x` to the notes' Bogoliubov
convention. The helpers in this file keep the older plotting API, but delegate
the actual energies and occupations to that canonical convention.
"""

"""
    generate_k_values(N::Int, bc::Symbol) -> Vector{Float64}

Generate a full fermionic momentum grid.

- Periodic fermions: `k = 2π n/N` for `n = 0, 1, ..., N-1`.
- Antiperiodic fermions: `k = π(2n+1)/N` for `n = 0, 1, ..., N-1`.

Simulation output files should normally use the saved `RESULT_K_VALUES`
dataset, since the fermionic boundary condition is fixed by both the spin
boundary condition and the parity sector.
"""
function generate_k_values(N::Int, bc::Symbol)::Vector{Float64}
    if bc == :periodic
        return [2pi * n / N for n in 0:N-1]
    elseif bc == :antiperiodic
        return [pi * (2n + 1) / N for n in 0:N-1]
    else
        error("Unsupported boundary condition: $bc")
    end
end

"""
    compute_energy_dispersion(k_values, J::Real, h::Real; N=length(k_values)) -> Vector{Float64}

Compute the positive code-unit quasiparticle energies on a momentum grid.

This is the plotting-facing wrapper around [`mode_energy_Jh`](@ref).  If
`k_values` is a full N-mode grid, the chain length is inferred from
`length(k_values)`; pass `N` explicitly for a nonstandard grid.
"""
function compute_energy_dispersion(k_values, J::Real, h::Real; N::Int=length(k_values))::Vector{Float64}
    return [mode_energy_Jh(_mode_index_from_momentum(k, N), J, h, N) for k in k_values]
end

"""
    compute_ground_state_occupation(k_values, J::Real, h::Real; N=length(k_values), spin_bc=nothing, gF=nothing) -> Vector{Float64}

Compute the ground-state Fourier occupation `⟨ã†_k ã_k⟩` on a momentum grid.

For a generic Bogoliubov vacuum mode this is `sin^2(phi_k)`, with `phi_k`
given by the canonical Bogoliubov angle. A result-file caller should pass the
spin boundary condition `spin_bc` and stored fermionic boundary sector `gF`;
the reference then obeys the corresponding fixed parity sector. Special modes
`k=0,N/2` are occupied as needed to satisfy that sector, while an odd
half-integer sector with no special modes contains one lowest-energy generic
quasiparticle; for that pair the Fourier occupations are `1` at the excited
momentum and `0` at its partner. Without sector information, the helper keeps
the legacy plotting fallback and minimizes each signed special-mode
contribution independently.
"""
function compute_ground_state_occupation(
    k_values,
    J::Real,
    h::Real;
    N::Int=length(k_values),
    spin_bc=nothing,
    gF=nothing,
)::Vector{Float64}
    θ = theta_from_Jh(J, h)
    mode_indices = [_mode_index_from_momentum(k, N) for k in k_values]
    parity = _reference_spin_parity(spin_bc, gF)
    special_occupations = _special_mode_occupations(mode_indices, θ, N, parity)
    excited_generic_mode = _generic_reference_excitation(mode_indices, θ, N, parity)
    return [
        _ground_state_momentum_occupation(k, θ, N, special_occupations, excited_generic_mode)
        for k in mode_indices
    ]
end

_mode_index_from_momentum(k::Real, N::Int) = Float64(k) * N / 2π

function _ground_state_momentum_occupation(k::Real, θ, N::Int, special_occupations, excited_generic_mode)
    if _is_special_mode_index(k, N)
        if special_occupations !== nothing
            canonical_k = _canonical_special_mode_index(k, N)
            return Float64(get(special_occupations, canonical_k, 0))
        end
        return w_k_coefficient(Float64(k), θ, N) < 0 ? 1.0 : 0.0
    end

    φ_bogo = bogoliubov_angle(Float64(k), θ, N)
    if excited_generic_mode !== nothing
        _same_mode_index(k, excited_generic_mode, N) && return 1.0
        _same_mode_index(k, -excited_generic_mode, N) && return 0.0
    end
    return sin(φ_bogo)^2
end

_is_special_mode_index(k::Real, N::Int) = abs(sin(2π * Float64(k) / N)) < 1e-12

function _canonical_special_mode_index(k::Real, N::Int)
    kf = mod(Float64(k), N)
    isapprox(kf, 0.0; atol=1e-10, rtol=0.0) && return 0
    isapprox(kf, N / 2; atol=1e-10, rtol=0.0) && return div(N, 2)
    error("Mode index $k is not a special mode for N=$N")
end

function _special_mode_occupations(mode_indices, θ, N::Int, parity)
    parity === nothing && return nothing

    special_modes = unique(
        _canonical_special_mode_index(k, N)
        for k in mode_indices
        if _is_special_mode_index(k, N)
    )
    isempty(special_modes) && return nothing

    target_nf_parity = parity == 1 ? 0 : 1
    best_cost = Inf
    best_mask = 0
    for mask in 0:(2^length(special_modes)-1)
        occupations = digits(mask; base=2, pad=length(special_modes))
        sum(occupations) % 2 == target_nf_parity || continue
        cost = sum(
            w_k_coefficient(special_modes[i], θ, N) * occupations[i]
            for i in eachindex(special_modes)
        )
        if cost < best_cost
            best_cost = cost
            best_mask = mask
        end
    end

    occupations = digits(best_mask; base=2, pad=length(special_modes))
    return Dict(special_modes[i] => occupations[i] for i in eachindex(special_modes))
end

function _generic_reference_excitation(mode_indices, θ, N::Int, parity)
    parity === nothing && return nothing
    parity == -1 || return nothing
    any(k -> _is_special_mode_index(k, N), mode_indices) && return nothing

    best_k = nothing
    best_key = nothing
    for k in mode_indices
        kf = Float64(k)
        key = (mode_energy(kf, θ, N), abs(kf), kf)
        if best_key === nothing || key < best_key
            best_k = kf
            best_key = key
        end
    end
    return best_k
end

function _same_mode_index(k1::Real, k2::Real, N::Int)
    δ = mod(Float64(k1) - Float64(k2) + N / 2, N) - N / 2
    return isapprox(δ, 0.0; atol=1e-10, rtol=0.0)
end

function _reference_spin_parity(spin_bc, gF)
    spin_bc === nothing && return nothing
    gF === nothing && return nothing

    gf = Int(gF)
    gf == 1 || gf == -1 || throw(ArgumentError("gF must be +1 or -1, got $gF"))
    gI = spin_bc == :periodic ? 1 :
         spin_bc == :antiperiodic ? -1 :
         throw(ArgumentError("spin_bc must be :periodic or :antiperiodic, got $spin_bc"))
    return -gI * gf
end
