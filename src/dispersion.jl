"""
Compatibility dispersion and k-space utilities for the transverse-field Ising model.

These exported helpers are used by plotting scripts. They follow the canonical
fermionic momentum and Bogoliubov conventions in `mode_analysis.jl`.
"""

"""
    generate_k_values(N::Int, gF) -> Vector{Float64}

Generate momenta ``2πk/N`` on the canonical fermionic grid.

The boundary condition is fermionic: `gF=+1` or `:periodic` gives integer
momenta, while `gF=-1` or `:antiperiodic` gives half-integer momenta. This is
the same convention as `allowed_k_indices`.
"""
function generate_k_values(N::Int, gF)::Vector{Float64}
    return [2pi * Float64(k) / N for k in allowed_k_indices(N, gF)]
end

"""
    compute_energy_dispersion(k_values, J::Real, h::Real) -> Vector{Float64}

Compute the positive quasiparticle dispersion ``ε_k`` in code units.

This is the notes/code dispersion
``2√(J²+h²) √(1 - sin(2θ) cos(k))``, with ``θ = atan(h,J)``. The input must be
a complete fermionic momentum grid, such as the output of `generate_k_values`.
"""
function compute_energy_dispersion(k_values, J::Real, h::Real)::Vector{Float64}
    N = length(k_values)
    return [mode_energy_Jh(_momentum_to_k_index(k, N), J, h, N) for k in k_values]
end

"""
    compute_bdg_reference_occupation(k_values, J::Real, h::Real) -> Vector{Float64}

Compute the parity-unconstrained BdG reference value of
``⟨ã†_k ã_k⟩`` on the supplied fermionic grid.

For generic modes this is ``sin²(varphi_k)``, where ``varphi_k`` is the
canonical Bogoliubov angle. The momentum angle is ``φ_k = 2πk/N``. For
integer-grid special modes, the reference occupies modes with negative signed
coefficient ``w_k`` and leaves modes with positive ``w_k`` empty; at exact
degeneracy this helper returns `0.5`.

This is a parity-unconstrained, mode-wise energy-minimizing BdG reference
curve on the chosen grid. It should not be identified with the fixed-parity
sector ground-state occupation; see the ED guard in `test/test_mode_analysis.jl`
for a concrete counterexample.
"""
function compute_bdg_reference_occupation(k_values, J::Real, h::Real)::Vector{Float64}
    N = length(k_values)
    θ = theta_from_Jh(J, h)
    return [_bdg_reference_occupation(_momentum_to_k_index(k, N), θ, N) for k in k_values]
end

"""
    compute_ground_state_occupation(k_values, J::Real, h::Real) -> Vector{Float64}

Compatibility wrapper for [`compute_bdg_reference_occupation`](@ref).

The historical name is retained for existing callers.  New plotting code should
prefer `compute_bdg_reference_occupation`, because this reference is not the
fixed-parity sector ground state in every closed-chain sector.
"""
compute_ground_state_occupation(k_values, J::Real, h::Real)::Vector{Float64} =
    compute_bdg_reference_occupation(k_values, J, h)

function _momentum_to_k_index(k::Real, N::Int)
    return Float64(k) * N / (2pi)
end

function _bdg_reference_occupation(k, θ, N)
    if abs(sin(2pi * Float64(k) / N)) < 1e-12
        wk = w_k_coefficient(Float64(k), θ, N)
        abs(wk) < 1e-12 && return 0.5
        return wk < 0 ? 1.0 : 0.0
    end

    varphi = bogoliubov_angle(Float64(k), θ, N)
    return sin(varphi)^2
end
