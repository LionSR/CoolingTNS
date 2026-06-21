"""
    mode_analysis.jl

Analytic mode structure for the transverse-field Ising model, unifying the
θ-parametrization from the notes (MapToSpin.tex) with the (J, h) parameters
used in the code.

# Two Hamiltonian forms

**Notes (XX + Z model):**
```
H_notes = (cos θ / 2) Σ σ_x σ_x + (sin θ / 2) Σ σ_z
```
with parity operator ``P_z = ∏ σ_z``.

**Code (ZZ + X model, Ising):**
```
H_code = J Σ σ_z σ_z + h Σ σ_x
```
with parity operator ``P_x = ∏ σ_x``.

# Connecting the two forms

A global π/2 rotation around the Y-axis (``R_y(π/2)``) sends
``σ_x → -σ_z``, ``σ_z → σ_x``, ``σ_y → σ_y``.
Under this rotation:
```
H_notes  →  (cos θ / 2)(-σ_z)(-σ_z) + (sin θ / 2)(σ_x)
         =  (cos θ / 2) Σ σ_z σ_z  +  (sin θ / 2) Σ σ_x
```
Comparing with ``H_code = J Σ σ_z σ_z + h Σ σ_x`` and defining the energy scale
``Λ = 2√(J² + h²)``, we obtain:

```
H_code = Λ · R_y(π/2) H_notes R_y†(π/2)
```
with the identification ``cos θ = J / √(J² + h²)`` and ``sin θ = h / √(J² + h²)``,
i.e., **``θ = atan(h, J)``** (two-argument arctangent).

The parity operator transforms correspondingly: ``P_z → P_x``, so the code's
conserved parity is ``P_x = ∏ σ_{x,i}``, **not** ``P_z``.

# Mode energies

The notes derive (for the unit-scale θ-Hamiltonian):
```
ε_k^(notes) = √(1 - sin(2θ) cos φ_k),    φ_k = 2πk/N
```
In code units (eigenvalues of ``H_code``):
```
ε_k^(code) = Λ · ε_k^(notes) = 2√(J² + h²) · √(1 - sin(2θ) cos φ_k)
```

# Boundary conditions and k-grids

For **spin PBC** (``g_I = +1``) with the code's parity ``P_x``:
- ``P_x = +1`` sector → ``g_F = -g_I · P_x = -1`` → fermionic **APBC** → half-integer k
- ``P_x = -1`` sector → ``g_F = -g_I · P_x = +1`` → fermionic **PBC** → integer k

For **spin APBC** (``g_I = -1``): the assignment is swapped.
"""

# ============================================================================
# Parameter Mapping
# ============================================================================

"""
    mode_occupation_from_hk(hk)

Convert the Bogoliubov mode observable ``h_k = 2 n_k^{Bog} - 1`` to the
Bogoliubov occupation number ``n_k^{Bog}``. This accepts either a scalar or an
array.
"""
mode_occupation_from_hk(hk) = (hk .+ 1) ./ 2

const _ISING_FOURIER_SPIN_BCS = (:periodic, :antiperiodic)

"""
    supports_ising_fourier_observables(ham_params) -> Bool

Return whether the package's Fourier-grid Ising observables are defined for
`ham_params`.

The current ``k``-space momentum and Bogoliubov-mode measurements use the
translation-invariant transverse-field Ising construction in
`Notes/NotesED/MapToSpin.tex`. They are therefore available for Ising
Hamiltonians with even `N` and spin boundary condition `:periodic` or
`:antiperiodic`. Open chains require open-boundary BdG modes, and
nonintegrable Ising or Rydberg Hamiltonians do not use this free-fermion
Fourier basis.
"""
supports_ising_fourier_observables(::Nothing) = false
supports_ising_fourier_observables(::HamiltonianParameters) = false
function supports_ising_fourier_observables(ham_params::HamiltonianParameters{IsingModel})
    return iseven(ham_params.N) && ham_params.bc in _ISING_FOURIER_SPIN_BCS
end

"""
    bath_detuning_energy(delta)

Return the positive energy `|delta|` associated with a bath detuning. A value
of `nothing`, zero, a non-number, or a detuning array whose length is not one
denotes the absence of a single resonant bath line.
"""
function bath_detuning_energy(delta)
    if delta === nothing
        return nothing
    end
    δ = if delta isa AbstractArray
        length(delta) == 1 || return nothing
        only(delta)
    else
        delta
    end
    δ isa Number || return nothing
    δ == 0 && return nothing
    return abs(δ)
end

"""
    nearest_bath_resonance_indices(εk_values, delta; atol=1e-12)

Return every mode index whose quasiparticle energy is closest to `|delta|`.
This implements the resonance condition `ε_k ≈ |Δ|` without assuming that the
nearest mode is unique.
"""
function nearest_bath_resonance_indices(εk_values, delta; atol=1e-12)
    δ_abs = bath_detuning_energy(delta)
    if δ_abs === nothing
        return Int[]
    end
    return _nearest_energy_resonance_indices(εk_values, δ_abs; atol=atol)
end

function _nearest_energy_resonance_indices(εk_values, δ_abs::Real; atol=1e-12)
    isempty(εk_values) && return Int[]

    distances = abs.(εk_values .- δ_abs)
    dmin = minimum(distances)
    return findall(d -> isapprox(d, dmin; atol=atol, rtol=sqrt(eps(Float64))), distances)
end

"""
    theta_from_Jh(J, h) -> θ

Map the code's Ising parameters ``(J, h)`` to the notes' angular parameter ``θ``.

The mapping is ``θ = atan(h, J)``, which gives ``cos θ = J/√(J²+h²)``
and ``sin θ = h/√(J²+h²)``.

The notes' unit-scale Hamiltonian eigenvalues relate to the code's by a factor
``Λ = 2√(J² + h²)``: ``E_code = Λ · E_notes``.

See also: [`Jh_from_theta`](@ref), [`energy_scale`](@ref).
"""
theta_from_Jh(J, h) = atan(h, J)

"""
    Jh_from_theta(θ; scale=1/2) -> (J, h)

Map the notes' angular parameter ``θ`` back to Ising parameters ``(J, h)``.

With the default ``scale = 1/2``, this gives the unit-norm parametrization where
``J² + h² = 1/4`` (matching the notes' prefactor of 1/2). Use a different
`scale` to set ``J = scale · cos θ`` and ``h = scale · sin θ``.

See also: [`theta_from_Jh`](@ref).
"""
function Jh_from_theta(θ; scale=1/2)
    return (scale * cos(θ), scale * sin(θ))
end

"""
    energy_scale(J, h) -> Λ

Return the overall energy scale ``Λ = 2√(J² + h²)`` that relates the code's
eigenvalues to the notes' unit-scale eigenvalues: ``E_code = Λ · E_notes``.
"""
energy_scale(J, h) = 2 * sqrt(J^2 + h^2)

# ============================================================================
# Analytic Dispersion Functions (θ-parametrization, unit scale)
# ============================================================================

"""
    mode_energy(k, θ, N) -> ε_k

Quasiparticle energy for mode ``k`` in the notes' unit-scale θ-parametrization:
``ε_k = √(1 - sin(2θ) cos(2πk/N))``.

Returns the energy in notes units. To convert to code units, multiply by
[`energy_scale`](@ref)`(J, h)`.

For the special modes ``k = 0`` and ``k = N/2`` (only present for ``g_F = +1``),
this returns ``|w_k|`` which equals ``|sin θ ∓ cos θ|``.
"""
function mode_energy(k, θ, N)
    φk = 2π * k / N
    return sqrt(1 - sin(2θ) * cos(φk))
end

"""
    mode_energy_Jh(k, J, h, N) -> ε_k

Quasiparticle energy for mode ``k`` in code units (eigenvalue gap of ``H_code``):
``ε_k = 2√(J²+h²) · √(1 - sin(2θ) cos(2πk/N))``.
"""
function mode_energy_Jh(k, J, h, N)
    θ = theta_from_Jh(J, h)
    return energy_scale(J, h) * mode_energy(k, θ, N)
end

"""
    mode_energies_Jh(k_indices, J, h, N) -> Vector{Float64}

Positive quasiparticle gaps for an allowed momentum grid in code units. These
are the gaps used for bath-resonance diagnostics. They are not the signed
coefficients used in the exact reconstruction
``E = (Λ/2) Σ_k coeff_k h_k``.
"""
function mode_energies_Jh(k_indices, J, h, N)
    θ = theta_from_Jh(J, h)
    Λ = energy_scale(J, h)
    return [Λ * mode_energy(Float64(k), θ, N) for k in k_indices]
end

"""
    is_generic_mode(k, N) -> Bool

Return whether the Fourier index ``k`` is a generic Ising mode.  The special
integer-grid modes have ``sin(2π k/N)=0`` and are excluded from the generic
two-quasiparticle detuning reference.
"""
function is_generic_mode(k, N::Int)
    return abs(sin(2π * Float64(k) / N)) > sqrt(eps(Float64))
end

"""
    generic_k_indices(N, gF) -> Vector

Allowed Fourier indices with the special modes removed.
"""
function generic_k_indices(N::Int, gF)
    return filter(k -> is_generic_mode(k, N), allowed_k_indices(N, gF))
end

"""
    ising_mode_detuning_reference(ham_params; parity=1) -> Float64

Return the lowest parity-preserving generic two-quasiparticle energy on the
Fourier grid used by the periodic/antiperiodic Ising mode observables.  The
default `parity=1` is the deterministic reference sector also used when a
measured state parity is not yet available.

This is the bath-detuning reference for the parity-preserving local ``X``
system coupling used by the default mode-resolved Ising cooling diagnostics:
the coupling can create or remove generic quasiparticles only in pairs, so the
reference scale is ``2 min_{sin φ_k != 0} ε_k``.  It is not the
single-quasiparticle mode energy, not the cross-parity many-body gap, and not a
variational DMRG excited-state estimate.  On Fourier grids containing special
modes this generic two-quasiparticle scale can exceed lower same-parity
special-mode transitions, so direct callers should use it only when the generic
pair scale is the intended reference.
"""
function ising_mode_detuning_reference(
    ham_params::HamiltonianParameters{IsingModel};
    parity::Int=1,
)
    supports_ising_fourier_observables(ham_params) || throw(ArgumentError(
        "Ising mode detuning references require even-size periodic or antiperiodic Ising parameters"
    ))
    (parity == 1 || parity == -1) ||
        throw(ArgumentError("parity must be +1 or -1, got $parity"))

    N = ham_params.N
    J, h = ham_params.params.J, ham_params.params.h
    gF = fermionic_bc(ham_params.bc, parity)
    energies = mode_energies_Jh(generic_k_indices(N, gF), J, h, N)
    positive_energies = filter(>(sqrt(eps(Float64))), energies)
    isempty(positive_energies) && throw(ArgumentError(
        "the reference Fourier grid has no strictly positive generic quasiparticle energy"
    ))
    return 2 * minimum(positive_energies)
end

function ising_mode_detuning_reference(
    ham_params::HamiltonianParameters;
    parity::Int=1,
)
    throw(ArgumentError(
        "Ising mode detuning references are only defined for integrable Ising parameters"
    ))
end

# ============================================================================
# Open-boundary BdG source of truth
# ============================================================================

"""
    obc_bdg_matrices(θ, N) -> (A, B)

Return the open-boundary real-space BdG matrices for the notes' canonical
Jordan-Wigner convention, in notes units.

For
```
H = (cos θ / 2) Σ_{n=1}^{N-1} (a_n - a_n†)(a_{n+1} + a_{n+1}†)
    - (sin θ / 2) Σ_{n=1}^N (a_n a_n† - a_n† a_n),
```
the nonzero matrix elements are
```
A_nn = sin θ,
A_{n,n+1} = A_{n+1,n} = -cos θ / 2,
B_{n,n+1} = -cos θ / 2,
B_{n+1,n} =  cos θ / 2.
```
"""
function obc_bdg_matrices(θ::Real, N::Int)
    N >= 1 || throw(ArgumentError("N must be positive, got $N"))
    s = sin(θ)
    c = cos(θ)
    A = zeros(Float64, N, N)
    B = zeros(Float64, N, N)
    for n in 1:N
        A[n, n] = s
    end
    for n in 1:(N - 1)
        A[n, n + 1] = -c / 2
        A[n + 1, n] = -c / 2
        B[n, n + 1] = -c / 2
        B[n + 1, n] = c / 2
    end
    return A, B
end

"""
    obc_bdg_matrix(θ, N) -> Matrix{Float64}

Return the ``2N x 2N`` open-boundary BdG matrix ``[A B; -B -A]`` in the
canonical notes convention.
"""
function obc_bdg_matrix(θ::Real, N::Int)
    A, B = obc_bdg_matrices(θ, N)
    return [A B; -B -A]
end

"""
    obc_mode_energies(θ, N) -> Vector{Float64}

Return the positive open-boundary BdG quasiparticle energies in notes units.
The corresponding many-body spectrum is
``Σ_k ε_k (n_k - 1/2)`` with ``n_k ∈ {0,1}``.
"""
function obc_mode_energies(θ::Real, N::Int)
    Hbdg = Symmetric(obc_bdg_matrix(θ, N))
    values = sort(eigvals(Hbdg))
    return values[(N + 1):(2N)]
end

"""
    obc_mode_energies_Jh(J, h, N) -> Vector{Float64}

Return the open-boundary BdG quasiparticle energies in code units for
``H_code = J Σ Z_n Z_{n+1} + h Σ X_n``.
"""
function obc_mode_energies_Jh(J::Real, h::Real, N::Int)
    θ = theta_from_Jh(J, h)
    return energy_scale(J, h) * obc_mode_energies(θ, N)
end

"""
    w_k_coefficient(k, θ, N) -> w_k

BdG diagonal coefficient: ``w_k = sin θ - cos θ · cos(2πk/N)``.

In the BdG Nambu block ``α†_k H_k α_k`` with ``α_k = (ã_k, ã†_{-k})^T``,
the diagonal entries are ``(w_k, -w_k)``.

This matches the notes (MapToSpin.tex, Eq. bdg_block).
"""
function w_k_coefficient(k, θ, N)
    φk = 2π * k / N
    return sin(θ) - cos(θ) * cos(φk)
end

"""
    r_k_coefficient(k, θ, N) -> r_k

BdG off-diagonal coefficient: ``r_k = cos θ · sin(2πk/N)``.

In the BdG Nambu block ``α†_k H_k α_k`` with ``α_k = (ã_k, ã†_{-k})^T``,
the off-diagonal entry is ``i r_k``:
```
H_k = [[ w_k,  i r_k ],
       [-i r_k, -w_k  ]]
```

!!! note "Fourier convention"
    The notes (MapToSpin.tex) define ``ã_k = (1/√N) Σ exp(-inφ_k) a_n``,
    which gives ``r_k = -cos θ sin(φ_k)``. The code's Fourier transform
    (in `_build_fourier_ops`) uses the opposite sign convention
    ``ã_k = (1/√N) Σ exp(+inφ_k) a_n``, effectively swapping ``k ↔ -k``.
    To keep the BdG block and Bogoliubov angles consistent with this choice,
    we use ``r_k = +cos θ sin(φ_k)`` here. The two conventions give identical
    physical predictions (mode energies, Bogoliubov occupations, ⟨h_k⟩).
"""
function r_k_coefficient(k, θ, N)
    φk = 2π * k / N
    return cos(θ) * sin(φk)
end

"""
    bogoliubov_angle(k, θ, N) -> varphi_k

Bogoliubov angle satisfying ``tan(2 varphi_k) = r_k / w_k`` where ``r_k`` and ``w_k``
are the BdG coefficients.

The Bogoliubov transformation is:
```
â_k = cos(varphi_k) ã_k + i sin(varphi_k) ã†_{-k}
```

For the ground state of a generic paired block:
``tilde n_k = ⟨ã†_k ã_k⟩ = sin²(varphi_k)``. Here ``varphi_k`` is the
Bogoliubov angle; the momentum angle remains ``φ_k = 2πk/N``.

For the special modes ``k = 0, N/2`` where ``r_k = 0``, returns ``0``.

The sign of ``r_k`` is already adjusted to the ``ã_k ∝ exp(+i n φ_k)``
convention used by ED `_build_fourier_ops` when constructing
`_build_hk_operator`; TN mode observables use the same resulting ``h_k``
formula.  This mode-observable convention is distinct from the raw ED
momentum-distribution loop, which reports notes-sign Fourier occupations.

See also: `_build_hk_operator` in `ed_backend_complex_jw.jl` and `measure_hk`
in `tn_mode_observables.jl`, which both use this angle in the same ``h_k``
formula.
"""
function bogoliubov_angle(k, θ, N)
    wk = w_k_coefficient(k, θ, N)
    rk = r_k_coefficient(k, θ, N)
    if !is_generic_mode(k, N)
        return 0.0
    end
    return atan(rk, wk) / 2
end

"""
    coeff_k(k, θ, N) -> coefficient

The symmetrized-form coefficient for mode ``k`` in the notes' decomposition
``H = (1/2) Σ_k coeff_k · h_k``.

For generic modes, ``coeff_k = ε_k``.  For the special modes ``k = 0`` and
``k = N/2``, ``coeff_k = w_k`` (signed, not absolute value).

This is in **notes units**; multiply by ``Λ`` for code units.
"""
function coeff_k(k, θ, N)
    if !is_generic_mode(k, N)
        return w_k_coefficient(k, θ, N)  # signed
    else
        return mode_energy(k, θ, N)  # always positive
    end
end

"""
    ising_energy_from_mode_hk(k_indices, hk_values, ham_params) -> energy

Reconstruct the Ising energy from the Bogoliubov mode observables ``h_k``.

For a fixed fermionic boundary-condition sector, the code-unit energy is
``E = (Λ/2) Σ_k coeff_k h_k``, where ``Λ = 2√(J²+h²)`` and `coeff_k` is the
signed coefficient used in the notes-basis diagonal decomposition. The input
`k_indices` must be the same mode grid used to produce `hk_values`.

If `hk_values` is a vector, this returns one energy. If `hk_values` is a matrix,
rows are interpreted as cooling steps and columns as modes, and a vector of
stepwise reconstructed energies is returned.
"""
function ising_energy_from_mode_hk(k_indices, hk_values::AbstractVector,
                                   ham_params::HamiltonianParameters{IsingModel})
    length(k_indices) == length(hk_values) || throw(ArgumentError(
        "k_indices length $(length(k_indices)) does not match hk_values length $(length(hk_values))"
    ))

    N = ham_params.N
    J, h = ham_params.params.J, ham_params.params.h
    θ = theta_from_Jh(J, h)
    Λ = energy_scale(J, h)

    return (Λ / 2) * sum(
        coeff_k(Float64(k), θ, N) * hk_values[i]
        for (i, k) in enumerate(k_indices)
    )
end

function ising_energy_from_mode_hk(k_indices, hk_values::AbstractMatrix,
                                   ham_params::HamiltonianParameters{IsingModel})
    size(hk_values, 2) == length(k_indices) || throw(ArgumentError(
        "hk_values has $(size(hk_values, 2)) mode columns, but k_indices has length $(length(k_indices))"
    ))

    return [
        ising_energy_from_mode_hk(k_indices, view(hk_values, step, :), ham_params)
        for step in axes(hk_values, 1)
    ]
end

# ============================================================================
# Vacuum Energy
# ============================================================================

"""
    vacuum_energy(N, θ, gF) -> E_vac

Bogoliubov vacuum energy in notes units: ``E_vac = -(1/2) Σ_k coeff_k``.

For ``g_F = -1`` (half-integer k, no special modes):
``E_vac = -Σ_{k=1/2}^{(N-1)/2} ε_k``.

For ``g_F = +1`` (integer k, with special modes ``k=0, N/2``):
``E_vac = -(1/2)(Σ_{all k} coeff_k)``.
"""
function vacuum_energy(N, θ, gF)
    ks = allowed_k_indices(N, gF)
    return -sum(coeff_k(k, θ, N) for k in ks) / 2
end

"""
    vacuum_energy_Jh(N, J, h, gF) -> E_vac

Bogoliubov vacuum energy in code units.
"""
function vacuum_energy_Jh(N, J, h, gF)
    θ = theta_from_Jh(J, h)
    return energy_scale(J, h) * vacuum_energy(N, θ, gF)
end

# ============================================================================
# Allowed k-grid Functions
# ============================================================================

"""
    allowed_k_indices(N, gF) -> Vector

Return the allowed k-indices for an N-site chain with fermionic boundary
condition ``g_F``.

- ``g_F = +1`` (fermionic PBC): integer k ∈ {-N/2+1, …, N/2}
- ``g_F = -1`` (fermionic APBC): half-integer k ∈ {-(N-1)/2, …, (N-1)/2}

The momentum values are ``φ_k = 2πk/N``.
"""
function allowed_k_indices(N::Int, gF::Int)
    @assert gF == 1 || gF == -1 "gF must be +1 or -1, got $gF"
    @assert iseven(N) "N must be even, got $N"
    if gF == 1
        # Integer k: -N/2+1, ..., N/2
        return collect(-div(N, 2) + 1:div(N, 2))
    else
        # Half-integer k: -(N-1)/2, ..., (N-1)/2
        return [k + 1//2 for k in -div(N, 2):div(N, 2) - 1]
    end
end

"""
    allowed_k_indices(N, gF::Symbol) -> Vector

Convenience method accepting `:periodic` (``g_F=+1``) or `:antiperiodic` (``g_F=-1``).
"""
function allowed_k_indices(N::Int, gF::Symbol)
    if gF == :periodic
        return allowed_k_indices(N, 1)
    elseif gF == :antiperiodic
        return allowed_k_indices(N, -1)
    else
        error("gF must be :periodic or :antiperiodic, got $gF")
    end
end

"""
    fermionic_bc(spin_bc::Symbol, parity::Int) -> Int

Compute the fermionic boundary condition ``g_F = -g_I · P`` from the spin
boundary condition and parity sector.

# Convention (for the code's ZZ+X Hamiltonian)

The code's conserved parity is ``P_x = ∏ σ_{x,i}`` (not ``P_z``!).
This is because H_code is related to H_notes by a global π/2 Y-rotation
that maps ``σ_z → -σ_x``, hence ``P_z = ∏ σ_z → (-1)^N ∏ σ_x = P_x``
(for even N).

- Spin PBC (``g_I = +1``) with ``P_x = +1``: ``g_F = -1`` (fermionic APBC, half-integer k)
- Spin PBC (``g_I = +1``) with ``P_x = -1``: ``g_F = +1`` (fermionic PBC, integer k)

# Arguments
- `spin_bc`: `:periodic` (``g_I=+1``) or `:antiperiodic` (``g_I=-1``)
- `parity`: ``+1`` or ``-1`` (eigenvalue of ``P_x``)

# Returns
- `gF`: ``+1`` or ``-1``
"""
function fermionic_bc(spin_bc::Symbol, parity::Int)
    @assert parity == 1 || parity == -1 "parity must be +1 or -1"
    gI = spin_bc == :periodic ? 1 : (spin_bc == :antiperiodic ? -1 : error("Unknown BC: $spin_bc"))
    return -gI * parity
end

"""
    _reference_parity_sector_with_source(px; atol=0.1, default=1)

Return `(parity, source)` for an automatic Fourier-grid choice from a measured
value of ``⟨P_x⟩``.  The source is `:state` when ``⟨P_x⟩`` selects a definite
parity sector within `atol`, and `:reference` when the deterministic fallback
sector is used.
"""
function _reference_parity_sector_with_source(px::Real; atol=0.1, default::Int=1)
    @assert default == 1 || default == -1 "default must be +1 or -1"
    abs(px - 1) <= atol && return (parity=1, source=:state)
    abs(px + 1) <= atol && return (parity=-1, source=:state)
    return (parity=default, source=:reference)
end

"""
    _reference_parity_sector(px; atol=0.1, default=1) -> Int

Return the parity sector used when an automatic Fourier grid must be chosen
from a measured value of ``⟨P_x⟩``.

If ``⟨P_x⟩`` is within `atol` of ``+1`` or ``-1``, the corresponding sector is
used.  Otherwise the state has no unique fermionic boundary condition; the
function returns the deterministic reference sector `default`, which is the
even sector by default.  This is a diagnostic convention, not a projection of
the state onto that sector.
"""
function _reference_parity_sector(px::Real; atol=0.1, default::Int=1)
    return _reference_parity_sector_with_source(px; atol=atol, default=default).parity
end

"""
    _reference_fermionic_bc(spin_bc, px; atol=0.1, default_parity=1) -> Int

Return the fermionic boundary condition used by automatic Fourier diagnostics.
Parity eigenstates use their physical sector.  States with no definite
``P_x`` parity use the deterministic reference sector from
[`_reference_parity_sector`](@ref), so ED, MPS, MPO, and cooling diagnostics
share the same grid convention.
"""
function _reference_fermionic_bc(spin_bc::Symbol, px::Real; atol=0.1, default_parity::Int=1)
    return fermionic_bc(
        spin_bc,
        _reference_parity_sector_with_source(px; atol=atol, default=default_parity).parity,
    )
end

# ============================================================================
# Parity Operator (for the code's Hamiltonian)
# ============================================================================

"""
    parity_operator_code(N) -> Matrix

Build the parity operator ``P_x = ∏_{i=1}^N σ_{x,i}`` for the code's
Hamiltonian ``H = J Σ σ_z σ_z + h Σ σ_x``.

This is the conserved quantity (commutes with H_code), analogous to
``P_z = ∏ σ_{z,i}`` in the notes' Hamiltonian.
"""
function parity_operator_code(N::Int)
    result = pauli_x(1, N)
    for i in 2:N
        result = result * pauli_x(i, N)
    end
    return result
end
