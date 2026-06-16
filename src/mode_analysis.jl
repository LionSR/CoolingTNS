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

using LinearAlgebra

# ============================================================================
# Parameter Mapping
# ============================================================================

"""
    mode_occupation_from_hk(hk)

Convert the Bogoliubov mode observable ``h_k = 2 n_k - 1`` to the occupation
number ``n_k``. This accepts either a scalar or an array.
"""
mode_occupation_from_hk(hk) = (hk .+ 1) ./ 2

const _ISING_FOURIER_SPIN_BCS = (:periodic, :antiperiodic)

"""
    supports_ising_fourier_observables(ham_params) -> Bool

Return whether the package's Fourier-grid Ising observables are defined for
`ham_params`.

The current ``k``-space momentum and Bogoliubov-mode measurements use the
translation-invariant transverse-field Ising construction in
`Notes/NotesED/MapToSpin.tex`. They are therefore available for even-length
Ising Hamiltonians with spin boundary condition `:periodic` or `:antiperiodic`.
Open chains require open-boundary BdG modes, odd chains are not covered by the
current paired-mode Fourier implementation, and nonintegrable Ising or Rydberg
Hamiltonians do not use this free-fermion Fourier basis.
"""
supports_ising_fourier_observables(::Nothing) = false
supports_ising_fourier_observables(::HamiltonianParameters) = false
function supports_ising_fourier_observables(ham_params::HamiltonianParameters{IsingModel})
    return iseven(ham_params.N) && ham_params.bc in _ISING_FOURIER_SPIN_BCS
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

# ============================================================================
# Open-boundary BdG functions (θ-parametrization, unit scale)
# ============================================================================

"""
    open_bdg_matrices(N, θ) -> (A, B)

Return the open-boundary BdG matrices for the notes' canonical JW convention.

For the open chain,
```
H = (cosθ/2) Σ_{n=1}^{N-1} (a_n - a†_n)(a_{n+1} + a†_{n+1})
    - (sinθ/2) Σ_{n=1}^N (a_n a†_n - a†_n a_n),
```
and the Nambu matrix is ``[[A, B], [-B, -A]]`` with
``A_nn = sinθ``, ``A_{n,n+1}=A_{n+1,n}= -cosθ/2``, and
``B_{n,n+1}= -cosθ/2 = -B_{n+1,n}``.
"""
function open_bdg_matrices(N::Int, θ)
    N >= 1 || throw(ArgumentError("N must be positive, got $N"))
    A = zeros(Float64, N, N)
    B = zeros(Float64, N, N)
    s = sin(θ)
    c = cos(θ)

    for n in 1:N
        A[n, n] = s
    end
    for n in 1:N-1
        A[n, n+1] = -c / 2
        A[n+1, n] = -c / 2
        B[n, n+1] = -c / 2
        B[n+1, n] = c / 2
    end

    return A, B
end

"""
    open_mode_energies(N, θ) -> Vector{Float64}

Return the positive open-boundary quasiparticle energies in notes units, sorted
in increasing order.

The energies are the square roots of the eigenvalues of
``(A - B)(A + B)`` for the matrices returned by [`open_bdg_matrices`](@ref).
"""
function open_mode_energies(N::Int, θ)
    A, B = open_bdg_matrices(N, θ)
    M = (A - B) * (A + B)
    M = (M + transpose(M)) / 2
    λ = eigvals(Symmetric(M))
    return sort(sqrt.(max.(real.(λ), 0.0)))
end

"""
    open_mode_energies_Jh(N, J, h) -> Vector{Float64}

Return the positive open-boundary quasiparticle energies in code units.
"""
function open_mode_energies_Jh(N::Int, J, h)
    θ = theta_from_Jh(J, h)
    return energy_scale(J, h) * open_mode_energies(N, θ)
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
    physical predictions (mode energies, occupations, ⟨h_k⟩).
"""
function r_k_coefficient(k, θ, N)
    φk = 2π * k / N
    return cos(θ) * sin(φk)
end

"""
    bogoliubov_angle(k, θ, N) -> φ_k

Bogoliubov angle satisfying ``tan(2φ_k) = r_k / w_k`` where ``r_k`` and ``w_k``
are the BdG coefficients.

The Bogoliubov transformation is:
```
â_k = cos(φ_k) ã_k + sin(φ_k) ã†_{-k}
```

For the ground state: ``n_k = ⟨ã†_k ã_k⟩ = sin²(φ_k)``.

For the special modes ``k = 0, N/2`` where ``r_k = 0``, returns ``0``.
"""
function bogoliubov_angle(k, θ, N)
    wk = w_k_coefficient(k, θ, N)
    rk = r_k_coefficient(k, θ, N)
    if abs(rk) < 1e-15
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
    φk = 2π * k / N
    # Special modes have sin(φ_k) = 0
    if abs(sin(φk)) < 1e-12
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
        for (i, k) in pairs(k_indices)
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
    reference_parity_sector(px; atol=0.1, default=1, label="state") -> Int

Choose a parity sector from a measured expectation value ``px = <P_x>``.

If ``px`` is close to ``+1`` or ``-1``, the corresponding sector is returned.
If the state is not close to a parity eigenstate, the sign of ``px`` selects the
sector with larger weight. When the expectation is also close to zero, the
`default` sector is used. This gives cooling measurements a deterministic
reference grid in degenerate ground spaces where an eigensolver may return a
valid ground-state vector that is not itself a parity eigenstate.
"""
function reference_parity_sector(px::Real; atol=0.1, default::Int=1, label="state")
    @assert default == 1 || default == -1 "default must be +1 or -1"

    if abs(px - 1) <= atol
        return 1
    elseif abs(px + 1) <= atol
        return -1
    end

    parity = abs(px) > atol ? (px > 0 ? 1 : -1) : default
    @warn "$label parity ⟨Px⟩ = $px is not close to ±1; using parity sector $parity"
    return parity
end

function reference_fermionic_bc(spin_bc::Symbol, px::Real; atol=0.1, default_parity::Int=1, label="state")
    return fermionic_bc(
        spin_bc,
        reference_parity_sector(px; atol=atol, default=default_parity, label=label),
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
