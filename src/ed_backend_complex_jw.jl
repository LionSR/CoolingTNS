"""
    jordan_wigner_transform_complex(site::Int, N::Int) -> (a, a†)

Jordan-Wigner transformation returning complex fermionic operators for the
spin-to-fermion mapping described in MapToSpin.tex.

# Convention (matching the notes)

The annihilation and creation operators are defined as:
```
  a_n    = -S_n · (σ_{n,x} - iσ_{n,y})/2  = -S_n · σ⁻_n
  a†_n   = -S_n · (σ_{n,x} + iσ_{n,y})/2  = -S_n · σ⁺_n
```
where ``S_n = ∏_{j<n} σ_{j,z}`` is the Jordan-Wigner string.

This gives ``a†a = σ⁺σ⁻ = (I + σ_z)/2``, so the **vacuum is all-spins-down**
(``σ_z = -1``) and **occupied = spin up** (``σ_z = +1``). The number operator is
``n = a†a = (1 + σ_z)/2``, hence ``σ_z = 2n - 1``.

# Why this convention?

The notes derive free-fermion formulas (mode energies ``ε_k``, Bogoliubov angles,
ground-state raw Fourier occupation ``tilde n_k = sin²(varphi_k)``) using this
JW mapping applied to ``H_notes = (cosθ/2) Σ σ_x σ_x + (sinθ/2) Σ σ_z``. The
code's Hamiltonian ``H_code = J Σ σ_z σ_z + h Σ σ_x`` is related by a global
``R_y(π/2)`` rotation, which does not affect the JW operators (they are defined
in terms of abstract Pauli matrices in the computational basis).

With this convention, ``⟨ã†_k ã_k⟩`` measured on the ground state of ``H_notes``
(or equivalently on the rotated ground state of ``H_code``) matches the
analytical prediction ``tilde n_k^{GS} = sin²(varphi_k)`` for generic modes,
where ``varphi_k`` is the Bogoliubov angle, not the momentum angle
``φ_k = 2πk/N``.

# Note on the minus sign

The global ``(-1)`` in ``a = -S·σ⁻`` cancels in all bilinears (``a†a``, ``a†a†``,
etc.), so the physical content is identical to using ``a = +S·σ⁻``. However,
we keep the minus sign to maintain exact correspondence with the notes' equations
(particularly the identity ``σ_{n,x} = -S_n(a_n + a_n†)``).
"""
function jordan_wigner_transform_complex(site::Int, N::Int)
    # Jordan-Wigner string operator ∏_{j < site} Z_j
    string_op = I(2^N)
    for j in 1:(site - 1)
        string_op *= pauli_z(j, N)
    end

    X_i = pauli_x(site, N)
    Y_i = pauli_y_complex(site, N)  # Need the COMPLEX Y = [0,-i;i,0], not real Y

    # Notes convention: a = -S · σ⁻ = -S · (X - iY)/2
    #                   a† = -S · σ⁺ = -S · (X + iY)/2
    a = -string_op * (X_i - im * Y_i) / 2
    a_dag = -string_op * (X_i + im * Y_i) / 2

    return (a, a_dag)
end

"""
    pauli_y_complex(i::Int, n_qubits::Int) -> SparseMatrixCSC{ComplexF64}

Complex Pauli Y operator ``σ_y = [0, -i; i, 0]`` acting on qubit `i`.

This is the standard complex σ_y, as opposed to the real representation
``Y_{real} = -iσ_y = [0, -1; 1, 0]`` used in `pauli_y`.
"""
function pauli_y_complex(i::Int, n_qubits::Int)
    PAULI_Y_COMPLEX = sparse(ComplexF64[0 -im; im 0])
    return single_site_operator(PAULI_Y_COMPLEX, i, n_qubits)
end

# Note: single_site_operator is defined in ed_backend.jl and only accepts
# SparseMatrixCSC{Float64}. We need a ComplexF64 version:
function single_site_operator(local_op::SparseMatrixCSC{ComplexF64, Int}, i::Int, n_qubits::Int)
    @assert 1 <= i <= n_qubits "Qubit index out of range"

    IDENTITY_2X2_C = sparse(ComplexF64[1 0; 0 1])
    op = sparse(ComplexF64[1;;])
    for j in n_qubits:-1:1
        op = kron(op, j == i ? local_op : IDENTITY_2X2_C)
    end
    return op
end

# -----------------------------------------------------------------------------
# Momentum distribution (shared implementation)
# -----------------------------------------------------------------------------

_expect_amdag_an(ψ::EDStateVector, a_m_dag, a_n) = dot(ψ.data, a_m_dag * a_n * ψ.data)
_expect_amdag_an(ρ::EDDensityMatrix, a_m_dag, a_n) = tr(ρ.data * a_m_dag * a_n)

function _measure_momentum_distribution_ed_clean(state, ham_params; gF=nothing)
    N = ham_params.N

    # Determine k-grid: parity-aware if gF not specified
    if isnothing(gF)
        px = measure_state_parity(state, N)
        sector = _reference_parity_sector_with_source(px)
        parity = sector.parity
        if sector.source === :reference
            @warn "measure_momentum_distribution: state has no definite P_x parity " *
                  "(⟨P_x⟩ = $px); using the P_x = $parity reference grid"
        end
        gF = fermionic_bc(ham_params.bc, parity)
    end

    k_indices = allowed_k_indices(N, gF)
    n_k = zeros(Float64, length(k_indices))

    # Rotate state to notes basis (JW operators are defined in computational/notes basis)
    if state isa EDStateVector
        ψ_notes = _rotate_state_to_notes(state)
        notes_state = EDStateVector(ψ_notes, N)
    else
        ρ_notes = _rotate_dm_to_notes(state)
        notes_state = EDDensityMatrix(ρ_notes, N)
    end

    # Precompute all JW operators once (these are expensive 2^N × 2^N matrices)
    a1, a1_dag = jordan_wigner_transform_complex(1, N)
    a_ops = Vector{typeof(a1)}(undef, N)
    a_dag_ops = Vector{typeof(a1_dag)}(undef, N)
    a_ops[1] = a1
    a_dag_ops[1] = a1_dag
    for j in 2:N
        a_ops[j], a_dag_ops[j] = jordan_wigner_transform_complex(j, N)
    end

    for (ki, k) in enumerate(k_indices)
        nk = 0.0 + 0.0im
        kf = Float64(k)

        # ⟨ã†_k ã_k⟩ with the notes Fourier convention
        # ã_k = (1/√N) Σ_j exp(-i n φ_k) a_j.
        # Therefore ã†_k ã_k carries exp(+i (m-n) φ_k).
        for m in 1:N, n in 1:N
            phase = exp(2π * im * kf * (m - n) / N) / N
            nk += phase * _expect_amdag_an(notes_state, a_dag_ops[m], a_ops[n])
        end

        n_k[ki] = real(nk)
    end

    k_momentum = [2π * Float64(k) / N for k in k_indices]
    return k_momentum, n_k
end

"""
    measure_momentum_distribution_ed_clean(state, ham_params; gF=nothing) -> (k_values, n_k)

Compute the Fourier-fermion occupation ``tilde n_k = ⟨ã_k^† ã_k⟩`` using
the complex Jordan–Wigner mapping (notes convention). The returned array is
named ``n_k`` by the existing API.

With this convention, ``tilde n_k = (1 + ⟨σ_z⟩)/2`` per mode, and the
generic-mode ground-state prediction from Bogoliubov theory is
``tilde n_k^{GS} = sin²(varphi_k)``, where ``varphi_k`` is the Bogoliubov
angle, not the momentum angle ``φ_k = 2πk/N``.

Supported boundary conditions: `:periodic`, `:antiperiodic`.

# Parity-aware k-grid selection

The correct Fourier basis depends on the **fermionic** boundary condition
``g_F = -g_I · P``, which combines the spin BC (``g_I``) with the state's
parity (``P = ⟨P_x⟩``). For spin PBC:

- Even parity (``P_x = +1``): ``g_F = -1`` → half-integer k (fermionic APBC)
- Odd parity (``P_x = -1``): ``g_F = +1`` → integer k (fermionic PBC)

By default (``gF=nothing``), the function measures the state's parity. Parity
eigenstates use their physical sector. States with no definite ``P_x`` parity
have no unique fermionic boundary condition, so the function uses the even
reference sector shared by the cooling diagnostics. Pass `gF=±1` to override.
"""
measure_momentum_distribution_ed_clean(ψ::EDStateVector, ham_params; gF=nothing) =
    _measure_momentum_distribution_ed_clean(ψ, ham_params; gF=gF)

measure_momentum_distribution_ed_clean(ρ::EDDensityMatrix, ham_params; gF=nothing) =
    _measure_momentum_distribution_ed_clean(ρ, ham_params; gF=gF)

# =============================================================================
# Basis rotation: code ↔ notes
# =============================================================================

"""
    _rotation_code_to_notes(N::Int) -> Matrix{ComplexF64}

Build the unitary U†^{⊗N} that rotates a state from the code basis to the notes basis.

The code's Hamiltonian H_code = J ΣZ_iZ_{i+1} + h ΣX_i is related to the notes'
Hamiltonian H_notes = (cosθ/2)ΣX_iX_{i+1} + (sinθ/2)ΣZ_i by
``H_code = Λ · U H_notes U†`` where ``U = R_y(π/2)^{⊗N}``.

Therefore ``|ψ_notes⟩ = U† |ψ_code⟩``.

``R_y(π/2) = (1/√2)[1 -1; 1 1]``, so ``R_y(-π/2) = R_y†(π/2) = (1/√2)[1 1; -1 1]``.
"""
function _rotation_code_to_notes(N::Int)
    # R_y(-π/2) = R_y(π/2)†
    Ry_dag = ComplexF64[1/√2  1/√2; -1/√2  1/√2]
    # Build tensor product R_y(-π/2)^⊗N
    U = ComplexF64[1;;]
    for _ in 1:N
        U = kron(U, Ry_dag)
    end
    return U
end

"""
    _rotate_state_to_notes(state::EDStateVector) -> Vector{ComplexF64}

Rotate a code-basis state vector to the notes basis.
"""
function _rotate_state_to_notes(state::EDStateVector)
    U = _rotation_code_to_notes(state.n_qubits)
    return U * state.data
end

"""
    _rotate_dm_to_notes(state::EDDensityMatrix) -> Matrix{ComplexF64}

Rotate a code-basis density matrix to the notes basis.
"""
function _rotate_dm_to_notes(state::EDDensityMatrix)
    U = _rotation_code_to_notes(state.n_qubits)
    ρ_notes = U * state.data * U'
    return (ρ_notes + ρ_notes') / 2
end

# =============================================================================
# Parity measurement
# =============================================================================

"""
    measure_state_parity(state::Union{EDStateVector, EDDensityMatrix}, N::Int) -> Float64

Measure ⟨P_x⟩ = ⟨∏σ_{x,i}⟩ for the code's Hamiltonian.

This determines which parity sector the state is in:
- Returns ≈ +1 for even parity sector
- Returns ≈ -1 for odd parity sector
- Fractional values indicate a superposition or mixture of parity sectors

Uses the `parity_operator_code` function from `mode_analysis.jl`.
"""
function measure_state_parity(state::EDStateVector, N::Int)
    Px = parity_operator_code(N)
    return real(dot(state.data, Px * state.data))
end

function measure_state_parity(state::EDDensityMatrix, N::Int)
    Px = parity_operator_code(N)
    return real(tr(Px * state.data))
end

# =============================================================================
# Bogoliubov mode observable measurement: h_k
# =============================================================================

"""
    _build_fourier_ops(a_ops, a_dag_ops, k, N) -> (ã_k, ã†_k)

Build the Fourier-transformed momentum-space operators:
```
ã_k    = (1/√N) Σ_n exp(+i·2πkn/N) a_n
ã†_k   = (1/√N) Σ_n exp(-i·2πkn/N) a†_n
```

!!! note "Fourier sign convention"
    This uses ``exp(+inφ_k)`` for the annihilation transform, which is the
    **opposite** sign from the notes (MapToSpin.tex Eq. fourier_def_ann uses
    ``exp(-inφ_k)``). The two conventions are related by ``ã_k^{here} = ã_{-k}^{notes}``.
    This sign difference is compensated by the sign of ``r_k`` in `mode_analysis.jl`,
    so that the BdG blocks, Bogoliubov transformation, and all physical
    observables (ε_k, n_k^{Bog}, ⟨h_k⟩) are correct.
"""
function _build_fourier_ops(a_ops, a_dag_ops, k, N)
    dim = size(a_ops[1], 1)
    ak = zeros(ComplexF64, dim, dim)
    akd = zeros(ComplexF64, dim, dim)
    for n in 1:N
        phase = exp(2π * im * k * n / N) / sqrt(N)
        ak .+= phase * a_ops[n]
        akd .+= conj(phase) * a_dag_ops[n]
    end
    return ak, akd
end

"""
    _build_hk_operator(k, θ, N, a_ops, a_dag_ops) -> Matrix{ComplexF64}

Build the mode operator ``h_k = 2â†_k â_k - 1`` as an explicit 2^N × 2^N matrix
in the notes basis.

The Bogoliubov transformation (with the correct phase from the BdG matrix) is:
```
â_k  = cos(varphi) ã_k + i sin(varphi) ã†_{-k}
â†_k = cos(varphi) ã†_k - i sin(varphi) ã_{-k}
```

This gives:
```
â†_k â_k = cos²(varphi) ã†_k ã_k + sin²(varphi)(1 - ã†_{-k} ã_{-k})
           + i sin(varphi)cos(varphi)(ã†_k ã†_{-k} - ã_{-k} ã_k)
```

For special modes (k = 0, N/2) where the Bogoliubov angle is 0:
``h_k = 2ã†_k ã_k - 1``
"""
function _build_hk_operator(k, θ, N, a_ops, a_dag_ops)
    dim = size(a_ops[1], 1)
    II = Matrix{ComplexF64}(I, dim, dim)

    ãk, ãkd = _build_fourier_ops(a_ops, a_dag_ops, k, N)

    if !is_generic_mode(k, N)
        # Special mode: k=0 or k=N/2 — Bogoliubov angle is 0
        # h_k = 2 ã†_k ã_k - 1
        nk_op = ãkd * ãk
        return 2 * nk_op - II
    else
        # Generic mode: use full Bogoliubov transformation
        varphi_bogo = bogoliubov_angle(Float64(k), θ, N)
        c2 = cos(varphi_bogo)^2
        s2 = sin(varphi_bogo)^2
        sc = sin(varphi_bogo) * cos(varphi_bogo)
        ãmk, ãmkd = _build_fourier_ops(a_ops, a_dag_ops, -k, N)

        nk_op = ãkd * ãk       # ã†_k ã_k
        nmk_op = ãmkd * ãmk    # ã†_{-k} ã_{-k}
        # Pairing term with correct i factor:
        # i·sc·(ã†_k ã†_{-k} - ã_{-k} ã_k)
        pair_op = im * sc * (ãkd * ãmkd - ãmk * ãk)

        bogo_nk = c2 * nk_op + s2 * (II - nmk_op) + pair_op
        return 2 * bogo_nk - II
    end
end

"""
    _expect_complex(op::AbstractMatrix, ψ::Vector{ComplexF64}) -> ComplexF64

Compute ⟨ψ|op|ψ⟩ for a raw state vector (not wrapped in EDStateVector).
"""
_expect_complex(op::AbstractMatrix, ψ::Vector{ComplexF64}) = dot(ψ, op * ψ)

"""
    _expect_complex(op::AbstractMatrix, ρ::Matrix{ComplexF64}) -> ComplexF64

Compute Tr(op·ρ) for a raw density matrix.
"""
_expect_complex(op::AbstractMatrix, ρ::Matrix{ComplexF64}) = tr(op * ρ)

"""
    measure_hk(state::Union{EDStateVector, EDDensityMatrix}, k, ham_params) -> Float64

Compute the Bogoliubov mode observable ``⟨h_k⟩`` for mode ``k``.

``h_k = 2â†_k â_k - 1`` where ``â_k`` is the Bogoliubov quasiparticle annihilation
operator for mode ``k``.

# Returns
- `Float64`: The dimensionless Bogoliubov mode observable, in the range [-1, +1].
  - ``⟨h_k⟩ = -1``: mode is in its ground state (Bogoliubov vacuum)
  - ``⟨h_k⟩ = +1``: mode is maximally excited
  - ``⟨h_k⟩ = 0``: mode is in an equal mixture (e.g., infinite temperature)

# Method
1. Rotates the state from the code basis (ZZ + X) to the notes basis (XX + Z)
   using ``R_y(-π/2)^{⊗N}``.
2. Builds the JW operators in the notes basis.
3. Constructs the ``h_k`` operator using Fourier + Bogoliubov transformation.
4. Computes the expectation value.

# Arguments
- `state`: The quantum state (pure or mixed) in the code basis
- `k`: The mode index (integer or half-integer, as returned by `allowed_k_indices`)
- `ham_params`: Hamiltonian parameters (must have fields `params.J`, `params.h`, `N`)
"""
function measure_hk(state::EDStateVector, k, ham_params)
    N = ham_params.N
    J = ham_params.params.J
    h = ham_params.params.h
    θ = theta_from_Jh(J, h)

    # Rotate state to notes basis
    ψ_notes = _rotate_state_to_notes(state)

    # Build JW operators (in computational/notes basis)
    a_ops = Vector{Matrix{ComplexF64}}(undef, N)
    a_dag_ops = Vector{Matrix{ComplexF64}}(undef, N)
    for n in 1:N
        a, ad = jordan_wigner_transform_complex(n, N)
        a_ops[n] = Matrix(a)
        a_dag_ops[n] = Matrix(ad)
    end

    # Build h_k operator
    hk_op = _build_hk_operator(k, θ, N, a_ops, a_dag_ops)

    # Compute expectation value
    hk_val = _expect_complex(hk_op, ψ_notes)

    # Check imaginary part
    if abs(imag(hk_val)) > 1e-8
        @warn "measure_hk: significant imaginary part $(imag(hk_val)) for k=$k"
    end

    return real(hk_val)
end

function measure_hk(state::EDDensityMatrix, k, ham_params)
    N = ham_params.N
    J = ham_params.params.J
    h = ham_params.params.h
    θ = theta_from_Jh(J, h)

    # Rotate density matrix to notes basis
    ρ_notes = _rotate_dm_to_notes(state)

    # Build JW operators
    a_ops = Vector{Matrix{ComplexF64}}(undef, N)
    a_dag_ops = Vector{Matrix{ComplexF64}}(undef, N)
    for n in 1:N
        a, ad = jordan_wigner_transform_complex(n, N)
        a_ops[n] = Matrix(a)
        a_dag_ops[n] = Matrix(ad)
    end

    # Build h_k operator
    hk_op = _build_hk_operator(k, θ, N, a_ops, a_dag_ops)

    # Compute expectation value
    hk_val = _expect_complex(hk_op, ρ_notes)

    # Check imaginary part
    if abs(imag(hk_val)) > 1e-8
        @warn "measure_hk: significant imaginary part $(imag(hk_val)) for k=$k"
    end

    return real(hk_val)
end

# =============================================================================
# All mode observables and quasiparticle gaps
# =============================================================================

"""
    measure_all_mode_energies(state, ham_params; gF=nothing) 
        -> (k_indices, hk_values, εk_values)

Measure ``⟨h_k⟩`` for all allowed modes and return the positive quasiparticle
gaps used for resonance labels.

The historical function name contains "mode energies", but the measured
observable is the dimensionless ``h_k``.  Energies are reconstructed from
``hk_values`` only through `ising_energy_from_mode_hk`.

# Arguments
- `state`: Quantum state in the code basis (EDStateVector or EDDensityMatrix)
- `ham_params`: Hamiltonian parameters (must be IsingModel with `params.J`, `params.h`)
- `gF`: Fermionic boundary condition (+1 or -1). If `nothing`, determined
  automatically from the state's parity and spin BC.

# Returns
- `k_indices`: Vector of allowed k-indices (integer or half-integer)
- `hk_values`: Vector of ⟨h_k⟩ values (each in [-1, +1])
- `εk_values`: Vector of positive quasiparticle gaps ε_k in code units,
  suitable for bath-resonance diagnostics

These stored gaps are not the signed energy-reconstruction coefficients.  The
total energy satisfies ``⟨H⟩ = (Λ/2) Σ_k coeff_k · ⟨h_k⟩``, where
``Λ = 2√(J²+h²)`` and `coeff_k` keeps the signed special-mode coefficients.
Use `ising_energy_from_mode_hk` for energy reconstruction from `hk_values`.
"""
function measure_all_mode_energies(state::Union{EDStateVector, EDDensityMatrix},
                                    ham_params; gF=nothing)
    N = ham_params.N
    J = ham_params.params.J
    h = ham_params.params.h
    θ = theta_from_Jh(J, h)

    # Determine fermionic BC if not given
    if isnothing(gF)
        px = measure_state_parity(state, N)
        sector = _reference_parity_sector_with_source(px)
        parity = sector.parity
        if sector.source === :reference
            @warn "measure_all_mode_energies: state has no definite P_x parity " *
                  "(⟨P_x⟩ = $px); using the P_x = $parity reference grid"
        end
        gF = fermionic_bc(ham_params.bc, parity)
    end

    ks = allowed_k_indices(N, gF)

    # Build JW operators once (shared across all k)
    a_ops = Vector{Matrix{ComplexF64}}(undef, N)
    a_dag_ops = Vector{Matrix{ComplexF64}}(undef, N)
    for n in 1:N
        a, ad = jordan_wigner_transform_complex(n, N)
        a_ops[n] = Matrix(a)
        a_dag_ops[n] = Matrix(ad)
    end

    # Rotate state to notes basis once
    if state isa EDStateVector
        notes_state = _rotate_state_to_notes(state)
    else
        notes_state = _rotate_dm_to_notes(state)
    end

    hk_values = Float64[]
    εk_values = mode_energies_Jh(ks, J, h, N)

    for k in ks
        # Build h_k operator
        hk_op = _build_hk_operator(k, θ, N, a_ops, a_dag_ops)

        # Compute expectation value
        hk_val = _expect_complex(hk_op, notes_state)

        if abs(imag(hk_val)) > 1e-8
            @warn "measure_all_mode_energies: significant imaginary part " *
                  "$(imag(hk_val)) for k=$k"
        end

        push!(hk_values, real(hk_val))
    end

    return ks, hk_values, εk_values
end
