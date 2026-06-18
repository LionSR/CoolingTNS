"""
    tn_mode_observables.jl

Tensor-network measurements of Ising mode observables.

The formulas follow `Notes/NotesED/MapToSpin.tex`: the code Hamiltonian is
rotated to the notes basis by `R_y(-π/2)`, and the Jordan-Wigner strings are
then evaluated as Pauli strings on the original tensor-network state.

For MPS states the O(N^2) split-string correlators are evaluated by direct
network contraction. For density matrices the same Pauli strings are converted
to MPOs and contracted as `Tr(ρ O)`. These paths are convention-correct; a later
production implementation can still cache environments across correlators for
large scans.
"""

using ITensors
using ITensorMPS

const _PAULI_LABELS_TN = Dict(:X => "X", :Y => "Y", :Z => "Z")

function _pauli_product(a::Symbol, b::Symbol)
    a == :I && return (1.0 + 0.0im, b)
    b == :I && return (1.0 + 0.0im, a)
    a == b && return (1.0 + 0.0im, :I)

    if a == :X && b == :Y
        return (1.0im, :Z)
    elseif a == :Y && b == :X
        return (-1.0im, :Z)
    elseif a == :Y && b == :Z
        return (1.0im, :X)
    elseif a == :Z && b == :Y
        return (-1.0im, :X)
    elseif a == :Z && b == :X
        return (1.0im, :Y)
    elseif a == :X && b == :Z
        return (-1.0im, :Y)
    end

    throw(ArgumentError("Unknown Pauli product $a * $b"))
end

function _notes_pauli_to_code(op::Symbol)
    op == :Z && return (1.0 + 0.0im, :X)
    op == :X && return (-1.0 + 0.0im, :Z)
    op == :Y && return (1.0 + 0.0im, :Y)
    throw(ArgumentError("Expected notes Pauli :X, :Y, or :Z; got $op"))
end

function _apply_notes_pauli!(ops::Vector{Symbol}, site::Int, op::Symbol)
    coeff, code_op = _notes_pauli_to_code(op)
    local_coeff, local_op = _pauli_product(ops[site], code_op)
    ops[site] = local_op
    return coeff * local_coeff
end

function _split_string_pauli_code(n::Int, m::Int, α::Symbol, β::Symbol, N::Int)
    ops = fill(:I, N)
    coeff = 1.0 + 0.0im

    for j in 1:n-1
        coeff *= _apply_notes_pauli!(ops, j, :Z)
    end
    coeff *= _apply_notes_pauli!(ops, n, α)

    for j in 1:m-1
        coeff *= _apply_notes_pauli!(ops, j, :Z)
    end
    coeff *= _apply_notes_pauli!(ops, m, β)

    return coeff, ops
end

function _expect_pauli_string(ψ::MPS, coeff::ComplexF64, ops::Vector{Symbol})
    sites = siteinds(ψ)
    length(sites) == length(ops) || throw(ArgumentError(
        "Pauli string length $(length(ops)) does not match MPS length $(length(sites))"
    ))

    contraction = ITensor(1.0)
    for i in eachindex(ops)
        A = ψ[i]
        s = sites[i]
        O = ops[i] == :I ? op("I", s) : op(_PAULI_LABELS_TN[ops[i]], s)
        contraction *= dag(prime(A)) * O * A
    end

    return coeff * scalar(contraction)
end

function _pauli_string_mpo(sites::Vector{<:Index}, ops::Vector{Symbol})
    length(sites) == length(ops) || throw(ArgumentError(
        "Pauli string length $(length(ops)) does not match site count $(length(sites))"
    ))

    if all(op -> op == :I, ops)
        return MPO(sites, "Id")
    end

    term = Any[1.0]
    for (i, op) in pairs(ops)
        op == :I && continue
        push!(term, _PAULI_LABELS_TN[op], i)
    end

    os = OpSum()
    os += Tuple(term)
    return MPO(os, sites)
end

function _expect_pauli_string(ρ::MPO, coeff::ComplexF64, ops::Vector{Symbol})
    sites = first.(siteinds(ρ; plev=0))
    O = _pauli_string_mpo(sites, ops)
    # ITensors computes inner(ρ, O) as Tr(ρ' O); cooling keeps ρ Hermitian.
    return coeff * inner(ρ, O)
end

function _split_string_correlator(state::Union{MPS,MPO}, n::Int, m::Int, α::Symbol, β::Symbol)
    N = length(state)
    1 <= n <= N || throw(ArgumentError("n=$n outside 1:$N"))
    1 <= m <= N || throw(ArgumentError("m=$m outside 1:$N"))
    coeff, ops = _split_string_pauli_code(n, m, α, β, N)
    return _expect_pauli_string(state, ComplexF64(coeff), ops)
end

function _split_string_correlators(state::Union{MPS,MPO})
    N = length(state)
    Cxx = Matrix{ComplexF64}(undef, N, N)
    Cyy = Matrix{ComplexF64}(undef, N, N)
    Cyx = Matrix{ComplexF64}(undef, N, N)
    Cxy = Matrix{ComplexF64}(undef, N, N)

    for n in 1:N, m in 1:N
        Cxx[n, m] = _split_string_correlator(state, n, m, :X, :X)
        Cyy[n, m] = _split_string_correlator(state, n, m, :Y, :Y)
        Cyx[n, m] = _split_string_correlator(state, n, m, :Y, :X)
        Cxy[n, m] = _split_string_correlator(state, n, m, :X, :Y)
    end

    return (Cxx=Cxx, Cyy=Cyy, Cyx=Cyx, Cxy=Cxy)
end

function _validate_tn_mode_state(state::Union{MPS,MPO}, ham_params::HamiltonianParameters{IsingModel})
    N = ham_params.N
    state_label = state isa MPS ? "MPS" : "MPO"
    length(state) == N || throw(ArgumentError("$state_label length $(length(state)) does not match N=$N"))
    supports_ising_fourier_observables(ham_params) && return nothing
    if !iseven(N)
        throw(ArgumentError("TN mode observables require even N for the Fourier/Bogoliubov grid; got N=$N"))
    end
    throw(ArgumentError(
        "TN mode observables require spin :periodic or :antiperiodic boundary conditions; got $(ham_params.bc)"
    ))
end

"""
    measure_state_parity(ψ::MPS, N::Int) -> Float64

Measure the code-basis Ising parity ``P_x = ∏_i σ^x_i`` on an MPS.
"""
function measure_state_parity(ψ::MPS, N::Int)
    length(ψ) == N || throw(ArgumentError("MPS length $(length(ψ)) does not match N=$N"))
    return real(_expect_pauli_string(ψ, 1.0 + 0.0im, fill(:X, N)))
end

"""
    measure_state_parity(ρ::MPO, N::Int) -> Float64

Measure the code-basis Ising parity ``P_x = ∏_i σ^x_i`` on a density matrix
represented as an MPO.
"""
function measure_state_parity(ρ::MPO, N::Int)
    length(ρ) == N || throw(ArgumentError("MPO length $(length(ρ)) does not match N=$N"))
    return real(_expect_pauli_string(ρ, 1.0 + 0.0im, fill(:X, N)))
end

"""
    _measure_hk_from_correlators(correlators, k, ham_params) -> Float64

Evaluate the Bogoliubov mode observable from precomputed split-string
correlators.
"""
function _measure_hk_from_correlators(correlators, k, ham_params::HamiltonianParameters{IsingModel})
    N = ham_params.N
    J, h = ham_params.params.J, ham_params.params.h
    θ = theta_from_Jh(J, h)
    φk = 2π * Float64(k) / N

    φ_bogo = bogoliubov_angle(Float64(k), θ, N)
    c2 = cos(φ_bogo)^2
    s2 = sin(φ_bogo)^2
    sc = sin(φ_bogo) * cos(φ_bogo)

    nk_sum = 0.0 + 0.0im
    nmk_sum = 0.0 + 0.0im
    pair_sum = 0.0 + 0.0im

    for n in 1:N, m in 1:N
        θnm = (n - m) * φk
        Cxx = correlators.Cxx[n, m]
        Cyy = correlators.Cyy[n, m]
        Cyx = correlators.Cyx[n, m]
        Cxy = correlators.Cxy[n, m]

        adag_a = (Cxx + Cyy + im * (Cyx - Cxy)) / 4
        pairdiff = im * (Cxy + Cyx) / 2

        nk_sum += exp(-im * θnm) * adag_a
        nmk_sum += exp(im * θnm) * adag_a
        pair_sum += exp(-im * θnm) * pairdiff
    end

    nk = nk_sum / N
    nmk = nmk_sum / N
    pair = pair_sum / N

    bogo_nk = c2 * nk + s2 * (1 - nmk) + im * sc * pair
    hk = 2 * bogo_nk - 1
    if abs(imag(hk)) > 1e-8
        @warn "measure_hk(TN): significant imaginary part $(imag(hk)) for k=$k"
    end
    return real(hk)
end

"""
    measure_hk(ψ::MPS, k, ham_params) -> Float64

Measure the Bogoliubov mode observable ``⟨h_k⟩`` from an MPS for the transverse
field Ising model. The implementation evaluates the split-string correlator
formula from `Notes/NotesED/MapToSpin.tex`, using the same conventions as the
ED routine `measure_hk`.

The returned quantity matches the ED convention used in this package: for each
allowed momentum index it is the individual-mode observable
``2\\hat a_k^†\\hat a_k - 1``. The energy decomposition then sums over the full
fermionic grid with the prefactor already used by `measure_all_mode_energies`.
"""
function measure_hk(ψ::MPS, k, ham_params::HamiltonianParameters{IsingModel})
    _validate_tn_mode_state(ψ, ham_params)
    correlators = _split_string_correlators(ψ)
    return _measure_hk_from_correlators(correlators, k, ham_params)
end

"""
    measure_hk(ρ::MPO, k, ham_params) -> Float64

Density-matrix analogue of [`measure_hk(::MPS, k, ham_params)`](@ref). The
same split-string Pauli formula is evaluated as ``Tr(ρ O_k)``.
"""
function measure_hk(ρ::MPO, k, ham_params::HamiltonianParameters{IsingModel})
    _validate_tn_mode_state(ρ, ham_params)
    correlators = _split_string_correlators(ρ)
    return _measure_hk_from_correlators(correlators, k, ham_params)
end

"""
    measure_all_mode_energies(ψ::MPS, ham_params; gF=nothing)

MPS analogue of the ED mode measurement. Returns allowed k-indices, ``⟨h_k⟩``,
and the corresponding positive code-unit quasiparticle gaps used for resonance
diagnostics. Energy reconstruction should use `ising_energy_from_mode_hk`, which
keeps the signed special-mode coefficients.
"""
function measure_all_mode_energies(ψ::MPS, ham_params::HamiltonianParameters{IsingModel};
                                   gF=nothing)
    return _measure_all_mode_energies_tn(ψ, ham_params; gF=gF)
end

function _measure_all_mode_energies_tn(state::Union{MPS,MPO}, ham_params::HamiltonianParameters{IsingModel};
                                       gF=nothing)
    _validate_tn_mode_state(state, ham_params)
    N = ham_params.N

    J, h = ham_params.params.J, ham_params.params.h

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

    correlators = _split_string_correlators(state)
    ks = allowed_k_indices(N, gF)
    hk_values = [_measure_hk_from_correlators(correlators, k, ham_params) for k in ks]
    εk_values = mode_energies_Jh(ks, J, h, N)
    return ks, hk_values, εk_values
end

"""
    measure_all_mode_energies(ρ::MPO, ham_params; gF=nothing)

MPO analogue of the ED mode measurement. Returns allowed k-indices,
``⟨h_k⟩``, and the corresponding positive code-unit quasiparticle gaps used for
resonance diagnostics. Energy reconstruction should use
`ising_energy_from_mode_hk`, which keeps the signed special-mode coefficients.
"""
function measure_all_mode_energies(ρ::MPO, ham_params::HamiltonianParameters{IsingModel};
                                   gF=nothing)
    return _measure_all_mode_energies_tn(ρ, ham_params; gF=gF)
end
