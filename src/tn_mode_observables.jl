"""
    tn_mode_observables.jl

Tensor-network measurements of Ising mode observables.

The formulas follow `Notes/NotesED/MapToSpin.tex`.  The Hamiltonian convention is
`H_code = Λ U H_notes U†` with `U = R_y(π/2)^⊗N`; equivalently, a code-basis
state is rotated to notes coordinates by `U† = R_y(-π/2)^⊗N`.  For tensor-network
measurements we leave the state in the code basis and pull notes-basis Pauli
operators back to code Pauli strings:
`Z_notes → X_code`, `X_notes → -Z_code`, and `Y_notes → Y_code`.

For MPS states the O(N^2) split-string correlators are evaluated by sweeping
cached environments through the Jordan-Wigner string intervals, filling the
four endpoint channels in one coordinated pass. For density matrices the same
Pauli strings are converted to MPOs and contracted as `Tr(ρ O)`.
"""

using ITensors
using ITensorMPS

const _PAULI_LABELS_TN = Dict(:X => "X", :Y => "Y", :Z => "Z")
const _MPS_SPLIT_STRING_CHANNELS = (
    (name=:Cxx, α=:X, β=:X),
    (name=:Cyy, α=:Y, β=:Y),
    (name=:Cyx, α=:Y, β=:X),
    (name=:Cxy, α=:X, β=:Y),
)

function _code_pauli_itensor(s::Index, op_label::Symbol)
    op_label == :I && return op("I", s)
    return op(_PAULI_LABELS_TN[op_label], s)
end

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
        O = _code_pauli_itensor(s, ops[i])
        contraction *= dag(prime(A)) * O * A
    end

    return coeff * scalar(contraction)
end

function _mps_apply_string_transfer(env::ITensor, ψ::MPS, sites, site::Int, op_label::Symbol)
    O = _code_pauli_itensor(sites[site], op_label)
    return (env * (O * dag(ψ[site])')) * ψ[site]
end

function _mps_right_endpoint_value(env::ITensor, ψ::MPS, sites, site::Int, op_label::Symbol)
    lind = commonind(ψ[site], env)
    O = _code_pauli_itensor(sites[site], op_label)
    block = env * ψ[site]
    value = (block * O) * prime(dag(ψ[site]), (sites[site], lind))
    return scalar(value)
end

function _mps_single_site_value(left_env::ITensor, ψ::MPS, sites, site::Int, op_label::Symbol)
    N = length(ψ)
    O = _code_pauli_itensor(sites[site], op_label)

    if N == 1
        value = dag(prime(ψ[site])) * O * ψ[site]
        return scalar(value)
    end

    block = left_env * ψ[site]
    if site < N
        right_link = commonind(ψ[site], ψ[site + 1])
        bra = prime(dag(ψ[site]), !right_link)
    else
        left_link = commonind(ψ[site], ψ[site - 1])
        bra = prime(dag(ψ[site]), (sites[site], left_link))
    end
    value = (block * O) * bra
    return scalar(value)
end

function _pauli_string_mpo(sites::Vector{<:Index}, ops::Vector{Symbol})
    length(sites) == length(ops) || throw(ArgumentError(
        "Pauli string length $(length(ops)) does not match site count $(length(sites))"
    ))

    all(==(:I), ops) && return MPO(sites, "Id")

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

function _split_string_correlators_direct(state::Union{MPS,MPO})
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

function _split_string_channel_data(channel, N::Int)
    α, β = channel.α, channel.β
    coeff_α, A = _notes_pauli_to_code(α)
    coeff_β, B = _notes_pauli_to_code(β)
    endpoint_coeff = ComplexF64(coeff_α * coeff_β)

    diag_local_coeff, diag_op = _pauli_product(A, B)
    upper_local_coeff, upper_left_op = _pauli_product(A, :X)
    lower_local_coeff, lower_left_op = _pauli_product(:X, B)

    return (
        C=Matrix{ComplexF64}(undef, N, N),
        A=A,
        B=B,
        diag_op=diag_op,
        diag_coeff=endpoint_coeff * diag_local_coeff,
        upper_left_op=upper_left_op,
        upper_coeff=endpoint_coeff * upper_local_coeff,
        lower_left_op=lower_left_op,
        lower_coeff=endpoint_coeff * lower_local_coeff,
    )
end

function _mps_string_envs_by_left_op(channels, ψ::MPS, sites, site::Int, left_block::ITensor,
                                     op_field::Symbol)
    envs = Dict{Symbol,ITensor}()
    for channel in channels
        op_label = getproperty(channel, op_field)
        haskey(envs, op_label) && continue
        envs[op_label] = (dag(ψ[site])' * _code_pauli_itensor(sites[site], op_label)) *
            left_block
    end
    return envs
end

function _mps_apply_string_transfer_all!(envs::Dict{Symbol,ITensor}, ψ::MPS, sites, site::Int)
    for op_label in collect(keys(envs))
        envs[op_label] = _mps_apply_string_transfer(envs[op_label], ψ, sites, site, :X)
    end
    return envs
end

function _split_string_correlators_four_sweep(ψ::MPS)
    ψs = ITensorMPS.orthogonalize(ψ, 1)
    Cxx = _split_string_correlator_matrix_mps(ψs, :X, :X)
    Cyy = _split_string_correlator_matrix_mps(ψs, :Y, :Y)
    Cyx = _split_string_correlator_matrix_mps(ψs, :Y, :X)
    Cxy = _split_string_correlator_matrix_mps(ψs, :X, :Y)
    return (Cxx=Cxx, Cyy=Cyy, Cyx=Cyx, Cxy=Cxy)
end

"""
    _split_string_correlator_matrix_mps(ψs, α, β)

Build one ordered split-string correlator matrix from an MPS that has already
been orthogonalized at site 1. The left environments are built explicitly; the
right tail is contracted using the right-orthogonality of sites beyond the
current sweep center.
"""
function _split_string_correlator_matrix_mps(ψs::MPS, α::Symbol, β::Symbol)
    N = length(ψs)
    sites = siteinds(ψs)

    coeff_α, A = _notes_pauli_to_code(α)
    coeff_β, B = _notes_pauli_to_code(β)
    endpoint_coeff = ComplexF64(coeff_α * coeff_β)

    diag_local_coeff, diag_op = _pauli_product(A, B)
    diag_coeff = endpoint_coeff * diag_local_coeff

    C = Matrix{ComplexF64}(undef, N, N)

    left_env = ITensor(1.0)
    p_left = 0

    if N == 1
        C[1, 1] = diag_coeff * _mps_single_site_value(left_env, ψs, sites, 1, diag_op)
        return C
    end

    for i in 1:N-1
        while p_left < i - 1
            p_left += 1
            s = sites[p_left]
            left_env = (left_env * ψs[p_left]) * prime(dag(ψs[p_left]), !s)
        end

        left_block = left_env * ψs[i]
        C[i, i] = diag_coeff * _mps_single_site_value(left_env, ψs, sites, i, diag_op)

        # For i < j, S_i α_i S_j β_j reduces to
        # (α_i Z_i)(Z_{i+1} ... Z_{j-1})β_j in notes coordinates.
        upper_local_coeff, upper_left_op = _pauli_product(A, :X)
        upper_coeff = endpoint_coeff * upper_local_coeff
        upper_env = (dag(ψs[i])' * _code_pauli_itensor(sites[i], upper_left_op)) * left_block
        p_upper = i

        for j in i+1:N
            while p_upper < j - 1
                p_upper += 1
                upper_env = _mps_apply_string_transfer(upper_env, ψs, sites, p_upper, :X)
            end

            C[i, j] = upper_coeff * _mps_right_endpoint_value(upper_env, ψs, sites, j, B)

            if j < N
                p_upper = j
                upper_env = _mps_apply_string_transfer(upper_env, ψs, sites, j, :X)
            end
        end

        # For i < j, the lower-triangular entry C[j,i] has β at the left
        # endpoint and α at the right endpoint; the local order at i is Z_i β_i.
        lower_local_coeff, lower_left_op = _pauli_product(:X, B)
        lower_coeff = endpoint_coeff * lower_local_coeff
        lower_env = (dag(ψs[i])' * _code_pauli_itensor(sites[i], lower_left_op)) * left_block
        p_lower = i

        for j in i+1:N
            while p_lower < j - 1
                p_lower += 1
                lower_env = _mps_apply_string_transfer(lower_env, ψs, sites, p_lower, :X)
            end

            C[j, i] = lower_coeff * _mps_right_endpoint_value(lower_env, ψs, sites, j, A)

            if j < N
                p_lower = j
                lower_env = _mps_apply_string_transfer(lower_env, ψs, sites, j, :X)
            end
        end

        p_left += 1
        s = sites[i]
        left_env = left_block * prime(dag(ψs[i]), !s)
    end

    C[N, N] = diag_coeff * _mps_single_site_value(left_env, ψs, sites, N, diag_op)

    return C
end

function _split_string_correlators_fused_mps(ψ::MPS)
    ψs = ITensorMPS.orthogonalize(ψ, 1)
    N = length(ψs)
    sites = siteinds(ψs)
    channels = ntuple(
        i -> _split_string_channel_data(_MPS_SPLIT_STRING_CHANNELS[i], N),
        length(_MPS_SPLIT_STRING_CHANNELS),
    )

    left_env = ITensor(1.0)

    if N == 1
        for channel in channels
            channel.C[1, 1] = channel.diag_coeff *
                _mps_single_site_value(left_env, ψs, sites, 1, channel.diag_op)
        end
        return (Cxx=channels[1].C, Cyy=channels[2].C, Cyx=channels[3].C, Cxy=channels[4].C)
    end

    for i in 1:N-1
        left_block = left_env * ψs[i]

        for channel in channels
            channel.C[i, i] = channel.diag_coeff *
                _mps_single_site_value(left_env, ψs, sites, i, channel.diag_op)
        end

        # For i < j, S_i α_i S_j β_j reduces to
        # (α_i Z_i)(Z_{i+1} ... Z_{j-1})β_j in notes coordinates. The lower
        # triangle has Z_i β_i at the left endpoint and α_j at the right.
        upper_envs = _mps_string_envs_by_left_op(channels, ψs, sites, i, left_block, :upper_left_op)
        lower_envs = _mps_string_envs_by_left_op(channels, ψs, sites, i, left_block, :lower_left_op)

        for j in i+1:N
            for idx in eachindex(channels)
                channel = channels[idx]
                channel.C[i, j] = channel.upper_coeff *
                    _mps_right_endpoint_value(upper_envs[channel.upper_left_op], ψs, sites, j, channel.B)
                channel.C[j, i] = channel.lower_coeff *
                    _mps_right_endpoint_value(lower_envs[channel.lower_left_op], ψs, sites, j, channel.A)
            end

            if j < N
                _mps_apply_string_transfer_all!(upper_envs, ψs, sites, j)
                _mps_apply_string_transfer_all!(lower_envs, ψs, sites, j)
            end
        end

        s = sites[i]
        left_env = left_block * prime(dag(ψs[i]), !s)
    end

    for channel in channels
        channel.C[N, N] = channel.diag_coeff *
            _mps_single_site_value(left_env, ψs, sites, N, channel.diag_op)
    end

    return (Cxx=channels[1].C, Cyy=channels[2].C, Cyx=channels[3].C, Cxy=channels[4].C)
end

_split_string_correlators(ψ::MPS) = _split_string_correlators_fused_mps(ψ)
_split_string_correlators(ρ::MPO) = _split_string_correlators_direct(ρ)

function _validate_tn_mode_state_length(state::Union{MPS,MPO}, N::Int)
    state_label = state isa MPS ? "MPS" : "MPO"
    length(state) == N || throw(ArgumentError("$state_label length $(length(state)) does not match N=$N"))
    return nothing
end

function _validate_tn_mode_state(state::Union{MPS,MPO}, ham_params::HamiltonianParameters{IsingModel})
    _validate_tn_mode_state_length(state, ham_params.N)
    return require_ising_fourier_observables(ham_params; observable="TN mode observables")
end

function _validate_tn_mode_state(state::Union{MPS,MPO}, ham_params::HamiltonianParameters)
    # Unsupported models should report the observable-domain error before any
    # state-length mismatch. If this guard is ever extended to another model,
    # add model-specific TN measurement methods rather than relying on this
    # generic fallback.
    require_ising_fourier_observables(ham_params; observable="TN mode observables")
    _validate_tn_mode_state_length(state, ham_params.N)
    return nothing
end

function _reject_unsupported_tn_mode_observable(state::Union{MPS,MPO}, ham_params::HamiltonianParameters)
    _validate_tn_mode_state(state, ham_params)
    throw(ArgumentError(
        "TN mode observables require a model-specific measurement implementation " *
        "for $(typeof(ham_params.model)); the shared observable-domain guard " *
        "accepted these Hamiltonian parameters, but no specialized TN method exists"
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

    varphi_bogo = bogoliubov_angle(Float64(k), θ, N)
    c2 = cos(varphi_bogo)^2
    s2 = sin(varphi_bogo)^2
    sc = sin(varphi_bogo) * cos(varphi_bogo)

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
``2\\hat a_k^†\\hat a_k - 1``. The energy reconstruction then uses
`ising_energy_from_mode_hk`, which sums over the full fermionic grid with the
signed special-mode coefficients.
"""
function measure_hk(ψ::MPS, k, ham_params::HamiltonianParameters{IsingModel})
    _validate_tn_mode_state(ψ, ham_params)
    correlators = _split_string_correlators(ψ)
    return _measure_hk_from_correlators(correlators, k, ham_params)
end

function measure_hk(ψ::MPS, k, ham_params::HamiltonianParameters)
    return _reject_unsupported_tn_mode_observable(ψ, ham_params)
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

function measure_hk(ρ::MPO, k, ham_params::HamiltonianParameters)
    return _reject_unsupported_tn_mode_observable(ρ, ham_params)
end

"""
    measure_all_mode_observables(ψ::MPS, ham_params; gF=nothing)

MPS analogue of the ED mode measurement. Returns allowed k-indices, ``⟨h_k⟩``,
and the corresponding positive code-unit quasiparticle gaps used for resonance
diagnostics. Energy reconstruction should use `ising_energy_from_mode_hk`, which
keeps the signed special-mode coefficients.
"""
function measure_all_mode_observables(ψ::MPS, ham_params::HamiltonianParameters{IsingModel};
                                      gF=nothing)
    return _measure_all_mode_observables_tn(ψ, ham_params; gF=gF)
end

function measure_all_mode_observables(ψ::MPS, ham_params::HamiltonianParameters;
                                      gF=nothing)
    return _reject_unsupported_tn_mode_observable(ψ, ham_params)
end

function _measure_all_mode_observables_tn(state::Union{MPS,MPO}, ham_params::HamiltonianParameters{IsingModel};
                                          gF=nothing)
    _validate_tn_mode_state(state, ham_params)
    N = ham_params.N

    J, h = ham_params.params.J, ham_params.params.h

    if isnothing(gF)
        px = measure_state_parity(state, N)
        sector = _reference_parity_sector_with_source(px)
        parity = sector.parity
        if sector.source === :reference
            @warn "measure_all_mode_observables: state has no definite P_x parity " *
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
    measure_all_mode_observables(ρ::MPO, ham_params; gF=nothing)

MPO analogue of the ED mode measurement. Returns allowed k-indices,
``⟨h_k⟩``, and the corresponding positive code-unit quasiparticle gaps used for
resonance diagnostics. Energy reconstruction should use
`ising_energy_from_mode_hk`, which keeps the signed special-mode coefficients.
"""
function measure_all_mode_observables(ρ::MPO, ham_params::HamiltonianParameters{IsingModel};
                                      gF=nothing)
    return _measure_all_mode_observables_tn(ρ, ham_params; gF=gF)
end

function measure_all_mode_observables(ρ::MPO, ham_params::HamiltonianParameters;
                                      gF=nothing)
    return _reject_unsupported_tn_mode_observable(ρ, ham_params)
end

"""
    measure_all_mode_energies(ψ_or_ρ, ham_params; gF=nothing)

Compatibility wrapper for [`measure_all_mode_observables`](@ref).

The historical name is retained for existing callers.  New code should prefer
`measure_all_mode_observables`, because the measured quantity is ``h_k`` and
the returned ``ε_k`` values are positive quasiparticle gaps for resonance
labels, not signed energy-reconstruction coefficients.
"""
measure_all_mode_energies(ψ::MPS, ham_params::HamiltonianParameters{IsingModel};
                          gF=nothing) =
    measure_all_mode_observables(ψ, ham_params; gF=gF)

measure_all_mode_energies(ρ::MPO, ham_params::HamiltonianParameters{IsingModel};
                          gF=nothing) =
    measure_all_mode_observables(ρ, ham_params; gF=gF)

measure_all_mode_energies(ψ::MPS, ham_params::HamiltonianParameters; gF=nothing) =
    measure_all_mode_observables(ψ, ham_params; gF=gF)

measure_all_mode_energies(ρ::MPO, ham_params::HamiltonianParameters; gF=nothing) =
    measure_all_mode_observables(ρ, ham_params; gF=gF)
