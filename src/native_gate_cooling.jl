"""
    native_gate_cooling.jl

Hardware-native Rydberg algorithmic-cooling circuit on the ED backend, in this
collaboration's house notation: native controlled-projector phase gates, global
single-qubit rotations, and a measurement-based ancilla-selective reset
(`sample_bath_ed`).

Target Hamiltonian: `IsingModel` system (J·ΣZZ + h·ΣX, via `construct_system_hamiltonian`)
plus a bath X-field ((Δ/2)·ΣX_bath, the existing `--coupling ZZ` bath convention
whose ground state is `bath_ground_state_amplitudes("ZZ")` = |X-⟩) plus ZZ
system-bath coupling of strength `g` — i.e. exactly what `construct_system_bath_hamiltonian`
already builds for `IsingModel` with `--coupling ZZ`. No new Hamiltonian-level physics.

Native two-qubit gate: the Rydberg controlled-projector phase gate
    CP_ij(γ) = diag(1, 1, 1, e^{iγ})   in the n = (1-Z)/2 ("occupied") basis.
Pure ZZ evolution `exp(-iθZ_iZ_j)` is compiled onto this native gate via
    exp(-iθZ_iZ_j) = e^{-iθ} · CP_ij(-4θ) · P_i(2θ) · P_j(2θ)
where `P(φ) = diag(1, e^{iφ})` is the single-atom local-phase gate. This resolves
`ProposalRydbergCooling/notation_translation.md` item #6: the non-uniform
boundary/bulk longitudinal-Z field in the collaborator note's projector-Ising H_S
is exactly the residual single-qubit phase left behind by CP_ij(γ) when compiling
ZZ evolution — a gate-compilation artifact, not separate target physics. The
target Hamiltonian in house notation is therefore the clean `IsingModel` above.

Controls matching the collaborator note: `r=1` is the coarsest one-slice
(Floquet-kick-like) circuit; `reset_bath=:zero` the no-reset-sandwich control;
`initial_sys=:maximally_mixed` the maximally-mixed-input control;
`run_exact_continuous_trajectory` the exact continuous-collision reference;
and `layers_fn=bsb_collision_layers` the B-S-B short-block mechanism test
(14 gates, depth 4 at N=5, matching the collaborator note exactly -- see
`bsb_gate_count_and_depth`). The purity diagnostic (Tr(ρ_sys²), a secondary
sanity check in the collaborator note, not the headline protocol, and O(8^N)
to compute exactly with no cheaper shortcut) is not covered here.
"""

using LinearAlgebra
using Random

"""
    NativeGateCircuitParams(N, J, h, Delta, g, r)

Parameters of the native-gate cooling circuit for `N` system spins (and `N` bath
ancillas): system Ising couplings `J`, `h`, bath X-field strength `Delta`, ZZ
system-bath coupling `g`, and `r` Trotter slices per collision.
"""
struct NativeGateCircuitParams
    N::Int
    J::Float64
    h::Float64
    Delta::Float64
    g::Float64
    r::Int
end

"""Total qubit count (system + bath) of the interleaved chain."""
n_qubits(p::NativeGateCircuitParams) = interleaved_total_sites(p.N)

"""
    native_cp_diag(γ, i, j, nq) -> Vector{ComplexF64}

Diagonal of the native controlled-projector phase gate `CP_ij(γ) = exp(iγ n_i n_j)`
on sites `i, j` (1-indexed, matching `interleaved_layout.jl`).
"""
function native_cp_diag(γ::Float64, i::Int, j::Int, nq::Int)
    dim = 1 << nq
    bi, bj = interleaved_bit_position(i), interleaved_bit_position(j)
    d = ones(ComplexF64, dim)
    phase = cis(γ)
    for k in 0:(dim - 1)
        if ((k >> bi) & 1 == 1) && ((k >> bj) & 1 == 1)
            d[k + 1] = phase
        end
    end
    return d
end

"""Diagonal of the single-atom local-phase gate `P(φ) = diag(1, e^{iφ})` on site `i`."""
function native_local_phase_diag(φ::Float64, i::Int, nq::Int)
    dim = 1 << nq
    bi = interleaved_bit_position(i)
    d = ones(ComplexF64, dim)
    phase = cis(φ)
    for k in 0:(dim - 1)
        if (k >> bi) & 1 == 1
            d[k + 1] = phase
        end
    end
    return d
end

"""Native-gate compilation of `exp(-iθZ_iZ_j)`; see module docstring for the identity."""
function native_zz_evolution_diag(θ::Float64, i::Int, j::Int, nq::Int)
    return native_cp_diag(-4θ, i, j, nq) .* native_local_phase_diag(2θ, i, nq) .*
           native_local_phase_diag(2θ, j, nq)
end

"""Site pairs of the `J·ΣZZ` system chain bonds, in bond order."""
chain_gate_pairs(N::Int) =
    [(interleaved_system_site(i), interleaved_system_site(i + 1)) for i in 1:(N - 1)]

"""Site pairs of the `g·ΣZZ` system-bath couplings, in spin order."""
coupling_gate_pairs(N::Int) =
    [(interleaved_system_site(i), interleaved_bath_site(i)) for i in 1:N]

"""All native two-qubit gate pairs applied in one diagonal ('collision') phase step."""
two_qubit_gate_pairs(N::Int) = vcat(chain_gate_pairs(N), coupling_gate_pairs(N))

"""
    greedy_edge_coloring(pairs) -> Vector{Vector{Tuple{Int,Int}}}

Greedy proper edge-coloring of the interaction graph: each color class is a set
of vertex-disjoint pairs that can fire as one simultaneous hardware sublayer
(no atom in two gates at once), so the number of classes is the entangling
depth of one diagonal step.
"""
function greedy_edge_coloring(pairs::Vector{Tuple{Int,Int}})
    color_of = Dict{Tuple{Int,Int},Int}()
    used_colors = Dict{Int,Set{Int}}()
    for (i, j) in pairs
        used = union(get(used_colors, i, Set{Int}()), get(used_colors, j, Set{Int}()))
        c = 0
        while c in used
            c += 1
        end
        color_of[(i, j)] = c
        push!(get!(Set{Int}, used_colors, i), c)
        push!(get!(Set{Int}, used_colors, j), c)
    end
    n_colors = isempty(color_of) ? 0 : maximum(values(color_of)) + 1
    classes = [Tuple{Int,Int}[] for _ in 1:n_colors]
    for (edge, c) in color_of
        push!(classes[c + 1], edge)
    end
    return classes
end

"""
    gate_count_and_depth(p) -> NamedTuple

Native two-qubit gate count and entangling depth of one full `collision_layers`
round (`r` Trotter slices), computed from the `greedy_edge_coloring` schedule of
`two_qubit_gate_pairs`.
"""
function gate_count_and_depth(p::NativeGateCircuitParams)
    pairs = two_qubit_gate_pairs(p.N)
    classes = greedy_edge_coloring(pairs)
    return (
        two_qubit_gates_per_round=p.r * length(pairs),
        entangling_depth_per_round=p.r * length(classes),
        gates_per_diag_step=length(pairs),
        sublayers_per_diag_step=length(classes),
        color_classes=classes,
    )
end

"""
    bsb_gate_count_and_depth(p) -> NamedTuple

Native two-qubit gate count and entangling depth of one `bsb_collision_layers`
block: two pair sublayers (N mutually-disjoint gates each, 1 sublayer each)
sandwiching one chain sublayer pair (N-1 gates, 2 sublayers from odd/even
edge-coloring) -- gates = 2N+(N-1) = 3N-1, depth = 1+2+1 = 4 for any N. Matches
the collaborator note's N=5 numbers (14, 4).
"""
function bsb_gate_count_and_depth(p::NativeGateCircuitParams)
    chain = chain_gate_pairs(p.N)
    coupling = coupling_gate_pairs(p.N)
    return (
        two_qubit_gates=2 * length(coupling) + length(chain),
        entangling_depth=2 * length(greedy_edge_coloring(coupling)) +
                         length(greedy_edge_coloring(chain)),
    )
end

"""Diagonal of the native compilation of `exp(-idt·J·ΣZZ_chain)` (system chain bonds only)."""
function native_chain_diagonal(p::NativeGateCircuitParams, dt::Float64, nq::Int)
    d = ones(ComplexF64, 1 << nq)
    for (i, j) in chain_gate_pairs(p.N)
        d .*= native_zz_evolution_diag(p.J * dt, i, j, nq)
    end
    return d
end

"""Diagonal of the native compilation of `exp(-idt·g·ΣZZ_coupling)` (system-bath pairs only)."""
function native_pair_diagonal(p::NativeGateCircuitParams, dt::Float64, nq::Int)
    d = ones(ComplexF64, 1 << nq)
    for (i, j) in coupling_gate_pairs(p.N)
        d .*= native_zz_evolution_diag(p.g * dt, i, j, nq)
    end
    return d
end

"""Diagonal of the native compilation of `exp[-idt·(J·ΣZZ_chain + g·ΣZZ_coupling)]` for one Trotter slice."""
function native_diagonal_slice(p::NativeGateCircuitParams, dt::Float64, nq::Int)
    return native_chain_diagonal(p, dt, nq) .* native_pair_diagonal(p, dt, nq)
end

"""
    apply_single_site_rotation_x!(state, θ, site) -> state

Apply `exp(-i(θ/2)X)` to one site of `state` in place, by pairing the basis
amplitudes that differ in the site's bit (linear in `length(state)`, no matrix
ever constructed). `single_site_operator` is deliberately not used here: it
rebuilds a full embedded operator via `kron` on every call, which is fine for
one-time Hamiltonian construction but prohibitive per-qubit, per-layer,
per-cycle, per-trajectory inside the MCWF loop.
"""
function apply_single_site_rotation_x!(state::Vector{ComplexF64}, θ::Float64, site::Int)
    θ == 0.0 && return state
    a = cos(θ / 2)
    b = -im * sin(θ / 2)
    stride = 1 << interleaved_bit_position(site)
    for block_start in 0:(2 * stride):(length(state) - 1)
        for m in block_start:(block_start + stride - 1)
            i0, i1 = m + 1, m + stride + 1
            x, y = state[i0], state[i1]
            state[i0] = a * x + b * y
            state[i1] = b * x + a * y
        end
    end
    return state
end

"""Apply `exp(-i(θ/2)ΣX)` to `sites` of `state` in place, via sequential single-site rotations (exact, since they commute)."""
function apply_global_x_rotation!(state::Vector{ComplexF64}, θ::Float64, sites::Vector{Int})
    θ == 0.0 && return state
    for site in sites
        apply_single_site_rotation_x!(state, θ, site)
    end
    return state
end

"""
    collision_layers(p, τ) -> Vector{Tuple{Symbol,Any}}

Symmetric r-slice Trotter decomposition of `exp[-iτ(H_X + H_D)]` compiled onto
native gates: each slice is a Strang split `X(dt/2) - D(dt) - X(dt/2)`. Since
`apply_global_x_rotation!` implements `exp(-i(θ/2)ΣX)`, the half-pulses need
`θ = h·dt` for the system (H_X_sys = h·ΣX) and `θ = Δ·dt/2` for the bath
(H_X_bath = (Δ/2)·ΣX), each being evaluated at physical time `dt/2`.
"""
function collision_layers(p::NativeGateCircuitParams, τ::Float64)
    dt = τ / p.r
    θ_sys_half = p.h * dt
    θ_bath_half = p.Delta * dt / 2
    nq = n_qubits(p)
    layers = Tuple{Symbol,Any}[]
    for _ in 1:p.r
        push!(layers, (:x_sys, θ_sys_half))
        push!(layers, (:x_bath, θ_bath_half))
        push!(layers, (:diag, native_diagonal_slice(p, dt, nq)))
        push!(layers, (:x_sys, θ_sys_half))
        push!(layers, (:x_bath, θ_bath_half))
    end
    return layers
end

"""
    bsb_collision_layers(p, τ) -> Vector{Tuple{Symbol,Any}}

The collaborator note's "B-S-B" (bath-system-bath) short block: a single-pass
(no r-slicing) Strang-like split around the *coupling* term instead of the
free/X term (contrast `collision_layers`): pair/2 -> chain(full τ) ->
global-X(full τ) -> pair/2. Reverse-engineered from `native14_joint_basis()`
in the collaborator's `reproduce_native_note.py`, which is not derivable from
the note's prose alone. At N=5 this gives 14 native two-qubit gates, entangling
depth 4 (`bsb_gate_count_and_depth`), matching the note's numbers exactly.
"""
function bsb_collision_layers(p::NativeGateCircuitParams, τ::Float64)
    nq = n_qubits(p)
    half_pair = native_pair_diagonal(p, τ / 2, nq)
    return [
        (:diag, half_pair),
        (:diag, native_chain_diagonal(p, τ, nq)),
        (:x_sys, 2τ * p.h),
        (:x_bath, τ * p.Delta),
        (:diag, half_pair),
    ]
end

"""
    apply_collision(state, p, layers, nq; noise_p=0.0) -> state

Apply one collision's circuit `layers` (from `collision_layers` or
`bsb_collision_layers`) to `state`, applying `apply_depolarizing_ed` noise after
every layer if `noise_p > 0`. The rotation layers overwrite `state` in place, so
callers that still need the input must pass a copy and always use the returned
vector.
"""
function apply_collision(state::Vector{ComplexF64}, p::NativeGateCircuitParams, layers, nq::Int; noise_p::Float64=0.0)
    sys_sites = interleaved_system_sites(p.N)
    bath_sites = interleaved_bath_sites(p.N)
    for (kind, payload) in layers
        if kind === :diag
            state = state .* payload
        elseif kind === :x_sys
            state = apply_global_x_rotation!(state, payload, sys_sites)
        elseif kind === :x_bath
            state = apply_global_x_rotation!(state, payload, bath_sites)
        end
        if noise_p > 0
            state = apply_depolarizing_ed(EDStateVector(state, nq), noise_p, collect(1:nq)).data
        end
    end
    return state
end

"""N-fold Kronecker product of the ZZ-coupling bath ground state, `bath_ground_state_amplitudes("ZZ")` = |X-⟩."""
function bath_ground_state_product(N::Int)
    _, amps = bath_ground_state_amplitudes("ZZ")
    return reduce(kron, fill(ComplexF64.(amps), N))
end

"""N-fold Kronecker product of |0⟩ -- the "no reset sandwich" control's bath state (not cold for an X-field bath)."""
bath_zero_state_product(N::Int) = reduce(kron, fill(ComplexF64[1, 0], N))

"""N-fold Kronecker product of |+⟩ -- the hot initial system state of the collaborator note."""
system_plus_state_product(N::Int) = reduce(kron, fill(ComplexF64[1, 1] / sqrt(2), N))

"""Bath product state selected by `kind` (`:cold` -> |X-⟩ via `bath_ground_state_product`, `:zero` -> |0⟩^N, the no-sandwich control)."""
function bath_reset_state(kind::Symbol, N::Int)
    kind === :cold && return bath_ground_state_product(N)
    kind === :zero && return bath_zero_state_product(N)
    throw(ArgumentError("bath_reset_state: unknown kind $kind (expected :cold or :zero)"))
end

"""Build the full interleaved system+bath state from separate N-qubit system/bath amplitude vectors."""
function build_interleaved_state(sys_amplitudes::Vector{ComplexF64}, bath_amplitudes::Vector{ComplexF64}, N::Int)
    dim_half = 1 << N
    full = zeros(ComplexF64, 1 << (2N))
    for s in 0:(dim_half - 1), b in 0:(dim_half - 1)
        full[interleaved_basis_state(s, b, N) + 1] = sys_amplitudes[s + 1] * bath_amplitudes[b + 1]
    end
    return full
end

"""System⊗bath amplitude matrix `M[s,b] = ⟨s,b|ψ⟩`, so `M*M'` is the system-reduced density matrix."""
function system_bath_matrix(state::Vector{ComplexF64}, N::Int)
    dim_half = 1 << N
    M = zeros(ComplexF64, dim_half, dim_half)
    for s in 0:(dim_half - 1), b in 0:(dim_half - 1)
        M[s + 1, b + 1] = state[interleaved_basis_state(s, b, N) + 1]
    end
    return M
end

"""Hot system state `|+⟩^N` with a cold bath -- the default protocol input, i.e. `initial_state(rng, N)`."""
initial_state_plus_cold(N::Int) =
    build_interleaved_state(system_plus_state_product(N), bath_ground_state_product(N), N)

"""
    initial_state(rng, N; sys=:plus, bath=:cold) -> Vector{ComplexF64}

`sys=:plus` (the default) is the collaborator note's hot initial system state
`|+⟩^N`, as built by `initial_state_plus_cold`. `sys=:maximally_mixed` is the
"maximally mixed input" control: each MCWF trajectory draws a uniformly random
computational basis state `|b⟩`, so the trajectory average is an *exact*
unraveling of `I/2^N` (`E[|b⟩⟨b|] = I/2^N`), stronger than the note's stated
"Pauli-randomized approximation".
"""
function initial_state(rng::AbstractRNG, N::Int; sys::Symbol=:plus, bath::Symbol=:cold)
    sys_amplitudes = if sys === :plus
        system_plus_state_product(N)
    elseif sys === :maximally_mixed
        b = rand(rng, 0:(1 << N - 1))
        v = zeros(ComplexF64, 1 << N)
        v[b + 1] = 1
        v
    else
        throw(ArgumentError("initial_state: unknown sys $sys (expected :plus or :maximally_mixed)"))
    end
    return build_interleaved_state(sys_amplitudes, bath_reset_state(bath, N), N)
end

"""Index drawn from the discrete distribution `probs`, consuming one `rand(rng)`."""
function _sample_categorical(rng::AbstractRNG, probs::AbstractVector{Float64})
    r = rand(rng)
    cum = 0.0
    for (i, pr) in enumerate(probs)
        cum += pr
        r <= cum && return i
    end
    return length(probs)
end

"""
    sample_bath_ed(rng, M) -> (bath_outcome::Int, sys_state::Vector{ComplexF64})
    sample_bath_ed(rng, state, N) -> (bath_outcome::Int, sys_state::Vector{ComplexF64})

Measurement-based collision-model bath sampling for the ED backend, the same
principle as the TN backend's `sample_bath` (`utils_mps.jl`): measure the bath
register in the computational basis and collapse the joint state accordingly,
rather than eigendecomposing the system-reduced density matrix. Measuring
either half of a bipartite pure state in any fixed basis and discarding it
reproduces the *same* ensemble-averaged reduced density matrix as an
eigenbasis-weighted sample (`P(b) = ||M[:,b]||^2`, collapsed state
`M[:,b]/||M[:,b]||`; averaged over `b` this sums to `M*M' = ρ_sys` exactly).
This is O(4^N) instead of O(8^N) for a full Hermitian eigendecomposition, which
dominated the cost at N≳8. The cost is dominated by the `system_bath_matrix`
extraction, so callers that already hold `M` (the energy measurement needs it
anyway) should pass it instead of `state, N`.

The TN backend measures bath sites one at a time (an MPS-specific efficiency
for its bond-dimension-limited tensor structure); here the whole bath register
is measured in one vectorized batch, since a dense ED state has no such
structure to preserve.
"""
function sample_bath_ed(rng::AbstractRNG, M::AbstractMatrix{ComplexF64})
    probs = vec(sum(abs2, M; dims=1))
    b = _sample_categorical(rng, probs)
    return b, M[:, b] ./ sqrt(probs[b])
end

sample_bath_ed(rng::AbstractRNG, state::Vector{ComplexF64}, N::Int) =
    sample_bath_ed(rng, system_bath_matrix(state, N))

"""
Shared per-cycle tail of both trajectory drivers: normalize, measure the energy
against `H_S` (and the ground-state fidelity if `ground_state` is given), then
reset the ancillas via `sample_bath_ed`. Keeping it in one place is what makes
`run_native_gate_trajectory` and `run_exact_continuous_trajectory` differ only
in how they propagate one collision.
"""
function _measure_and_reset(
    state::Vector{ComplexF64}, rng::AbstractRNG, N::Int, H_S::AbstractMatrix,
    bath::Vector{ComplexF64}, ground_state::Union{Nothing,Vector{ComplexF64}},
)
    state = state ./ norm(state)
    M = system_bath_matrix(state, N)
    HM = H_S * M
    energy = real(sum(conj(M) .* HM))
    fidelity = ground_state === nothing ? NaN : sum(abs2, ground_state' * M)
    _, sys_state = sample_bath_ed(rng, M)
    return build_interleaved_state(sys_state, bath, N), energy, fidelity
end

"""System Hamiltonian used when a trajectory driver is called without `H_S` (sparse `IsingModel`, open BC)."""
function _default_system_hamiltonian(p::NativeGateCircuitParams)
    ham = HamiltonianParameters(IsingModel(), p.N, (J=p.J, h=p.h), :open)
    return construct_system_hamiltonian(ham, EDBackend(), p.N)
end

"""
    run_native_gate_trajectory(p, n_cycles, rng; kwargs...) -> (energies, fidelities)

Run one MCWF trajectory: `n_cycles` rounds of (collision, energy/fidelity
measurement, ancilla-selective reset via `sample_bath_ed`). `H_S` should be
`SparseMatrixCSC` (as returned by `construct_system_hamiltonian` for
`EDBackend`) so the energy measurement `Tr(ρ_sys H_S) = Σ_b M[:,b]'*(H_S*M[:,b])`
stays O(N·4^N) (sparse H_S times dense M) rather than densifying to O(8^N).
`ground_state::Union{Nothing,Vector}` enables the ground-state fidelity
`F_0 = |⟨ψ_0|ψ_sys⟩|^2`, matching the collaborator note's Table~2 diagnostic
(pass `nothing`, the default, to skip it).

Controls matching the collaborator note (see `notation_translation.md`):
`initial_sys=:maximally_mixed` for the maximally-mixed-input control (default
`:plus`, the note's hot `|+⟩^N`); `reset_bath=:zero` for the no-reset-sandwich
control -- reset straight to `|0⟩`, not cold for the X-field bath (default
`:cold`, i.e. the reset sandwich's `|X-⟩`); `layers_fn=bsb_collision_layers`
for the B-S-B short-block mechanism test (default `collision_layers`, the
recommended r-slice collision). Note that `reset_bath` selects the per-cycle
reset target only -- the bath always starts cold.
"""
function run_native_gate_trajectory(
    p::NativeGateCircuitParams, n_cycles::Int, rng::AbstractRNG;
    H_S::Union{Nothing,AbstractMatrix}=nothing, noise_p::Float64=0.0,
    randomized_tau::Bool=false, tau_max::Float64=0.0,
    ground_state::Union{Nothing,Vector{ComplexF64}}=nothing,
    initial_sys::Symbol=:plus, reset_bath::Symbol=:cold,
    layers_fn::Function=collision_layers,
)
    nq = n_qubits(p)
    H_S = something(H_S, _default_system_hamiltonian(p))
    state = initial_state(rng, p.N; sys=initial_sys, bath=:cold)
    energies = zeros(n_cycles)
    fidelities = ground_state === nothing ? nothing : zeros(n_cycles)
    bath = bath_reset_state(reset_bath, p.N)
    for c in 1:n_cycles
        τ = randomized_tau ? rand(rng) * tau_max : tau_max
        layers = layers_fn(p, τ)
        state = apply_collision(state, p, layers, nq; noise_p=noise_p)
        state, energies[c], fid = _measure_and_reset(state, rng, p.N, H_S, bath, ground_state)
        ground_state !== nothing && (fidelities[c] = fid)
    end
    return energies, fidelities
end

"""
    exact_collision_operator(p) -> (evals, evecs)

Eigendecomposition of the full continuum system+bath Hamiltonian `H_S+H_bath+V`
(via `construct_system_bath_hamiltonian` for `IsingModel` + `--coupling ZZ`) --
the Hamiltonian this native-gate circuit approximates via Trotterization. Done
once (O(8^N)) and reused to propagate `exp(-iτH_full)` for many `τ` draws at
O(4^N) each, giving the collaborator note's "exact continuous-collision"
reference (Table 2's exact-collision `q_M` column).
"""
function exact_collision_operator(p::NativeGateCircuitParams)
    ham = HamiltonianParameters(IsingModel(), p.N, (J=p.J, h=p.h), :open)
    sites_total = interleaved_total_sites(p.N)
    coupling_params = BasicCouplingParameters("ZZ", p.g, 1, 1.0, p.Delta)
    H_full = Matrix(construct_system_bath_hamiltonian(ham, EDBackend(), sites_total, coupling_params))
    return eigen(Hermitian(H_full))
end

"""Propagate `state` by `exp(-iτH_full)` in the eigenbasis from `exact_collision_operator`."""
function _exact_collision_propagate(state::Vector{ComplexF64}, τ::Float64, evals, evecs)
    coeffs = evecs' * state
    coeffs = cis.(-τ .* evals) .* coeffs
    return evecs * coeffs
end

"""
    run_exact_continuous_trajectory(p, n_cycles, rng; kwargs...) -> (energies, fidelities)

Same MCWF collision-model driver as `run_native_gate_trajectory` (identical
`_measure_and_reset` tail), but propagating each collision via the exact
continuum Hamiltonian instead of the Trotterized native-gate circuit --
isolates Trotter/discretization error from everything else (reset mechanism,
noise, controls). Pass `evals, evecs` from a single `exact_collision_operator`
call when running many trajectories, to avoid repeating the O(8^N)
diagonalization.
"""
function run_exact_continuous_trajectory(
    p::NativeGateCircuitParams, n_cycles::Int, rng::AbstractRNG;
    H_S::Union{Nothing,AbstractMatrix}=nothing,
    randomized_tau::Bool=false, tau_max::Float64=0.0,
    ground_state::Union{Nothing,Vector{ComplexF64}}=nothing,
    initial_sys::Symbol=:plus, reset_bath::Symbol=:cold,
    evals=nothing, evecs=nothing,
)
    H_S = something(H_S, _default_system_hamiltonian(p))
    if evals === nothing || evecs === nothing
        evals, evecs = exact_collision_operator(p)
    end
    state = initial_state(rng, p.N; sys=initial_sys, bath=:cold)
    energies = zeros(n_cycles)
    fidelities = ground_state === nothing ? nothing : zeros(n_cycles)
    bath = bath_reset_state(reset_bath, p.N)
    for c in 1:n_cycles
        τ = randomized_tau ? rand(rng) * tau_max : tau_max
        state = _exact_collision_propagate(state, τ, evals, evecs)
        state, energies[c], fid = _measure_and_reset(state, rng, p.N, H_S, bath, ground_state)
        ground_state !== nothing && (fidelities[c] = fid)
    end
    return energies, fidelities
end
