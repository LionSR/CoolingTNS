"""
    native_gate_cooling.jl

Hardware-native Rydberg algorithmic-cooling circuit on the ED backend, in this
collaboration's house notation: native controlled-projector phase gates, global
single-qubit rotations, and a measurement-based ancilla-selective reset via the
existing `process_bath_ed_monte_carlo` (`_measure_and_reset`).

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
(Floquet-kick-like) circuit; `reset_bath=ZeroReset()` the no-reset-sandwich
control; `initial_sys=MaximallyMixedState()` the maximally-mixed-input control;
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

"""
Accumulate the real phase of `native_zz_evolution_diag(θ, i, j, nq)` (i.e.
`CP_ij(-4θ)·P_i(2θ)·P_j(2θ)`'s diagonal phase, before `cis`) into `phase` in
place. Used by `native_chain_diagonal`/`native_pair_diagonal` to sum many
pairs' contributions with a single `O(2^nq)`-sized array and a single `cis.`
at the end, instead of allocating a fresh `2^nq`-sized diagonal per pair and
multiplying them together (prohibitive once `2^nq` and the pair count both
grow -- this dominated the per-cycle cost at N≳10).
"""
function _accumulate_zz_phase!(phase::Vector{Float64}, θ::Float64, i::Int, j::Int, nq::Int)
    bi, bj = interleaved_bit_position(i), interleaved_bit_position(j)
    γ, φ = -4θ, 2θ
    for k in 0:(length(phase) - 1)
        bit_i = (k >> bi) & 1 == 1
        bit_j = (k >> bj) & 1 == 1
        contribution = 0.0
        bit_i && (contribution += φ)
        bit_j && (contribution += φ)
        bit_i && bit_j && (contribution += γ)
        phase[k + 1] += contribution
    end
end

"""Diagonal of the native compilation of `exp(-idt·J·ΣZZ_chain)` (system chain bonds only)."""
function native_chain_diagonal(p::NativeGateCircuitParams, dt::Float64, nq::Int)
    phase = zeros(Float64, 1 << nq)
    for (i, j) in chain_gate_pairs(p.N)
        _accumulate_zz_phase!(phase, p.J * dt, i, j, nq)
    end
    return cis.(phase)
end

"""Diagonal of the native compilation of `exp(-idt·g·ΣZZ_coupling)` (system-bath pairs only)."""
function native_pair_diagonal(p::NativeGateCircuitParams, dt::Float64, nq::Int)
    phase = zeros(Float64, 1 << nq)
    for (i, j) in coupling_gate_pairs(p.N)
        _accumulate_zz_phase!(phase, p.g * dt, i, j, nq)
    end
    return cis.(phase)
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
Circuit layer kinds applied by `apply_collision`, dispatched via `apply_layer`
rather than a `Symbol` tag (CLAUDE.md: type-based dispatch, not string/symbol
branching for method selection). `collision_layers`/`bsb_collision_layers`
build a `Vector{CircuitLayerUnion}`, a small (3-type) `Union` rather than the
abstract `CircuitLayer` -- Julia unboxes small unions in arrays, so this keeps
the hot MCWF loop free of the `Any`-boxing a `Vector{Tuple{Symbol,Any}}` would
otherwise incur, while still giving each layer kind its own dispatch method.
"""
abstract type CircuitLayer end

"""Native diagonal (entangling) phase layer; `phase` is a precomputed `2^nq`-length diagonal."""
struct DiagonalLayer <: CircuitLayer
    phase::Vector{ComplexF64}
end

"""Global `exp(-i(θ/2)ΣX)` rotation on the system register."""
struct SystemRotationLayer <: CircuitLayer
    θ::Float64
end

"""Global `exp(-i(θ/2)ΣX)` rotation on the bath register."""
struct BathRotationLayer <: CircuitLayer
    θ::Float64
end

const CircuitLayerUnion = Union{DiagonalLayer,SystemRotationLayer,BathRotationLayer}

apply_layer(state::Vector{ComplexF64}, layer::DiagonalLayer, sys_sites::Vector{Int}, bath_sites::Vector{Int}) =
    state .* layer.phase
apply_layer(state::Vector{ComplexF64}, layer::SystemRotationLayer, sys_sites::Vector{Int}, bath_sites::Vector{Int}) =
    apply_global_x_rotation!(state, layer.θ, sys_sites)
apply_layer(state::Vector{ComplexF64}, layer::BathRotationLayer, sys_sites::Vector{Int}, bath_sites::Vector{Int}) =
    apply_global_x_rotation!(state, layer.θ, bath_sites)

"""
    collision_layers(p, τ) -> Vector{CircuitLayerUnion}

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
    layers = CircuitLayerUnion[]
    for _ in 1:p.r
        push!(layers, SystemRotationLayer(θ_sys_half))
        push!(layers, BathRotationLayer(θ_bath_half))
        push!(layers, DiagonalLayer(native_diagonal_slice(p, dt, nq)))
        push!(layers, SystemRotationLayer(θ_sys_half))
        push!(layers, BathRotationLayer(θ_bath_half))
    end
    return layers
end

"""
    bsb_collision_layers(p, τ) -> Vector{CircuitLayerUnion}

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
    return CircuitLayerUnion[
        DiagonalLayer(half_pair),
        DiagonalLayer(native_chain_diagonal(p, τ, nq)),
        SystemRotationLayer(2τ * p.h),
        BathRotationLayer(τ * p.Delta),
        DiagonalLayer(half_pair),
    ]
end

"""
    apply_collision(state, p, layers, nq; noise_p=0.0, rng=Random.default_rng()) -> state

Apply one collision's circuit `layers` (from `collision_layers` or
`bsb_collision_layers`) to `state`, applying `apply_depolarizing_ed` noise after
every layer if `noise_p > 0`. Pass `rng` (the trajectory's own RNG) for a
reproducible noisy run -- the default matches the previous, non-reproducible
behavior. The rotation layers overwrite `state` in place, so callers that
still need the input must pass a copy and always use the returned vector.
"""
function apply_collision(
    state::Vector{ComplexF64}, p::NativeGateCircuitParams, layers, nq::Int;
    noise_p::Float64=0.0, rng::AbstractRNG=Random.default_rng(),
)
    sys_sites = interleaved_system_sites(p.N)
    bath_sites = interleaved_bath_sites(p.N)
    for layer in layers
        state = apply_layer(state, layer, sys_sites, bath_sites)
        if noise_p > 0
            state = apply_depolarizing_ed(EDStateVector(state, nq), noise_p, collect(1:nq), rng).data
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

"""Ancilla-reset target selecting `bath_reset_state`, dispatched by type rather than a `Symbol` tag (CLAUDE.md)."""
abstract type ResetTarget end

"""Reset to the cold `|X-⟩^N` bath ground state -- the reset sandwich's intended target."""
struct ColdReset <: ResetTarget end

"""Reset straight to `|0⟩^N` -- the no-reset-sandwich control (not cold for the X-field bath)."""
struct ZeroReset <: ResetTarget end

"""Bath product state selected by `target` (`ColdReset()` -> |X-⟩, `ZeroReset()` -> |0⟩^N, the no-sandwich control)."""
bath_reset_state(::ColdReset, N::Int) = bath_ground_state_product(N)
bath_reset_state(::ZeroReset, N::Int) = bath_zero_state_product(N)

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

"""Initial system-state choice selecting `initial_system_amplitudes`, dispatched by type rather than a `Symbol` tag (CLAUDE.md)."""
abstract type InitialSystemState end

"""The collaborator note's hot initial system state `|+⟩^N`."""
struct HotState <: InitialSystemState end

"""
The "maximally mixed input" control: each MCWF trajectory draws a uniformly
random computational basis state `|b⟩`, so the trajectory average is an
*exact* unraveling of `I/2^N` (`E[|b⟩⟨b|] = I/2^N`), stronger than the note's
stated "Pauli-randomized approximation".
"""
struct MaximallyMixedState <: InitialSystemState end

initial_system_amplitudes(::HotState, N::Int, rng::AbstractRNG) = system_plus_state_product(N)
function initial_system_amplitudes(::MaximallyMixedState, N::Int, rng::AbstractRNG)
    b = rand(rng, 0:(1 << N - 1))
    v = zeros(ComplexF64, 1 << N)
    v[b + 1] = 1
    return v
end

"""
    initial_state(rng, N; sys=HotState(), bath=ColdReset()) -> Vector{ComplexF64}

Build the full interleaved initial state from an `InitialSystemState` choice
(`sys`) and a `ResetTarget` bath choice (`bath`); see `initial_system_amplitudes`
and `bath_reset_state`.
"""
function initial_state(rng::AbstractRNG, N::Int; sys::InitialSystemState=HotState(), bath::ResetTarget=ColdReset())
    return build_interleaved_state(initial_system_amplitudes(sys, N, rng), bath_reset_state(bath, N), N)
end

"""
Shared per-cycle tail of both trajectory drivers: normalize, measure the energy
against `H_S` via the system-reduced density matrix `ρ_sys = M*M'`
(`system_bath_matrix`; and the ground-state fidelity if `ground_state` is
given), then reset the ancillas by measuring the bath register through the
existing `process_bath_ed_monte_carlo`/`measure_ed!` (`cooling_evolution_ed_shared.jl`,
`ed_backend.jl`) -- the same measurement-based collision-model bath collapse
already used by the general ED MCWF cooling driver, rather than a parallel
reimplementation. Measuring either half of a bipartite pure state in any fixed
basis and discarding it reproduces the same ensemble-averaged reduced density
matrix as an eigenbasis-weighted sample, so this is exact, not an
approximation, and (via `measure_ed!`'s single O(4^N) pass) avoids the O(8^N)
full Hermitian eigendecomposition that dominated the cost at N≳8. Keeping this
tail in one place is what makes `run_native_gate_trajectory` and
`run_exact_continuous_trajectory` differ only in how they propagate one collision.
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
    ψ_sys, _ = process_bath_ed_monte_carlo(EDStateVector(state, 2N), N, rng)
    return build_interleaved_state(ψ_sys.data, bath, N), energy, fidelity
end

"""System Hamiltonian used when a trajectory driver is called without `H_S` (sparse `IsingModel`, open BC)."""
function _default_system_hamiltonian(p::NativeGateCircuitParams)
    ham = HamiltonianParameters(IsingModel(), p.N, (J=p.J, h=p.h), :open)
    return construct_system_hamiltonian(ham, EDBackend(), p.N)
end

"""
    run_native_gate_trajectory(p, n_cycles, rng; kwargs...) -> (energies, fidelities)

Run one MCWF trajectory: `n_cycles` rounds of (collision, energy/fidelity
measurement, ancilla-selective reset via `_measure_and_reset`). `H_S` should be
`SparseMatrixCSC` (as returned by `construct_system_hamiltonian` for
`EDBackend`) so the energy measurement `Tr(ρ_sys H_S) = Σ_b M[:,b]'*(H_S*M[:,b])`
stays O(N·4^N) (sparse H_S times dense M) rather than densifying to O(8^N).
`ground_state::Union{Nothing,Vector}` enables the ground-state fidelity
`F_0 = |⟨ψ_0|ψ_sys⟩|^2`, matching the collaborator note's Table~2 diagnostic
(pass `nothing`, the default, to skip it).

Controls matching the collaborator note (see `notation_translation.md`):
`initial_sys=MaximallyMixedState()` for the maximally-mixed-input control
(default `HotState()`, the note's hot `|+⟩^N`); `reset_bath=ZeroReset()` for
the no-reset-sandwich control -- reset straight to `|0⟩`, not cold for the
X-field bath (default `ColdReset()`, i.e. the reset sandwich's `|X-⟩`);
`layers_fn=bsb_collision_layers` for the B-S-B short-block mechanism test
(default `collision_layers`, the recommended r-slice collision). Note that
`reset_bath` selects the per-cycle reset target only -- the bath always
starts cold.
"""
function run_native_gate_trajectory(
    p::NativeGateCircuitParams, n_cycles::Int, rng::AbstractRNG;
    H_S::Union{Nothing,AbstractMatrix}=nothing, noise_p::Float64=0.0,
    randomized_tau::Bool=false, tau_max::Float64=0.0,
    ground_state::Union{Nothing,Vector{ComplexF64}}=nothing,
    initial_sys::InitialSystemState=HotState(), reset_bath::ResetTarget=ColdReset(),
    layers_fn::Function=collision_layers,
)
    nq = n_qubits(p)
    H_S = something(H_S, _default_system_hamiltonian(p))
    state = initial_state(rng, p.N; sys=initial_sys, bath=ColdReset())
    energies = zeros(n_cycles)
    fidelities = ground_state === nothing ? nothing : zeros(n_cycles)
    bath = bath_reset_state(reset_bath, p.N)
    for c in 1:n_cycles
        τ = randomized_tau ? rand(rng) * tau_max : tau_max
        layers = layers_fn(p, τ)
        state = apply_collision(state, p, layers, nq; noise_p=noise_p, rng=rng)
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
    initial_sys::InitialSystemState=HotState(), reset_bath::ResetTarget=ColdReset(),
    evals=nothing, evecs=nothing,
)
    H_S = something(H_S, _default_system_hamiltonian(p))
    if evals === nothing || evecs === nothing
        evals, evecs = exact_collision_operator(p)
    end
    state = initial_state(rng, p.N; sys=initial_sys, bath=ColdReset())
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
