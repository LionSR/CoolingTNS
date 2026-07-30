"""
    native_gate_cooling.jl

Hardware-native Rydberg algorithmic-cooling circuit on **both** the ED and the
TN (MPS/ITensors) backend, in this collaboration's house notation: native
controlled-projector phase gates, global single-qubit rotations, and a
measurement-based ancilla-selective reset (`_measure_and_reset`) -- via the
existing `process_bath_ed_monte_carlo` on ED, and via the existing
`sample_bath!`/`appendzeros_MPS` machinery (`utils_mps.jl`) on TN.

Backend split (CLAUDE.md: one unified file, pure type dispatch, no `*_ed.jl`
duplicates). Everything structural is shared and written exactly once:
`NativeGateCircuitParams`, the interaction graph (`chain_gate_pairs`,
`coupling_gate_pairs`, `two_qubit_gate_pairs`), the `greedy_edge_coloring`
schedule and the gate-count/entangling-depth accounting built on it
(`gate_count_and_depth`, `bsb_gate_count_and_depth`), the collision *schedules*
(`collision_layers`, `bsb_collision_layers`), the layer loop
(`apply_collision`), the noise clock (`noise_passes`/`noise_sites`), and the
MCWF cycle driver (`_run_native_gate_cycles`). Only the genuinely
representation-dependent pieces dispatch:

| concern                | ED                                | TN                                        |
|------------------------|-----------------------------------|-------------------------------------------|
| state                  | `Vector{ComplexF64}` (`2^{2N}`)   | `NativeGateMPS` (MPS + truncation budget) |
| diagonal step          | `DiagonalLayer` (dense `2^{2N}`)  | `NativeDiagonalGateLayer` (window gates)  |
| gate application       | `apply_layer` on the vector       | `apply_layer` via `ITensorMPS.apply`      |
| depolarizing noise     | `apply_depolarizing_ed`           | `apply_depolarizing_noise`                |
| `H_S` / energy         | `SparseMatrixCSC`, `Tr(ρ_sys H_S)`| `MPO`, `inner(ψ', H_S, ψ)`                |
| measurement/reset      | `process_bath_ed_monte_carlo`     | `sample_bath!` + `appendzeros_MPS`        |

Both backends measure the *same* estimator for the headline observable: the
energy is `Tr(ρ_sys H_S)` of the pre-collapse reduced state, exactly (on TN,
`inner(ψ', H_S ⊗ I_bath, ψ)` over the interleaved chain is that trace, with no
sampling noise from the current cycle), so TN and ED trajectory ensembles are
directly comparable -- see the "TN backend reproduces the ED cooling
trajectory" cross-validation in `test/test_native_gate_cooling.jl`.

The TN diagonal step is applied as *window* gates rather than one gate per
interacting pair: in the interleaved layout the chain bonds `(s_i, s_{i+1})`
are next-nearest neighbours (bath site `b_i` sits between them), and letting
`apply` transport sites for a non-adjacent two-site gate is exactly the hazard
that produced the historical interleaved-Trotter bug (see
`build_trotter_circuit_interleaved` in `trotter.jl`). `native_diagonal_windows`
instead fuses `g·n_{s_i}n_{b_i}` and `J·n_{s_i}n_{s_{i+1}}` into one three-site
gate on the contiguous window `(s_i, b_i, s_{i+1})`, which is legitimate
because every diagonal term commutes with every other.

Target Hamiltonian (Eq. model in `AlgoCool2026.tex`, "Eq. \\ref{eq:model}"):
    H_S = J·Σ n_i n_{i+1} + h·ΣX_i,   H_A = (Δ/2)·ΣX_bath,   V = g·Σ n_i n_{A_i}
with `n = (1-Z)/2` the Rydberg-occupation projector -- **not** the clean
Pauli-`IsingModel` (J·ΣZZ + h·ΣX). Correction of history: an earlier resolution
of this point (see `ProposalRydbergCooling/notation_translation.md` item #1,
and GitHub issue #675) claimed the projector Hamiltonian's non-uniform
boundary/bulk longitudinal-Z field (visible on expanding `n_i n_{i+1} =
(1-Z_i-Z_{i+1}+Z_iZ_{i+1})/4`) was a gate-compilation artifact that should be
cancelled by extra local phase pulses, leaving clean ZZ Ising as the "true"
target. That is wrong, and is the *literal mistake the collaborator note warns
against*: "This section chooses the Hamiltonian after specifying the physical
gate. This order avoids the common mistake of calling the hardware pulse an
ideal CZ while ignoring its one-particle phase" (`AlgoCool2026.tex`, Sec.
"Native gate and Hamiltonian"), and explicitly, right after expanding
`n_i n_{i+1}` in Pauli operators: "These longitudinal fields are part of the
target Hamiltonian." Reproducing the collaborator's own N=5 headline numbers
(q_20=0.207, q_25=0.186) requires simulating and measuring energy against
*this* projector Hamiltonian, confirmed numerically to <0.1% agreement once
this fix was made (previously off by ~1.5-2x using the clean-ZZ target).

Native two-qubit gate: the Rydberg controlled-projector phase gate
    CP_ij(γ) = diag(1, 1, 1, e^{iγ}) = exp(iγ n_i n_j)   ("Eq. \\ref{eq:projector}").
The diagonal terms of H_S/V compile *directly* onto this gate with no
compensating single-qubit phase ("Eq. \\ref{eq:compile}"):
    exp(-itJ n_i n_j) = CP_ij(-Jt).
`native_zz_evolution_diag`/`native_local_phase_diag` (below) implement the
*different*, clean-ZZ-targeting identity
    exp(-iθZ_iZ_j) = e^{-iθ} · CP_ij(-4θ) · P_i(2θ) · P_j(2θ)
where `P(φ) = diag(1, e^{iφ})`; this is mathematically true and independently
verified (see the "ZZ compilation identity" test), and is kept as a documented
reference/utility -- it is deliberately *not* used by `collision_layers` or
any other function in this file, since it targets a different Hamiltonian
than the one this file is trying to reproduce.

Controls matching the collaborator note: `r=1` is the coarsest one-slice
(Floquet-kick-like) circuit; `reset_bath=ZeroReset()` the no-reset-sandwich
control; `initial_sys=MaximallyMixedState()` the maximally-mixed-input control;
`run_exact_continuous_trajectory` the exact continuous-collision reference;
and `layers_fn=bsb_collision_layers` the B-S-B short-block mechanism test
(14 gates, depth 4 at N=5, matching the collaborator note exactly -- see
`bsb_gate_count_and_depth`). The purity diagnostic (Tr(ρ_sys²), a secondary
sanity check in the collaborator note, not the headline protocol) is
available via the opt-in `compute_purity=true` keyword on both trajectory
drivers (see `purity_from_matrix`) -- it is genuinely O(8^N) to compute
exactly from a pure state with no cheaper shortcut, so it stays off by
default and costs nothing unless requested, matching the existing
`ground_state=nothing`/`fidelities` opt-in pattern.

`NativeGateCircuitParams.residual_alpha` additionally models an *uncompensated*
residual single-atom phase left over per real Rydberg pulse after imperfect
frame-tracking of the calibrated global `P(alpha)` (see the module docstring's
`exp(-iθZ_iZ_j)` identity above: the compiled `P(2θ)` phases are physically
realized by the Rydberg pulse's own single-particle phase accumulation, which
in a real experiment may not be perfectly calibrated). This mirrors
`residual_alpha_per_pulse` in the collaborator's `reproduce_native_note.py`
(`native_diagonal_phase`) as closely as this codebase's own gate-application
structure allows -- see `native_residual_phase_diag` and `collision_layers`.
`scripts/native_phase_sensitivity_scan.jl` scans it, analogous to the
collaborator note's `data/native_phase_sensitivity.csv`.
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using Random

"""
    NativeGateCircuitParams(N, J, h, Delta, g, r, residual_alpha=0.0)

Parameters of the native-gate cooling circuit for `N` system spins (and `N`
bath ancillas), matching Eq. \ref{eq:model}/\ref{eq:recommended} in
`AlgoCool2026.tex`: `J` is the system chain's number-operator (projector)
coupling `Σ n_i n_{i+1}` (Benjamin's `K`, *not* a Pauli-ZZ coefficient -- no
factor of 4 relative to his note), `h` the system transverse field, `Delta`
the bath X-field strength (`H_A = (Δ/2)ΣX_bath`, Delta = 2·Benjamin's `g`),
`g` the system-bath number-operator coupling `Σ n_i n_{A_i}` (Benjamin's `L`,
likewise un-rescaled), `r` Trotter slices per collision, and an optional
`residual_alpha` uncompensated single-atom phase (rad) left over per real
Rydberg pulse after imperfect calibration/tracking of the global `P(alpha)`
phase (see the module docstring and `native_residual_phase_diag`); `0.0`
(default, and the only value reachable via the 6-argument form below)
reproduces the perfectly-calibrated circuit exactly. At the recommended
operating point (K=1 unit): `J=1.0, h=3.4, Delta=6.8, g=5.4`.
"""
struct NativeGateCircuitParams
    N::Int
    J::Float64
    h::Float64
    Delta::Float64
    g::Float64
    r::Int
    residual_alpha::Float64
end

NativeGateCircuitParams(N::Int, J::Float64, h::Float64, Delta::Float64, g::Float64, r::Int) =
    NativeGateCircuitParams(N, J, h, Delta, g, r, 0.0)

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

"""
    native_residual_phase_diag(residual_alpha, n_pulses, nq) -> Vector{ComplexF64}

Diagonal of the global uncompensated single-atom phase `P(n_pulses·residual_alpha)^{⊗nq}`
accumulated by one native diagonal ('collision') step, mirroring
`residual_alpha_per_pulse` in the collaborator's `reproduce_native_note.py`
(`native_diagonal_phase`) as closely as this codebase's own gate-application
structure allows: `n_pulses` real Rydberg-pulse (entangling-sublayer) events
are folded into one diagonal step (`sublayers_per_diag_step` from
`gate_count_and_depth` -- 3 for N≥3, chain-odd + chain-even + pair, matching
the collaborator note's hardcoded "3" at its N=5 operating point exactly,
rather than re-hardcoding it here), and global illumination means every atom
(system *and* bath) picks up its own untracked `P(residual_alpha)` per pulse,
so one diagonal step accumulates `exp(i·n_pulses·residual_alpha·n_total)`
where `n_total = count_ones(k)` is the number of excited qubits summed over
*every* atom in basis state `k`.
"""
function native_residual_phase_diag(residual_alpha::Float64, n_pulses::Int, nq::Int)
    dim = 1 << nq
    d = ones(ComplexF64, dim)
    residual_alpha == 0.0 && return d
    φ = n_pulses * residual_alpha
    for k in 0:(dim - 1)
        d[k + 1] = cis(φ * count_ones(k))
    end
    return d
end

"""
Native-gate compilation of `exp(-iθZ_iZ_j)` (clean Pauli-ZZ evolution, via
`CP_ij(-4θ)·P_i(2θ)·P_j(2θ)`; see module docstring for the identity and why it
is *not* what `native_chain_diagonal`/`native_pair_diagonal` use). Verified
correct (see the "ZZ compilation identity" test) and kept as a documented
reference/utility, deliberately unused elsewhere in this file.
"""
function native_zz_evolution_diag(θ::Float64, i::Int, j::Int, nq::Int)
    return native_cp_diag(-4θ, i, j, nq) .* native_local_phase_diag(2θ, i, nq) .*
           native_local_phase_diag(2θ, j, nq)
end

"""Site pairs of the `J·Σ n_i n_{i+1}` system chain bonds, in bond order."""
chain_gate_pairs(N::Int) =
    [(interleaved_system_site(i), interleaved_system_site(i + 1)) for i in 1:(N - 1)]

"""Site pairs of the `g·Σ n_i n_{A_i}` system-bath couplings, in spin order."""
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
Accumulate the exponent of `CP_ij(-θ) = exp(-iθ n_i n_j)` (Eq. \ref{eq:projector}/
\ref{eq:compile} in `AlgoCool2026.tex` -- the *direct*, uncompensated native
compilation of the target projector Hamiltonian's diagonal terms; see module
docstring) into `phase` in place: `-θ` where both bits `i,j` are set, `0`
otherwise. Used by `native_chain_diagonal`/`native_pair_diagonal` to sum many
pairs' contributions with a single `O(2^nq)`-sized array and a single `cis.`
at the end, instead of allocating a fresh `2^nq`-sized diagonal per pair and
multiplying them together (prohibitive once `2^nq` and the pair count both
grow -- this dominated the per-cycle cost at N≳10).

A zero angle returns immediately rather than sweeping all `2^nq` amplitudes to
add zero. This is not a micro-optimization: `native_chain_diagonal` and
`native_pair_diagonal` deliberately zero out the *other* interaction family, so
without the early-out every chain-only or pair-only step -- both of
`bsb_collision_layers`' distinct diagonal constructions among them -- would pay
a full exponential pass per edge of a family that contributes nothing.
"""
function _accumulate_projector_phase!(phase::Vector{Float64}, θ::Float64, i::Int, j::Int, nq::Int)
    iszero(θ) && return nothing
    bi, bj = interleaved_bit_position(i), interleaved_bit_position(j)
    for k in 0:(length(phase) - 1)
        if ((k >> bi) & 1 == 1) && ((k >> bj) & 1 == 1)
            phase[k + 1] -= θ
        end
    end
    return nothing
end

"""
    native_projector_diagonal(p, chain_dt, pair_dt, nq) -> Vector{ComplexF64}

Diagonal of the native compilation of
`exp[-i(chain_dt·J·Σ n_i n_{i+1} + pair_dt·g·Σ n_i n_{A_i})]` -- the single
source of truth every ED diagonal step is built from, covering the chain-only
(`pair_dt=0`), pair-only (`chain_dt=0`) and combined-slice cases with one
`O(2^nq)` phase accumulator and one `cis.` at the end (see
`_accumulate_projector_phase!` for why the accumulate-then-exponentiate order
matters), rather than exponentiating each term separately and multiplying the
resulting `2^nq`-sized diagonals together.
"""
function native_projector_diagonal(p::NativeGateCircuitParams, chain_dt::Float64,
                                   pair_dt::Float64, nq::Int)
    phase = zeros(Float64, 1 << nq)
    for (i, j) in chain_gate_pairs(p.N)
        _accumulate_projector_phase!(phase, p.J * chain_dt, i, j, nq)
    end
    for (i, j) in coupling_gate_pairs(p.N)
        _accumulate_projector_phase!(phase, p.g * pair_dt, i, j, nq)
    end
    return cis.(phase)
end

"""Diagonal of the native compilation of `exp(-idt·J·Σ n_i n_{i+1})` (system chain bonds only)."""
native_chain_diagonal(p::NativeGateCircuitParams, dt::Float64, nq::Int) =
    native_projector_diagonal(p, dt, 0.0, nq)

"""Diagonal of the native compilation of `exp(-idt·g·Σ n_i n_{A_i})` (system-bath pairs only)."""
native_pair_diagonal(p::NativeGateCircuitParams, dt::Float64, nq::Int) =
    native_projector_diagonal(p, 0.0, dt, nq)

"""Diagonal of the native compilation of `exp[-idt·(J·Σ n_i n_{i+1} + g·Σ n_i n_{A_i})]` for one Trotter slice."""
native_diagonal_slice(p::NativeGateCircuitParams, dt::Float64, nq::Int) =
    native_projector_diagonal(p, dt, dt, nq)

"""
    NativeDiagonalWindow(sites, pairs)

One contiguous-window factor of a TN diagonal step: `sites` are the chain
positions the gate acts on (contiguous, so `ITensorMPS.apply` never has to
transport sites), and `pairs` are `(a, b, γ)` triples of *local* indices into
`sites` carrying the native `CP(γ) = exp(iγ n_a n_b)` phases fused into that
window. See `native_diagonal_windows` for the decomposition and the module
docstring for why fusing is both necessary (next-nearest-neighbour chain bonds)
and legitimate (all diagonal terms commute).
"""
struct NativeDiagonalWindow
    sites::Vector{Int}
    pairs::Vector{Tuple{Int,Int,Float64}}
end

"""
Drop the zero-angle pairs from a candidate window and shrink `sites` to the
span the surviving pairs actually reach, returning `nothing` if none survive.
This is what lets a pair-only step (`bsb_collision_layers`' `half_pair`) fall
back to cheap two-site gates on `(s_i, b_i)` instead of paying three-site
factorizations for a chain term whose angle is zero.
"""
function _trimmed_window(sites::Vector{Int}, pairs::Vector{Tuple{Int,Int,Float64}})
    kept = filter(t -> !iszero(t[3]), pairs)
    isempty(kept) && return nothing
    span = maximum(t -> max(t[1], t[2]), kept)
    return NativeDiagonalWindow(sites[1:span], kept)
end

"""
    native_diagonal_windows(p, chain_dt, pair_dt) -> Vector{NativeDiagonalWindow}

Contiguous-window decomposition of the same diagonal step
`exp[-i(chain_dt·J·Σ n_i n_{i+1} + pair_dt·g·Σ n_i n_{A_i})]` that
`native_projector_diagonal` builds densely for ED. Window `i` is
`(s_i, b_i, s_{i+1})` and carries both the `g` coupling gate on `(s_i, b_i)`
and the `J` chain gate on `(s_i, s_{i+1})`; the last window is the two-site
`(s_N, b_N)` coupling gate. Every native two-qubit gate of
`two_qubit_gate_pairs` appears exactly once across the returned windows, so
the TN circuit realizes literally the same gate set the ED circuit does and
the same set `gate_count_and_depth` counts.
"""
function native_diagonal_windows(p::NativeGateCircuitParams, chain_dt::Float64, pair_dt::Float64)
    γ_chain, γ_pair = -p.J * chain_dt, -p.g * pair_dt
    windows = NativeDiagonalWindow[]
    for i in 1:p.N
        sites = [interleaved_system_site(i), interleaved_bath_site(i)]
        pairs = Tuple{Int,Int,Float64}[(1, 2, γ_pair)]
        if i < p.N
            push!(sites, interleaved_system_site(i + 1))
            push!(pairs, (1, 3, γ_chain))
        end
        window = _trimmed_window(sites, pairs)
        window === nothing || push!(windows, window)
    end
    return windows
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
build a `Vector{CircuitLayerUnion}`, a small (4-type) `Union` rather than the
abstract `CircuitLayer` -- Julia unboxes small unions in arrays, so this keeps
the hot MCWF loop free of the `Any`-boxing a `Vector{Tuple{Symbol,Any}}` would
otherwise incur, while still giving each layer kind its own dispatch method.

The two global-rotation layers are backend-independent (they carry only an
angle, and `apply_layer` resolves them against whichever state representation
it is handed). Only the diagonal step has a backend-specific realization:
`DiagonalLayer` (ED, a dense `2^{2N}` diagonal) versus
`NativeDiagonalGateLayer` (TN, contiguous-window gates). Which one a schedule
emits is chosen by `native_diagonal_layer`, dispatched on the backend, so
`collision_layers`/`bsb_collision_layers` themselves stay single, shared
definitions.
"""
abstract type CircuitLayer end

"""
Native diagonal (entangling) phase layer; `phase` is a precomputed `2^nq`-length
diagonal, and `sublayers` the number of graph-colored hardware pulse sublayers
(`greedy_edge_coloring` color classes) the layer's gates fire in -- the same
schedule `gate_count_and_depth`/`bsb_gate_count_and_depth` count entangling
depth by, so the depolarizing-noise clock (`noise_passes`) stays consistent
with the reported depth.
"""
struct DiagonalLayer <: CircuitLayer
    phase::Vector{ComplexF64}
    sublayers::Int
end

"""
Native diagonal (entangling) phase layer, TN realization: the same gate set as
the `DiagonalLayer` above, but kept as contiguous-window factors
(`native_diagonal_windows`) plus the per-atom `residual_phase` of that step,
instead of a dense `2^{2N}` diagonal that could never be materialized at the
system sizes this backend exists to reach. `sublayers` has the identical
meaning and feeds the identical `noise_passes` clock.

The windows carry chain positions, not `ITensors.Index` objects: the concrete
site indices are resolved from the state's own MPS at apply time
(`native_diagonal_gates`), so a schedule built by `collision_layers` stays a
pure description of the circuit and never has to be rebuilt when the MPS's
site indices change (they do -- `sample_bath!` consumes bath sites and
`appendzeros_MPS` re-attaches fresh ones every cooling cycle).
"""
struct NativeDiagonalGateLayer <: CircuitLayer
    windows::Vector{NativeDiagonalWindow}
    residual_phase::Float64
    sublayers::Int
end

"""Global `exp(-i(θ/2)ΣX)` rotation on the system register."""
struct SystemRotationLayer <: CircuitLayer
    θ::Float64
end

"""Global `exp(-i(θ/2)ΣX)` rotation on the bath register."""
struct BathRotationLayer <: CircuitLayer
    θ::Float64
end

const CircuitLayerUnion =
    Union{DiagonalLayer,NativeDiagonalGateLayer,SystemRotationLayer,BathRotationLayer}

apply_layer(state::Vector{ComplexF64}, layer::DiagonalLayer, sys_sites::Vector{Int}, bath_sites::Vector{Int}) =
    (state .*= layer.phase; state)
apply_layer(state::Vector{ComplexF64}, layer::SystemRotationLayer, sys_sites::Vector{Int}, bath_sites::Vector{Int}) =
    apply_global_x_rotation!(state, layer.θ, sys_sites)
apply_layer(state::Vector{ComplexF64}, layer::BathRotationLayer, sys_sites::Vector{Int}, bath_sites::Vector{Int}) =
    apply_global_x_rotation!(state, layer.θ, bath_sites)

"""
    native_diagonal_layer(p, backend, chain_dt, pair_dt, residual_pulses, sublayers) -> CircuitLayer

Backend-specific realization of one native diagonal ('collision') step of the
*shared* schedules `collision_layers`/`bsb_collision_layers`: the step
compiling `exp[-i(chain_dt·J·Σ n_i n_{i+1} + pair_dt·g·Σ n_i n_{A_i})]`,
fired in `sublayers` graph-colored hardware sublayers, with `residual_pulses`
real Rydberg pulses' worth of uncompensated single-atom phase
(`p.residual_alpha`; `residual_pulses=0` models the step as
residual-phase-free, which is what the collaborator's own B-S-B construction
does -- see `collision_layers`).
"""
function native_diagonal_layer(p::NativeGateCircuitParams, ::EDBackend, chain_dt::Float64,
                               pair_dt::Float64, residual_pulses::Int, sublayers::Int)
    nq = n_qubits(p)
    phase = native_projector_diagonal(p, chain_dt, pair_dt, nq)
    residual = residual_pulses * p.residual_alpha
    iszero(residual) ||
        (phase = phase .* native_residual_phase_diag(p.residual_alpha, residual_pulses, nq))
    return DiagonalLayer(phase, sublayers)
end

function native_diagonal_layer(p::NativeGateCircuitParams, ::TNBackend, chain_dt::Float64,
                               pair_dt::Float64, residual_pulses::Int, sublayers::Int)
    return NativeDiagonalGateLayer(
        native_diagonal_windows(p, chain_dt, pair_dt),
        residual_pulses * p.residual_alpha,
        sublayers,
    )
end

"""
    noise_passes(layer::CircuitLayer) -> Int

Number of depolarizing passes `apply_collision` applies after `layer`: one per
hardware pulse sublayer. A `DiagonalLayer` fires its gates in
`layer.sublayers` graph-colored entangling sublayers (the unit
`gate_count_and_depth` reports depth in), so it draws that many noise passes;
each global rotation layer is a single pulse.
"""
noise_passes(layer::DiagonalLayer) = layer.sublayers
noise_passes(layer::NativeDiagonalGateLayer) = layer.sublayers
noise_passes(::SystemRotationLayer) = 1
noise_passes(::BathRotationLayer) = 1

"""
    noise_sites(layer::CircuitLayer, registers) -> Vector{Int}

Qubits depolarized per noise pass after `layer`, from `registers =
(sys=..., bath=..., all=...)`: an entangling sublayer's global Rydberg
illumination exposes every atom (`registers.all`), while each global
single-qubit rotation pulses only its own register -- system and bath
rotations act on disjoint atoms and must not double-noise each other's
register.
"""
noise_sites(::DiagonalLayer, registers) = registers.all
noise_sites(::NativeDiagonalGateLayer, registers) = registers.all
noise_sites(::SystemRotationLayer, registers) = registers.sys
noise_sites(::BathRotationLayer, registers) = registers.bath

"""
    collision_layers(p, τ, backend=EDBackend()) -> Vector{CircuitLayerUnion}

Symmetric r-slice Trotter decomposition of `exp[-iτ(H_X + H_D)]` compiled onto
native gates: each slice is a Strang split `X(dt/2) - D(dt) - X(dt/2)`. Since
`apply_global_x_rotation!` implements `exp(-i(θ/2)ΣX)`, the half-pulses need
`θ = h·dt` for the system (H_X_sys = h·ΣX) and `θ = Δ·dt/2` for the bath
(H_X_bath = (Δ/2)·ΣX), each being evaluated at physical time `dt/2`.

If `p.residual_alpha != 0`, each slice's diagonal step additionally picks up
`native_residual_phase_diag(p.residual_alpha, n_pulses, nq)` (see its
docstring), matching the collaborator note's residual-phase modeling scope
exactly: only the recommended r-slice collision models it (`bsb_collision_layers`
does not, since the collaborator's own B-S-B construction, `native14_joint_basis`,
has no `residual_alpha_per_pulse` parameter). `p.residual_alpha == 0.0` (the
default) skips the extra diagonal entirely, reproducing the perfectly-calibrated
circuit bit-for-bit.

The schedule itself -- how many slices, in what order, at what angles -- is
backend-independent and written here exactly once; `backend` selects only how
each diagonal step is realized (`native_diagonal_layer`), so the ED and TN
circuits are the same circuit by construction rather than by two
implementations agreeing.
"""
function collision_layers(p::NativeGateCircuitParams, τ::Float64,
                          backend::CoolingBackend=EDBackend())
    dt = τ / p.r
    θ_sys_half = p.h * dt
    θ_bath_half = p.Delta * dt / 2
    sublayers = gate_count_and_depth(p).sublayers_per_diag_step
    layers = CircuitLayerUnion[]
    for _ in 1:p.r
        push!(layers, SystemRotationLayer(θ_sys_half))
        push!(layers, BathRotationLayer(θ_bath_half))
        push!(layers, native_diagonal_layer(p, backend, dt, dt, sublayers, sublayers))
        push!(layers, SystemRotationLayer(θ_sys_half))
        push!(layers, BathRotationLayer(θ_bath_half))
    end
    return layers
end

"""
    bsb_collision_layers(p, τ, backend=EDBackend()) -> Vector{CircuitLayerUnion}

The collaborator note's "B-S-B" (bath-system-bath) short block: a single-pass
(no r-slicing) Strang-like split around the *coupling* term instead of the
free/X term (contrast `collision_layers`): pair/2 -> chain(full τ) ->
global-X(full τ) -> pair/2. Reverse-engineered from `native14_joint_basis()`
in the collaborator's `reproduce_native_note.py`, which is not derivable from
the note's prose alone. At N=5 this gives 14 native two-qubit gates, entangling
depth 4 (`bsb_gate_count_and_depth`), matching the note's numbers exactly.
"""
function bsb_collision_layers(p::NativeGateCircuitParams, τ::Float64,
                              backend::CoolingBackend=EDBackend())
    pair_sublayers = length(greedy_edge_coloring(coupling_gate_pairs(p.N)))
    chain_sublayers = length(greedy_edge_coloring(chain_gate_pairs(p.N)))
    half_pair = native_diagonal_layer(p, backend, 0.0, τ / 2, 0, pair_sublayers)
    return CircuitLayerUnion[
        half_pair,
        native_diagonal_layer(p, backend, τ, 0.0, 0, chain_sublayers),
        SystemRotationLayer(2τ * p.h),
        BathRotationLayer(τ * p.Delta),
        half_pair,
    ]
end

"""
    NativeGateMPS(psi, sites, maxdim, cutoff, truncation_weight)

TN state of the native-gate circuit: the interleaved `2N`-site MPS together
with the truncation budget every gate application and every reset must honor,
and the discarded weight accumulated so far. This is the TN counterpart of the
ED `Vector{ComplexF64}`, and it is what `apply_layer`/`apply_collision`/
`_measure_and_reset` dispatch on -- carrying `sites`, `maxdim` and `cutoff`
*in the state* is what keeps every one of those shared entry points free of
backend-specific extra arguments.

`sites` is the full interleaved site list, retained because the reset is
destructive: `sample_bath!` returns an MPS on the `N` system sites only, and
`appendzeros_MPS` needs the full `2N`-site list to re-attach the fresh cold
ancillas.

`truncation_weight` is the *cumulative* discarded Schmidt weight since the
state was built. Because every native gate is unitary and the MPS is
renormalized after each diagonal layer, `1 - ‖ψ‖²` measured right after an
`apply` is exactly the weight that `maxdim`/`cutoff` threw away in that layer,
so summing it needs no separate instrumentation of the factorizations.

`max_bond_dim` is likewise a *running maximum over every intermediate state*,
not the bond dimension of the state as it stands. It has to be: the peak is
reached inside a collision, and the ancilla reset then discards it -- once
`sample_bath!` has consumed the bath and `appendzeros_MPS` re-attached a
product one, the surviving bond dimension is only the system MPS's. Reading
the bond dimension off the state at the end of a cycle would therefore report
a number that says nothing about whether `maxdim` was binding, which is the
entire question a large-`N` claim rests on. See
`NativeGateTrajectoryDiagnostics` for reading both back out per cycle.
"""
struct NativeGateMPS{IndexT<:Index}
    psi::MPS
    sites::Vector{IndexT}
    maxdim::Int
    cutoff::Float64
    truncation_weight::Float64
    max_bond_dim::Int
end

"""Fresh `NativeGateMPS` around `psi`: no truncation accumulated yet, and its own bond dimension as the running peak."""
NativeGateMPS(psi::MPS, sites::Vector{<:Index}, maxdim::Int, cutoff::Float64) =
    NativeGateMPS(psi, sites, maxdim, cutoff, 0.0, maxlinkdim(psi))

"""
Default retained bond dimension of a TN native-gate trajectory. Chosen so the
`N ≤ 6` sizes the ED backend also reaches are represented exactly (an
interleaved `2N`-site chain needs at most `2^N` across its middle bond), which
makes the default configuration of the cross-validation tests a genuine
circuit comparison rather than a truncation comparison. Larger runs should
raise it deliberately and check convergence -- see
`scripts/native_gate_tn_scaling.jl`.
"""
const NATIVE_GATE_TN_MAXDIM = 64

"""
Default singular-value cutoff of a TN native-gate trajectory. A discarded
weight `w` costs `O(√w)` in state amplitudes, so `1e-12` keeps a collision's
state error near `1e-6` -- far below the Monte Carlo error of any realistic
trajectory ensemble, while still letting the small Schmidt values that
fine-grained (large `r`, small `dt`) diagonal steps generate be dropped instead
of inflating every bond. Tests that compare TN against ED *amplitude by
amplitude* should pass `cutoff=0.0` so the comparison is limited only by
`maxdim`.
"""
const NATIVE_GATE_TN_CUTOFF = 1e-12

"""Rebuild a `NativeGateMPS` around a new MPS, carrying the budget, adding `discarded` to the accumulated truncation weight and advancing the running peak bond dimension."""
_with_psi(state::NativeGateMPS, psi::MPS, discarded::Float64=0.0) =
    NativeGateMPS(psi, state.sites, state.maxdim, state.cutoff,
                  state.truncation_weight + discarded,
                  max(state.max_bond_dim, maxlinkdim(psi)))

"""
    native_product_mps(sites, amplitudes) -> MPS

Bond-dimension-one MPS with the given one-site `amplitudes` (one length-2
vector per site, in chain order). Used for the interleaved product initial
state; `MPS(sites, ::Vector{String})` cannot express the `|+⟩` and `|X-⟩`
amplitudes this circuit starts from.
"""
function native_product_mps(sites::Vector{<:Index}, amplitudes::Vector{Vector{ComplexF64}})
    n = length(sites)
    length(amplitudes) == n || throw(ArgumentError(
        "native_product_mps needs one amplitude vector per site; got $(length(amplitudes)) for $n sites."))
    links = [Index(1, "Link,l=$j") for j in 1:(n - 1)]
    tensors = Vector{ITensor}(undef, n)
    for j in 1:n
        neighbors = Index[]
        j > 1 && push!(neighbors, links[j - 1])
        j < n && push!(neighbors, links[j])
        T = ITensor(ComplexF64, sites[j], neighbors...)
        for k in 1:dim(sites[j])
            T[sites[j] => k, (l => 1 for l in neighbors)...] = amplitudes[j][k]
        end
        tensors[j] = T
    end
    psi = MPS(tensors)
    normalize!(psi)
    return psi
end

"""
    native_window_gate(window, sites) -> ITensor

Diagonal ITensor of one `NativeDiagonalWindow`: the product of its native
`CP(γ) = exp(iγ n_a n_b)` factors over the window's contiguous sites. Site
index value `k` carries occupation `n = k - 1` (ITensors' `"S=1/2"` states are
ordered `Up, Dn`, and `n = (1-Z)/2` is `0` on `Up`), matching the ED bit
convention (`interleaved_bit_position`) exactly.
"""
function native_window_gate(window::NativeDiagonalWindow, sites::Vector{<:Index})
    idx = [sites[s] for s in window.sites]
    gate = ITensor(ComplexF64, prime.(idx)..., idx...)
    for I in CartesianIndices(ntuple(_ -> 2, length(idx)))
        φ = sum(γ * (I[a] - 1) * (I[b] - 1) for (a, b, γ) in window.pairs; init=0.0)
        gate[(prime(idx[m]) => I[m] for m in eachindex(idx))...,
             (idx[m] => I[m] for m in eachindex(idx))...] = cis(φ)
    end
    return gate
end

"""Single-site `P(φ) = diag(1, e^{iφ})` ITensor -- the TN form of `native_local_phase_diag`."""
native_phase_gate(φ::Float64, s::Index) = ITensor(ComplexF64[1 0; 0 cis(φ)], prime(s), s)

"""Single-site `exp(-i(θ/2)X)` ITensor -- the TN form of `apply_single_site_rotation_x!`."""
native_x_rotation_gate(θ::Float64, s::Index) =
    ITensor(ComplexF64[cos(θ / 2) -im*sin(θ / 2); -im*sin(θ / 2) cos(θ / 2)], prime(s), s)

"""
    native_diagonal_gates(layer, sites) -> Vector{ITensor}

Gate list of one TN diagonal step: the window gates in left-to-right order
(so the MPS orthogonality center sweeps monotonically rightwards), followed by
the global uncompensated single-atom `P(residual_phase)` on *every* atom when
that phase is nonzero -- the same "global illumination touches system and bath
alike" model `native_residual_phase_diag` implements densely for ED.
"""
function native_diagonal_gates(layer::NativeDiagonalGateLayer, sites::Vector{<:Index})
    gates = ITensor[native_window_gate(w, sites) for w in layer.windows]
    iszero(layer.residual_phase) && return gates
    append!(gates, native_phase_gate(layer.residual_phase, s) for s in sites)
    return gates
end

"""
Apply one TN diagonal step. The window gates are the only place the native-gate
circuit can lose norm, so the discarded weight `1 - ‖ψ‖²` is harvested here
(the input is normalized and every gate is unitary) before renormalizing.
"""
function apply_layer(state::NativeGateMPS, layer::NativeDiagonalGateLayer,
                     sys_sites::Vector{Int}, bath_sites::Vector{Int})
    psi = apply(native_diagonal_gates(layer, state.sites), state.psi;
                cutoff=state.cutoff, maxdim=state.maxdim)
    discarded = max(0.0, 1 - norm(psi)^2)
    normalize!(psi)
    return _with_psi(state, psi, discarded)
end

"""Apply `exp(-i(θ/2)ΣX)` to `positions` of a TN state: single-site gates, so exact and bond-dimension-preserving."""
function _apply_tn_x_rotation(state::NativeGateMPS, θ::Float64, positions::Vector{Int})
    iszero(θ) && return state
    gates = ITensor[native_x_rotation_gate(θ, state.sites[j]) for j in positions]
    return _with_psi(state, apply(gates, state.psi; cutoff=state.cutoff, maxdim=state.maxdim))
end

apply_layer(state::NativeGateMPS, layer::SystemRotationLayer, sys_sites::Vector{Int}, bath_sites::Vector{Int}) =
    _apply_tn_x_rotation(state, layer.θ, sys_sites)
apply_layer(state::NativeGateMPS, layer::BathRotationLayer, sys_sites::Vector{Int}, bath_sites::Vector{Int}) =
    _apply_tn_x_rotation(state, layer.θ, bath_sites)

"""
The diagonal step is the one part of a schedule that is backend-specific
(`native_diagonal_layer`), so running a schedule on the other backend's state
is a real programming error rather than an unimplemented combination. These
methods complete the type coverage and say so, instead of leaving a bare
`MethodError` naming two unfamiliar types.
"""
apply_layer(::NativeGateMPS, ::DiagonalLayer, ::Vector{Int}, ::Vector{Int}) = throw(ArgumentError(
    "a dense DiagonalLayer is the ED realization of the diagonal step and cannot be applied " *
    "to an MPS; compile the schedule with collision_layers(p, tau, TNBackend())."))
apply_layer(::Vector{ComplexF64}, ::NativeDiagonalGateLayer, ::Vector{Int}, ::Vector{Int}) = throw(ArgumentError(
    "a NativeDiagonalGateLayer is the TN realization of the diagonal step and cannot be " *
    "applied to an ED state vector; compile the schedule with collision_layers(p, tau, EDBackend())."))

"""
    apply_native_depolarizing(state, p, noise_p, sites, rng) -> state

One depolarizing pass over `sites` (chain positions), dispatched on the state
representation: `apply_depolarizing_ed` on the ED state vector,
`apply_depolarizing_noise` (`noise.jl`) on the TN MPS. Both realize the same
channel -- each listed atom independently receives `X`, `Y` or `Z` with
probability `noise_p/3` each -- so the two backends' noisy ensembles are
comparable even though their individual Pauli draws are not.
"""
apply_native_depolarizing(state::Vector{ComplexF64}, p::NativeGateCircuitParams,
                          noise_p::Float64, sites::Vector{Int}, rng::AbstractRNG) =
    apply_depolarizing_ed(EDStateVector(state, n_qubits(p)), noise_p, sites, rng).data

apply_native_depolarizing(state::NativeGateMPS, p::NativeGateCircuitParams,
                          noise_p::Float64, sites::Vector{Int}, rng::AbstractRNG) =
    _with_psi(state, apply_depolarizing_noise(rng, state.psi, state.sites, sites, noise_p))

"""
    apply_collision(state, p, layers, nq; noise_p=0.0, rng=Random.default_rng()) -> state

Apply one collision's circuit `layers` (from `collision_layers` or
`bsb_collision_layers`) to `state` -- shared by both backends, since the layer
sequence, the noise clock and the register bookkeeping are all
representation-independent; only `apply_layer` and `apply_native_depolarizing`
dispatch on the state. If `noise_p > 0`, each layer is followed by
`noise_passes(layer)` depolarizing passes on `noise_sites(layer, …)` -- one
all-qubit pass per graph-colored entangling sublayer for a diagonal layer (the
same sublayer schedule `gate_count_and_depth` reports entangling depth in, so
noise applications track the printed depth), and one own-register-only pass
per global rotation pulse (system and bath rotations act on disjoint atoms and
do not double-noise each other's register). Pass `rng` (the trajectory's own
RNG) for a reproducible noisy run -- the default matches the previous,
non-reproducible behavior.

`nq` is redundant (it is always `n_qubits(p)`) and retained only so existing
ED call sites keep working; it is validated against `p` rather than used, so a
mismatched value is an error instead of silent nonsense. The ED state vector is
overwritten in place, so callers that still need the input must pass a copy and
always use the returned value.
"""
function apply_collision(
    state, p::NativeGateCircuitParams, layers, nq::Int=n_qubits(p);
    noise_p::Float64=0.0, rng::AbstractRNG=Random.default_rng(),
)
    nq == n_qubits(p) || throw(ArgumentError(
        "apply_collision was given nq=$nq but the circuit has $(n_qubits(p)) qubits."))
    sys_sites = interleaved_system_sites(p.N)
    bath_sites = interleaved_bath_sites(p.N)
    registers = (sys=sys_sites, bath=bath_sites, all=collect(1:nq))
    for layer in layers
        state = apply_layer(state, layer, sys_sites, bath_sites)
        if noise_p > 0
            sites = noise_sites(layer, registers)
            for _ in 1:noise_passes(layer)
                state = apply_native_depolarizing(state, p, noise_p, sites, rng)
            end
        end
    end
    return state
end

"""Ancilla-reset target selecting `bath_reset_amplitudes`, dispatched by type rather than a `Symbol` tag (CLAUDE.md)."""
abstract type ResetTarget end

"""Reset to the cold `|X-⟩^N` bath ground state -- the reset sandwich's intended target."""
struct ColdReset <: ResetTarget end

"""Reset straight to `|0⟩^N` -- the no-reset-sandwich control (not cold for the X-field bath)."""
struct ZeroReset <: ResetTarget end

"""
    bath_reset_amplitudes(target) -> Vector{ComplexF64}

One-ancilla amplitudes of the reset target -- the single source of truth both
backends build their reset from: ED krons it up into a `2^N` vector
(`bath_reset_state`), TN hands it straight to `appendzeros_MPS` as the
one-site bath state, so the two backends cannot drift apart on what "cold"
means. `ColdReset()` is `|X-⟩`, the `bath_ground_state_amplitudes("ZZ")` ground
state of the `(Δ/2)ΣX` ancilla field; `ZeroReset()` is `|0⟩`, the
no-reset-sandwich control (not cold for an X-field bath).
"""
bath_reset_amplitudes(::ColdReset) = ComplexF64.(bath_ground_state_amplitudes("ZZ")[2])
bath_reset_amplitudes(::ZeroReset) = ComplexF64[1, 0]

"""Bath product state selected by `target` (`ColdReset()` -> |X-⟩^N, `ZeroReset()` -> |0⟩^N, the no-sandwich control)."""
bath_reset_state(target::ResetTarget, N::Int) =
    reduce(kron, fill(bath_reset_amplitudes(target), N))

"""N-fold Kronecker product of the ZZ-coupling bath ground state, `bath_ground_state_amplitudes("ZZ")` = |X-⟩."""
bath_ground_state_product(N::Int) = bath_reset_state(ColdReset(), N)

"""N-fold Kronecker product of |0⟩ -- the "no reset sandwich" control's bath state (not cold for an X-field bath)."""
bath_zero_state_product(N::Int) = bath_reset_state(ZeroReset(), N)

"""One-site amplitudes of `|+⟩`, the hot initial system state of the collaborator note."""
_plus_amplitudes() = ComplexF64[1, 1] / sqrt(2)

"""N-fold Kronecker product of |+⟩ -- the hot initial system state of the collaborator note."""
system_plus_state_product(N::Int) = reduce(kron, fill(_plus_amplitudes(), N))

"""
    kron_site_amplitudes(site_amplitudes) -> Vector{ComplexF64}

Dense ED amplitude vector of a product state given one length-2 amplitude
vector per spin, in spin order. Spin `i` occupies bit `i-1` of the ED basis
label (`interleaved_basis_state`), i.e. spin 1 is the *least* significant bit,
so it must be the *last* Kronecker factor -- hence the `reverse`. Both
`initial_system_amplitudes` and its TN counterpart read the same
`initial_system_site_amplitudes`, and this is the one place the two orderings
are reconciled.
"""
kron_site_amplitudes(site_amplitudes::Vector{Vector{ComplexF64}}) =
    reduce(kron, reverse(site_amplitudes))

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

"""
    purity_from_matrix(M) -> Float64

Exact system purity `Tr(ρ_sys²)` from the system⊗bath amplitude matrix `M`
(`system_bath_matrix`), where `ρ_sys = M*M'`. Since `ρ_sys` is Hermitian,
`Tr(ρ_sys²) = ‖ρ_sys‖_F² = ‖M*M'‖_F²`; and since `M` is square (`dim_half ×
dim_half`), `M*M'` and `M'*M` share the same eigenvalues, so `‖M'*M‖_F²` gives
the identical answer. This computes it as one BLAS `M'*M` matrix product
(`dim_half × dim_half`, `O(dim_half³) = O(8^N)` flops -- unavoidable for an
*exact* purity from a pure state, the reason this is opt-in-only) followed by
one `sum(abs2, ·)` Frobenius-norm-squared reduction (`O(dim_half²)`), rather
than an explicit `tr(A*A)`/`tr(A^2)` on `A = M'*M`, which would cost a second
`O(dim_half³)` matrix multiply for no benefit. Never call this by default --
see `compute_purity` on `run_native_gate_trajectory`/`run_exact_continuous_trajectory`.
"""
function purity_from_matrix(M::AbstractMatrix{ComplexF64})
    return real(sum(abs2, M' * M))
end

"""Hot system state `|+⟩^N` with a cold bath -- the default protocol input, i.e. `native_gate_initial_state(rng, N)`."""
initial_state_plus_cold(N::Int) =
    build_interleaved_state(system_plus_state_product(N), bath_ground_state_product(N), N)

"""Initial system-state choice selecting `initial_system_site_amplitudes`, dispatched by type rather than a `Symbol` tag (CLAUDE.md)."""
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

"""
    initial_system_site_amplitudes(sys, N, rng) -> Vector{Vector{ComplexF64}}

One length-2 amplitude vector per system spin, in spin order -- the shared,
representation-free description of the initial system state that ED krons into
a dense vector (`initial_system_amplitudes`) and TN feeds straight into
`native_product_mps`. Both of this file's initial system states are product
states, which is exactly why one description serves both backends.

`MaximallyMixedState` draws its basis label with a single
`rand(rng, 0:(2^N - 1))`, deliberately the same one draw the dense-only
implementation used, so a seeded ED run draws the same basis states from the
same RNG stream as before this refactor.
"""
initial_system_site_amplitudes(::HotState, N::Int, rng::AbstractRNG) =
    [_plus_amplitudes() for _ in 1:N]

function initial_system_site_amplitudes(::MaximallyMixedState, N::Int, rng::AbstractRNG)
    b = rand(rng, 0:(1 << N - 1))
    return [(b >> (i - 1)) & 1 == 1 ? ComplexF64[0, 1] : ComplexF64[1, 0] for i in 1:N]
end

"""Dense `2^N` system amplitudes of the initial state selected by `sys`; see `initial_system_site_amplitudes`."""
initial_system_amplitudes(sys::InitialSystemState, N::Int, rng::AbstractRNG) =
    kron_site_amplitudes(initial_system_site_amplitudes(sys, N, rng))

"""
    native_gate_initial_state(rng, N, backend=EDBackend(); sys=HotState(), bath=ColdReset())
    native_gate_initial_state(rng, N, ::TNBackend, sites; sys, bath, maxdim, cutoff)

Build the full interleaved initial state from an `InitialSystemState` choice
(`sys`) and a `ResetTarget` bath choice (`bath`) -- a dense
`Vector{ComplexF64}` on ED, a `NativeGateMPS` on TN. Both are the same product
state, assembled from the same one-site amplitudes
(`initial_system_site_amplitudes`, `bath_reset_amplitudes`); only the container
differs. The `native_gate_` prefix keeps this module-local helper distinct from
the general `initial_state.jl`/`setup_initial_state` machinery on the package's
exported surface.
"""
function native_gate_initial_state(rng::AbstractRNG, N::Int, ::EDBackend=EDBackend();
                                   sys::InitialSystemState=HotState(), bath::ResetTarget=ColdReset())
    return build_interleaved_state(initial_system_amplitudes(sys, N, rng), bath_reset_state(bath, N), N)
end

function native_gate_initial_state(rng::AbstractRNG, N::Int, ::TNBackend, sites::Vector{<:Index};
                                   sys::InitialSystemState=HotState(), bath::ResetTarget=ColdReset(),
                                   maxdim::Int=NATIVE_GATE_TN_MAXDIM, cutoff::Float64=NATIVE_GATE_TN_CUTOFF)
    length(sites) == interleaved_total_sites(N) || throw(ArgumentError(
        "native_gate_initial_state needs $(interleaved_total_sites(N)) interleaved sites for N=$N, got $(length(sites))."))
    sys_amplitudes = initial_system_site_amplitudes(sys, N, rng)
    bath_amplitude = bath_reset_amplitudes(bath)
    amplitudes = Vector{Vector{ComplexF64}}(undef, interleaved_total_sites(N))
    for i in 1:N
        amplitudes[interleaved_system_site(i)] = sys_amplitudes[i]
        amplitudes[interleaved_bath_site(i)] = bath_amplitude
    end
    return NativeGateMPS(native_product_mps(sites, amplitudes), sites, maxdim, cutoff)
end

"""
Per-cycle tail of every trajectory driver, dispatched on the state
representation: normalize, measure the energy (and the opt-in diagnostics),
then reset the ancillas by measuring the bath register. The ED method below
and the TN method further down have identical signatures and identical
semantics, which is what lets `_run_native_gate_cycles` be written once.

ED: measure the energy against `H_S` via the system-reduced density matrix
`ρ_sys = M*M'`
(`system_bath_matrix`; and the ground-state fidelity if `ground_state` is
given, and the purity `Tr(ρ_sys²)` if `compute_purity` is true -- see
`purity_from_matrix`), then reset the ancillas by measuring the bath register
through the existing `process_bath_ed_monte_carlo`/`measure_ed!`
(`cooling_evolution_ed_shared.jl`, `ed_backend.jl`) -- the same
measurement-based collision-model bath collapse already used by the general ED
MCWF cooling driver, rather than a parallel reimplementation. Measuring either
half of a bipartite pure state in any fixed basis and discarding it reproduces
the same ensemble-averaged reduced density matrix as an eigenbasis-weighted
sample, so this is exact, not an approximation, and (via `measure_ed!`'s
single O(4^N) pass) avoids the O(8^N) full Hermitian eigendecomposition that
dominated the cost at N≳8. Keeping this tail in one place is what makes
`run_native_gate_trajectory` and `run_exact_continuous_trajectory` differ only
in how they propagate one collision.

`compute_purity` defaults to `false` and, when so, does no extra work at all
(not even an `if`-guarded cheap check -- `purity` is simply `NaN`), matching
the `ground_state === nothing` short-circuit already used for `fidelity`: this
diagnostic must never cost anything unless explicitly requested.
"""
function _measure_and_reset(
    state::Vector{ComplexF64}, rng::AbstractRNG, N::Int, H_S::AbstractMatrix,
    bath::Vector{ComplexF64}, ground_state::Union{Nothing,Vector{ComplexF64}};
    compute_purity::Bool=false,
)
    state = state ./ norm(state)
    M = system_bath_matrix(state, N)
    HM = H_S * M
    energy = real(sum(conj(M) .* HM))
    fidelity = ground_state === nothing ? NaN : sum(abs2, ground_state' * M)
    purity = compute_purity ? purity_from_matrix(M) : NaN
    ψ_sys, _ = process_bath_ed_monte_carlo(EDStateVector(state, 2N), N, rng)
    return build_interleaved_state(ψ_sys.data, bath, N), energy, fidelity, purity
end

"""
TN counterpart of the ED `_measure_and_reset` above, with the same signature
and the same physics, built entirely from the TN backend's existing
measurement-based collision-model machinery rather than a parallel
reimplementation: `sample_bath!` (`utils_mps.jl`) collapses and consumes the
ancilla register exactly as the general TN MCWF cooling driver does, and
`appendzeros_MPS` re-attaches a fresh product bath in `bath_amplitudes`. The
collapse is exact for the same reason it is on ED -- measuring either half of a
bipartite pure state in any fixed basis and discarding it reproduces the same
ensemble-averaged reduced density matrix -- and on an MPS it is *cheaper* than
the circuit itself (one right-to-left sweep of rank-1 projections), so the
reset is not the obstacle to large `N` that a naive dense reset would be.

The energy is the identical estimator ED uses: `H_S` is an MPO over the full
interleaved chain acting as `H_S ⊗ I_bath` (`native_projector_system_hamiltonian`
for `TNBackend`), so `inner(ψ', H_S, ψ)` on the *pre-collapse* state is exactly
`Tr(ρ_sys H_S)` -- no sampling noise from the current cycle, and directly
comparable to an ED ensemble cycle by cycle.

The ground-state fidelity, by contrast, is measured *post*-collapse as
`|⟨ψ_0|ψ_sys⟩|²` on the pure collapsed system MPS. Its trajectory average
`Σ_b p_b · |⟨ψ_0|m_b⟩/√p_b|² = Σ_b |⟨ψ_0|m_b⟩|² = ⟨ψ_0|ρ_sys|ψ_0⟩` is exactly
the quantity ED reports, but per trajectory it is a sample rather than the
mean: this is the standard MCWF trade, taken here because the pre-collapse form
would need `|ψ_0⟩⟨ψ_0| ⊗ I_bath` as an MPO, and one MPS-MPS overlap is orders
of magnitude cheaper. Pass `ground_state` as an MPS on the *system* sites
(`sites[1:2:end]`).

`compute_purity` has no TN method by design and raises: an exact `Tr(ρ_sys²)`
needs the full reduced density matrix, which is precisely the `O(4^N)` object
this backend exists to avoid. The available unbiased MPS estimator,
`E[|⟨ψ_sys^{(1)}|ψ_sys^{(2)}⟩|²]` over two *independent* trajectories, is a
cross-trajectory quantity and so cannot be produced by a single-trajectory
driver; it belongs in an ensemble driver if it is ever wanted.
"""
function _measure_and_reset(
    state::NativeGateMPS, rng::AbstractRNG, N::Int, H_S::MPO,
    bath_amplitudes::Vector{ComplexF64}, ground_state::Union{Nothing,MPS};
    compute_purity::Bool=false,
)
    compute_purity && throw(ArgumentError(
        "compute_purity is ED-only: an exact Tr(ρ_sys²) requires the full 2^N × 2^N " *
        "reduced density matrix, which the TN backend exists to avoid. See the " *
        "_measure_and_reset docstring for the two-trajectory MPS estimator."))
    ψ = normalize!(state.psi)
    energy = real(inner(ψ', H_S, ψ))
    _, ψ_sys = sample_bath!(rng, ψ)
    truncate!(ψ_sys; cutoff=state.cutoff, maxdim=state.maxdim)
    discarded = max(0.0, 1 - norm(ψ_sys)^2)
    normalize!(ψ_sys)
    fidelity = ground_state === nothing ? NaN : abs2(inner(ground_state, ψ_sys))
    return _with_psi(state, appendzeros_MPS(ψ_sys, state.sites, bath_amplitudes), discarded),
           energy, fidelity, NaN
end

"""
    native_bath_reset_payload(target, N, backend)

Precomputed per-cycle reset payload `_measure_and_reset` consumes, in whichever
form its backend's reset needs: the dense `2^N` bath vector for ED's
`build_interleaved_state`, the one-site amplitudes for TN's `appendzeros_MPS`.
Hoisting it out of the cycle loop keeps the ED `kron` off the hot path while
letting the loop itself stay backend-agnostic.
"""
native_bath_reset_payload(target::ResetTarget, N::Int, ::EDBackend) = bath_reset_state(target, N)
native_bath_reset_payload(target::ResetTarget, ::Int, ::TNBackend) = bath_reset_amplitudes(target)

"""
    NativeGateTrajectoryDiagnostics()

Opt-in per-cycle record of the TN truncation state of a trajectory. Both series
are *running* quantities read off `NativeGateMPS` (`max_bond_dim`,
`truncation_weight`), so entry `c` is the peak bond dimension reached, and the
total Schmidt weight discarded, up to and including cycle `c` -- not a
per-cycle snapshot, which for the bond dimension would be measured after the
ancilla reset has already thrown the peak away (see `NativeGateMPS`).

Pass one as the `diagnostics` keyword of `run_native_gate_trajectory` to find
out whether a chosen `maxdim` was actually binding -- the question that decides
whether a large-`N` result is converged. Reusing one across several
trajectories accumulates their records, which is what a scan wants: the worst
bond dimension and worst discarded weight over the whole ensemble.
"""
mutable struct NativeGateTrajectoryDiagnostics
    bond_dims::Vector{Int}
    truncation_weights::Vector{Float64}
end

NativeGateTrajectoryDiagnostics() = NativeGateTrajectoryDiagnostics(Int[], Float64[])

"""
    record_diagnostics!(diagnostics, state)

Append one cycle's truncation record, dispatched so that the default
(`nothing`) costs nothing and the ED state -- which has no bond dimension and
discards nothing -- reports that as an error rather than a misleading zero.
"""
record_diagnostics!(::Nothing, state) = nothing

function record_diagnostics!(d::NativeGateTrajectoryDiagnostics, state::NativeGateMPS)
    push!(d.bond_dims, state.max_bond_dim)
    push!(d.truncation_weights, state.truncation_weight)
    return nothing
end

record_diagnostics!(::NativeGateTrajectoryDiagnostics, ::Vector{ComplexF64}) = throw(ArgumentError(
    "bond-dimension/truncation diagnostics are a tensor-network concept; the ED state " *
    "vector is exact, with no bond dimension and no discarded weight."))

"""
    native_projector_system_hamiltonian(N, J, h) -> SparseMatrixCSC{Float64}

The system Hamiltonian actually targeted by the collaborator note (Eq.
\ref{eq:model}): `H_S = J·Σ n_i n_{i+1} + h·ΣX_i`, `n=(1-Z)/2`, on the `N`-spin
system-only Hilbert space -- deliberately *not* the clean Pauli-ZZ
`IsingModel`; see module docstring for why the resulting non-uniform
longitudinal-Z field (visible on Pauli-expanding `n_i n_{i+1}`) is genuine
target physics rather than compilation residue. Same qubit convention as
`pauli_x`/`construct_system_hamiltonian` (spin `i` = bit `i-1`, LSB=spin 1),
so `H_S` here is a drop-in replacement for the old
`construct_system_hamiltonian(IsingModel, ...)` call at every use site
(`_measure_and_reset`'s `Tr(ρ_sys H_S)`, ground-state/fidelity diagnostics).
"""
function native_projector_system_hamiltonian(N::Int, J::Float64, h::Float64)
    dim = 1 << N
    diagE = zeros(Float64, dim)
    for i in 1:(N - 1)
        for k in 0:(dim - 1)
            (((k >> (i - 1)) & 1 == 1) && ((k >> i) & 1 == 1)) && (diagE[k + 1] += J)
        end
    end
    H = spdiagm(0 => diagE)
    for i in 1:N
        H += h .* pauli_x(i, N)
    end
    return H
end
native_projector_system_hamiltonian(p::NativeGateCircuitParams) =
    native_projector_system_hamiltonian(p.N, p.J, p.h)

"""
    native_projector_system_hamiltonian(p, ::EDBackend) -> SparseMatrixCSC{Float64}
    native_projector_system_hamiltonian(p, ::TNBackend, sites, spin_sites=interleaved_system_sites(p.N)) -> MPO

Backend dispatch of the same target `H_S = J·Σ n_i n_{i+1} + h·ΣX_i`.

The TN form is an MPO over `sites`, placing spin `i` of the model at chain
position `spin_sites[i]`. The default places the spins on the odd sites of the
**full interleaved `2N`-site chain**, so the MPO acts as `H_S ⊗ I_bath` and
`inner(ψ', H_S, ψ)` on the pre-collapse system+bath MPS is `Tr(ρ_sys H_S)`
directly -- the same estimator the ED path computes from `ρ_sys = M*M'`, with
no partial trace ever formed. Passing `1:N` over an `N`-site chain instead
gives the system-only MPO to hand to DMRG for the `E_0` reference of a
large-`N` run, from this one definition of the model rather than a second copy
of it (`scripts/native_gate_tn_scaling.jl`).

In ITensors' `"S=1/2"` convention `Z = 2·Sz` and `X = 2·Sx`, so the projector
expands as `n_i n_j = (1/2 - Sz_i)(1/2 - Sz_j) = 1/4 - Sz_i/2 - Sz_j/2 +
Sz_i·Sz_j`; the `J·(N-1)/4` constant that expansion leaves behind is *part of
the target energy* and is carried explicitly as an `"Id"` term, which is what
makes TN and ED energies comparable as absolute numbers rather than only up to
an offset (and is the same non-uniform-longitudinal-field point the module
docstring makes about issue #675).
"""
native_projector_system_hamiltonian(p::NativeGateCircuitParams, ::EDBackend) =
    native_projector_system_hamiltonian(p)

function native_projector_system_hamiltonian(
    p::NativeGateCircuitParams, ::TNBackend, sites::Vector{<:Index},
    spin_sites::Vector{Int}=interleaved_system_sites(p.N),
)
    length(spin_sites) == p.N || throw(ArgumentError(
        "native_projector_system_hamiltonian needs $(p.N) spin positions, got $(length(spin_sites))."))
    maximum(spin_sites; init=0) <= length(sites) || throw(ArgumentError(
        "native_projector_system_hamiltonian spin positions run past the $(length(sites))-site chain."))
    os = OpSum()
    os += p.J * (p.N - 1) / 4, "Id", 1
    for i in 1:(p.N - 1)
        a, b = spin_sites[i], spin_sites[i + 1]
        os += -p.J / 2, "Sz", a
        os += -p.J / 2, "Sz", b
        os += p.J, "Sz", a, "Sz", b
    end
    for a in spin_sites
        os += 2 * p.h, "Sx", a
    end
    return MPO(os, sites)
end

"""
    native_projector_total_hamiltonian(N, J, h, Delta, g) -> SparseMatrixCSC{ComplexF64}

The full system+bath Hamiltonian `H_S + H_A + V` actually targeted by the
collaborator note (`H_S`, `V` from Eq. \ref{eq:model}; `H_A = (Δ/2)·ΣX_bath`,
the already-correct `notation_translation.md` §1 Delta=2·Benjamin's-`g`
translation, untouched by this fix), over the interleaved `2N`-qubit space.
The exact (non-Trotterized) reference this native-gate circuit approximates;
see `exact_collision_operator`.
"""
function native_projector_total_hamiltonian(N::Int, J::Float64, h::Float64, Delta::Float64, g::Float64)
    nq = 2N
    dim = 1 << nq
    diagE = zeros(Float64, dim)
    for (i, j) in chain_gate_pairs(N)
        bi, bj = interleaved_bit_position(i), interleaved_bit_position(j)
        for k in 0:(dim - 1)
            (((k >> bi) & 1 == 1) && ((k >> bj) & 1 == 1)) && (diagE[k + 1] += J)
        end
    end
    for (i, j) in coupling_gate_pairs(N)
        bi, bj = interleaved_bit_position(i), interleaved_bit_position(j)
        for k in 0:(dim - 1)
            (((k >> bi) & 1 == 1) && ((k >> bj) & 1 == 1)) && (diagE[k + 1] += g)
        end
    end
    H = spdiagm(0 => ComplexF64.(diagE))
    for i in interleaved_system_sites(N)
        H += h .* pauli_x(i, nq)
    end
    for i in interleaved_bath_sites(N)
        H += (Delta / 2) .* pauli_x(i, nq)
    end
    return H
end
native_projector_total_hamiltonian(p::NativeGateCircuitParams) =
    native_projector_total_hamiltonian(p.N, p.J, p.h, p.Delta, p.g)

"""System Hamiltonian used when a trajectory driver is called without `H_S`: the
collaborator note's own projector-Ising `H_S` (Eq. \ref{eq:model}), *not* the
clean `IsingModel` -- see module docstring."""
_default_system_hamiltonian(p::NativeGateCircuitParams) = native_projector_system_hamiltonian(p)

"""
    _run_native_gate_cycles(state, p, n_cycles, rng, propagate, H_S, bath, ground_state; kwargs...)

The MCWF cooling loop itself, written once for every backend *and* every
collision propagator: draw `τ`, propagate one collision through `propagate`,
then measure and reset through `_measure_and_reset`. `propagate(state, τ)` is
the only thing that distinguishes the Trotterized native-gate circuit from the
exact continuum reference, and the state's own type is the only thing that
distinguishes ED from TN, so this loop needs to know neither.

The RNG is consumed in a fixed order -- `τ` draw, then any noise draws inside
`propagate`, then the bath measurement -- identical to the order the two
separate ED loops this replaced used, so a seeded ED trajectory sees exactly
the same random stream as before. (Its *numbers* shift by ~1e-16, from
`native_projector_diagonal` now exponentiating a summed phase instead of
multiplying separately exponentiated ones -- strictly better conditioned, but
not bit-for-bit identical.)
"""
function _run_native_gate_cycles(
    state, p::NativeGateCircuitParams, n_cycles::Int, rng::AbstractRNG, propagate,
    H_S, bath, ground_state;
    randomized_tau::Bool=false, tau_max::Float64=0.0, compute_purity::Bool=false,
    diagnostics::Union{Nothing,NativeGateTrajectoryDiagnostics}=nothing,
)
    energies = zeros(n_cycles)
    fidelities = ground_state === nothing ? nothing : zeros(n_cycles)
    purities = compute_purity ? zeros(n_cycles) : nothing
    for c in 1:n_cycles
        τ = randomized_tau ? rand(rng) * tau_max : tau_max
        state = propagate(state, τ)
        state, energies[c], fid, pur = _measure_and_reset(
            state, rng, p.N, H_S, bath, ground_state; compute_purity=compute_purity,
        )
        ground_state !== nothing && (fidelities[c] = fid)
        compute_purity && (purities[c] = pur)
        record_diagnostics!(diagnostics, state)
    end
    return energies, fidelities, purities
end

"""
    run_native_gate_trajectory(p, n_cycles, rng, ::EDBackend=EDBackend(); kwargs...)
    run_native_gate_trajectory(p, n_cycles, rng, ::TNBackend, sites; maxdim, cutoff, kwargs...)

Run one MCWF trajectory: `n_cycles` rounds of (collision, energy/fidelity/purity
measurement, ancilla-selective reset via `_measure_and_reset`), returning
`(energies, fidelities, purities)`. Both backends run the *same* circuit
schedule and report the same energy estimator `Tr(ρ_sys H_S)` per cycle, so
their trajectory ensembles are directly comparable.

`fidelities` is **not** the same estimator on both backends: ED returns the
exact pre-collapse `⟨ψ_0|ρ_sys|ψ_0⟩`, TN a post-collapse per-trajectory sample
`|⟨ψ_0|ψ_sys⟩|²` whose *ensemble mean* is that same quantity. Compare
fidelities across backends only after averaging; single trajectories will not
match (see the TN `_measure_and_reset` docstring for why, and note that the
energy return above does not have this caveat).

On ED, `H_S` should be `SparseMatrixCSC` (as returned by
`construct_system_hamiltonian` for `EDBackend`) so the energy measurement
`Tr(ρ_sys H_S) = Σ_b M[:,b]'*(H_S*M[:,b])` stays O(N·4^N) (sparse H_S times
dense M) rather than densifying to O(8^N).
`ground_state::Union{Nothing,Vector}` enables the ground-state fidelity
`F_0 = |⟨ψ_0|ψ_sys⟩|^2`, matching the collaborator note's Table~2 diagnostic
(pass `nothing`, the default, to skip it).

`compute_purity::Bool=false` enables the secondary purity diagnostic
`Tr(ρ_sys²)` (`purity_from_matrix`), the collaborator note's per-cycle
non-unitality sanity check -- off by default since, unlike the energy and
fidelity measurements above, it is genuinely O(8^N) to compute exactly (no
O(4^N)-or-cheaper shortcut exists for an exact purity from a pure state) and
must never cost anything unless explicitly requested. `purities` is `nothing`
when not requested (matching the `fidelities`/`ground_state=nothing` pattern),
else a `Vector{Float64}` of length `n_cycles`. It is ED-only; see the TN
`_measure_and_reset` docstring.

On TN, `sites` is the interleaved `2N`-site list the trajectory lives on,
`H_S` an `MPO` over those same sites (defaulting to
`native_projector_system_hamiltonian(p, TNBackend(), sites)`), `ground_state`
an `MPS` over the system sites `sites[1:2:end]`, and `maxdim`/`cutoff` the
retained bond dimension and singular-value cutoff. Pass a
`NativeGateTrajectoryDiagnostics()` as `diagnostics` to record the bond
dimension and accumulated discarded weight per cycle -- the evidence needed to
claim a large-`N` run is converged.

Controls matching the collaborator note (see `notation_translation.md`), shared
by both backends: `initial_sys=MaximallyMixedState()` for the
maximally-mixed-input control (default `HotState()`, the note's hot `|+⟩^N`);
`reset_bath=ZeroReset()` for the no-reset-sandwich control -- reset straight to
`|0⟩`, not cold for the X-field bath (default `ColdReset()`, i.e. the reset
sandwich's `|X-⟩`); `layers_fn=bsb_collision_layers` for the B-S-B short-block
mechanism test (default `collision_layers`, the recommended r-slice collision).
Note that `reset_bath` selects the per-cycle reset target only -- the bath
always starts cold.

A custom `layers_fn` should take `(p, τ, backend)` and return a
`Vector{CircuitLayerUnion}` realized for `backend` -- the contract the built-in
schedules follow, with `backend` a defaulted third positional argument. The
older two-argument `(p, τ)` form is still accepted (see
`_backend_aware_schedule`); being unable to see the backend, it can only build
ED layers, so passing one on `TNBackend` raises `apply_layer`'s cross-backend
`ArgumentError`.
"""
function run_native_gate_trajectory(
    p::NativeGateCircuitParams, n_cycles::Int, rng::AbstractRNG, backend::EDBackend=EDBackend();
    H_S::Union{Nothing,AbstractMatrix}=nothing, noise_p::Float64=0.0,
    randomized_tau::Bool=false, tau_max::Float64=0.0,
    ground_state::Union{Nothing,Vector{ComplexF64}}=nothing,
    initial_sys::InitialSystemState=HotState(), reset_bath::ResetTarget=ColdReset(),
    layers_fn::Function=collision_layers, compute_purity::Bool=false,
    diagnostics::Union{Nothing,NativeGateTrajectoryDiagnostics}=nothing,
)
    return _run_native_gate_cycles(
        native_gate_initial_state(rng, p.N, backend; sys=initial_sys, bath=ColdReset()),
        p, n_cycles, rng,
        _native_gate_propagator(p, backend, layers_fn, noise_p, rng),
        H_S === nothing ? _default_system_hamiltonian(p) : H_S,
        native_bath_reset_payload(reset_bath, p.N, backend), ground_state;
        randomized_tau=randomized_tau, tau_max=tau_max,
        compute_purity=compute_purity, diagnostics=diagnostics,
    )
end

function run_native_gate_trajectory(
    p::NativeGateCircuitParams, n_cycles::Int, rng::AbstractRNG, backend::TNBackend,
    sites::Vector{<:Index};
    H_S::Union{Nothing,MPO}=nothing, noise_p::Float64=0.0,
    maxdim::Int=NATIVE_GATE_TN_MAXDIM, cutoff::Float64=NATIVE_GATE_TN_CUTOFF,
    randomized_tau::Bool=false, tau_max::Float64=0.0,
    ground_state::Union{Nothing,MPS}=nothing,
    initial_sys::InitialSystemState=HotState(), reset_bath::ResetTarget=ColdReset(),
    layers_fn::Function=collision_layers, compute_purity::Bool=false,
    diagnostics::Union{Nothing,NativeGateTrajectoryDiagnostics}=nothing,
)
    return _run_native_gate_cycles(
        native_gate_initial_state(rng, p.N, backend, sites; sys=initial_sys,
                                  bath=ColdReset(), maxdim=maxdim, cutoff=cutoff),
        p, n_cycles, rng,
        _native_gate_propagator(p, backend, layers_fn, noise_p, rng),
        H_S === nothing ? native_projector_system_hamiltonian(p, backend, sites) : H_S,
        native_bath_reset_payload(reset_bath, p.N, backend), ground_state;
        randomized_tau=randomized_tau, tau_max=tau_max,
        compute_purity=compute_purity, diagnostics=diagnostics,
    )
end

"""
    _backend_aware_schedule(layers_fn, backend) -> (p, τ) -> layers

Bind `backend` into a schedule callback, accepting both the backend-aware
`(p, τ, backend)` contract the built-in schedules use and the two-argument
`(p, τ)` contract that was `layers_fn`'s public contract before the TN port --
a custom ED schedule written against the old signature keeps working instead of
raising a `MethodError` on the first cycle.

The arity is resolved once here, per trajectory, never inside the cycle loop.
A two-argument schedule can only build ED layers, so on `TNBackend` it will hit
`apply_layer`'s explicit cross-backend `ArgumentError` rather than being
silently reinterpreted -- which is the correct outcome: such a schedule has no
TN realization to offer.
"""
function _backend_aware_schedule(layers_fn::Function, backend::CoolingBackend)
    hasmethod(layers_fn, Tuple{NativeGateCircuitParams,Float64,typeof(backend)}) &&
        return (p, τ) -> layers_fn(p, τ, backend)
    return (p, τ) -> layers_fn(p, τ)
end

"""
    _native_gate_propagator(p, backend, layers_fn, noise_p, rng) -> (state, τ) -> state

One-collision propagator of the Trotterized native-gate circuit: compile the
schedule for this `τ` on `backend` (see `_backend_aware_schedule`), then run it
through the shared `apply_collision`. Returned as a closure so
`_run_native_gate_cycles` never has to know which schedule, backend or noise
level it is driving.
"""
function _native_gate_propagator(p::NativeGateCircuitParams, backend::CoolingBackend,
                                 layers_fn::Function, noise_p::Float64, rng::AbstractRNG)
    schedule = _backend_aware_schedule(layers_fn, backend)
    return (state, τ) -> apply_collision(state, p, schedule(p, τ); noise_p=noise_p, rng=rng)
end

"""
    exact_collision_operator(p) -> (evals, evecs)

Eigendecomposition of the full continuum system+bath Hamiltonian `H_S+H_A+V`
(`native_projector_total_hamiltonian` -- the collaborator note's own projector
Hamiltonian, Eq. \ref{eq:model}, *not* `IsingModel` + `--coupling ZZ`) -- the
Hamiltonian this native-gate circuit approximates via Trotterization. Done
once (O(8^N)) and reused to propagate `exp(-iτH_full)` for many `τ` draws at
O(4^N) each, giving the collaborator note's "exact continuous-collision"
reference (Table 2's exact-collision `q_M` column).
"""
function exact_collision_operator(p::NativeGateCircuitParams)
    H_full = Matrix(native_projector_total_hamiltonian(p))
    return eigen(Hermitian(H_full))
end

"""Propagate `state` by `exp(-iτH_full)` in the eigenbasis from `exact_collision_operator`."""
function _exact_collision_propagate(state::Vector{ComplexF64}, τ::Float64, evals, evecs)
    coeffs = evecs' * state
    coeffs = cis.(-τ .* evals) .* coeffs
    return evecs * coeffs
end

"""
    run_exact_continuous_trajectory(p, n_cycles, rng; kwargs...) -> (energies, fidelities, purities)

Same MCWF collision-model driver as `run_native_gate_trajectory` (identical
`_measure_and_reset` tail, including the opt-in `compute_purity` diagnostic --
see its docstring), but propagating each collision via the exact continuum
Hamiltonian instead of the Trotterized native-gate circuit -- isolates
Trotter/discretization error from everything else (reset mechanism, noise,
controls). Pass `evals, evecs` from a single `exact_collision_operator` call
when running many trajectories, to avoid repeating the O(8^N)
diagonalization.
"""
function run_exact_continuous_trajectory(
    p::NativeGateCircuitParams, n_cycles::Int, rng::AbstractRNG;
    H_S::Union{Nothing,AbstractMatrix}=nothing,
    randomized_tau::Bool=false, tau_max::Float64=0.0,
    ground_state::Union{Nothing,Vector{ComplexF64}}=nothing,
    initial_sys::InitialSystemState=HotState(), reset_bath::ResetTarget=ColdReset(),
    evals=nothing, evecs=nothing, compute_purity::Bool=false,
)
    if evals === nothing || evecs === nothing
        evals, evecs = exact_collision_operator(p)
    end
    backend = EDBackend()
    return _run_native_gate_cycles(
        native_gate_initial_state(rng, p.N, backend; sys=initial_sys, bath=ColdReset()),
        p, n_cycles, rng,
        (state, τ) -> _exact_collision_propagate(state, τ, evals, evecs),
        H_S === nothing ? _default_system_hamiltonian(p) : H_S,
        native_bath_reset_payload(reset_bath, p.N, backend), ground_state;
        randomized_tau=randomized_tau, tau_max=tau_max, compute_purity=compute_purity,
    )
end
