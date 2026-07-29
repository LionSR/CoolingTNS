"""
    native_phase_sensitivity_scan.jl

Residual local-phase sensitivity scan for the hardware-native Rydberg
algorithmic-cooling circuit (see src/native_gate_cooling.jl,
`NativeGateCircuitParams.residual_alpha` and `native_residual_phase_diag`).

Scans an uncompensated, per-real-Rydberg-pulse single-atom phase left over
after imperfect calibration/tracking of the global `P(alpha)` phase (the
"Native two-qubit pulse" identity `exp(-iθZ_iZ_j) = e^{-iθ}·CP_ij(-4θ)·P_i(2θ)·P_j(2θ)`
in the module docstring of native_gate_cooling.jl: the compiled `P(2θ)`
phases are physically realized by the Rydberg pulse's own single-particle
phase accumulation), and reports the resulting cooling-quality degradation.

This reproduces, in house notation (J, h, Delta, g -- see
ProposalRydbergCooling/notation_translation.md), the collaborator's
`residual_alpha_per_pulse` sweep in `reproduce_native_note.py`
(`data/native_phase_sensitivity.csv`, `figures/native_phase_sensitivity.pdf`):
a genuine hardware-calibration-robustness study informing how tightly the
single-particle phase tracked in the global rotation frame would need to be
calibrated for the real experiment. It is complementary to (and independent
of) the depolarizing-noise-channel robustness study already covered by
`noise_p` in `native_gate_floquet_trotter_scan.jl`.

House-notation operating point matches native_gate_floquet_trotter_scan.jl:
J=K/4, h=3.4, Delta=2*g_theirs=6.8, g=L/4=1.35, tau_max=0.54 (K=1 unit), and
r=2 (the collaborator note's recommended two-slice collision).
"""

using CoolingTNS
using LinearAlgebra
using Random
using Printf

const J_, h_, DELTA_, G_, TAU_MAX = 0.25, 3.4, 6.8, 1.35, 0.54
const R_SLICES = 2  # the collaborator note's recommended two-slice collision

"""Normalized residual energy `(E-E0)/(E_init-E0)`: 1 at the hot initial state, 0 at the ground state."""
function normalized_residual_energy(E_traj, E0, E_init)
    return (E_traj .- E0) ./ (E_init - E0)
end

function system_energetics(N::Int)
    ham = HamiltonianParameters(IsingModel(), N, (J=J_, h=h_), :open)
    H_S = CoolingTNS.construct_system_hamiltonian(ham, EDBackend(), N)  # keep sparse
    E0, _, _ = CoolingTNS.find_ground_state(H_S, EDBackend())
    plus = ComplexF64[1, 1] / sqrt(2)
    sys_plus = reduce(kron, fill(plus, N))
    E_init = real(dot(sys_plus, H_S * sys_plus))
    return H_S, E0, E_init
end

function run_batch(p::NativeGateCircuitParams, H_S, n_cycles::Int, n_traj::Int, seed::Int)
    rng = MersenneTwister(seed)
    all_E = zeros(n_traj, n_cycles)
    for t in 1:n_traj
        energies, _ = run_native_gate_trajectory(
            p, n_cycles, rng; H_S=H_S, randomized_tau=true, tau_max=TAU_MAX,
        )
        all_E[t, :] = energies
    end
    return vec(sum(all_E, dims=1)) ./ n_traj
end

"""
    phase_sensitivity_scan(N; n_cycles, n_traj, residual_alphas, early_cycle) -> rows

Scan `residual_alpha` over `residual_alphas` (default: an 9-point grid over
[-0.20, 0.20] rad, analogous to the collaborator note's
`np.linspace(-.20, .20, 17)`), reporting the ensemble-averaged normalized
residual energy at an early diagnostic cycle (`early_cycle`, analogous to the
note's cycle-20 diagnostic) and at the final cycle (analogous to its cycle-25
diagnostic) for each `residual_alpha`. Returns a `Vector{NamedTuple}` of rows
(`residual_alpha`, `normalized_residual_energy_early`,
`normalized_residual_energy_final`) for further processing (e.g. writing to a
file), matching the collaborator note's `native_phase_sensitivity.csv`
columns in house-notation-consistent naming (no `q_20`/`q_25`/bare `alpha`).
"""
function phase_sensitivity_scan(
    N::Int; n_cycles::Int=25, n_traj::Int=200,
    residual_alphas=range(-0.20, 0.20; length=9),
    early_cycle::Int=20,
)
    println("="^78)
    println("Residual local-phase sensitivity scan at N=$N ($(2N) total qubits), r=$R_SLICES")
    println("="^78)
    H_S, E0, E_init = system_energetics(N)
    rows = NamedTuple[]
    for (idx, residual_alpha) in enumerate(residual_alphas)
        p = NativeGateCircuitParams(N, J_, h_, DELTA_, G_, R_SLICES, residual_alpha)
        t0 = time()
        E = run_batch(p, H_S, n_cycles, n_traj, 1000 + idx)
        q = normalized_residual_energy(E, E0, E_init)
        q_early = q[min(early_cycle, n_cycles)]
        q_final = q[end]
        @printf("residual_alpha=%+.4f rad: normalized residual energy at cycle %2d = %.4f, at cycle %2d (final) = %.4f   [%.1fs]\n",
                residual_alpha, min(early_cycle, n_cycles), q_early, n_cycles, q_final, time() - t0)
        push!(rows, (
            residual_alpha=residual_alpha,
            normalized_residual_energy_early=q_early,
            normalized_residual_energy_final=q_final,
        ))
    end
    println()
    return rows
end

phase_sensitivity_scan(4; n_cycles=25, n_traj=250)
