"""
    native_gate_tn_scaling.jl

Large-`N` demonstration of the TN (MPS) backend of the hardware-native Rydberg
algorithmic-cooling circuit (see `src/native_gate_cooling.jl`), i.e. the point
of the port: the ED backend stores a dense `2^{2N}` state vector and is done at
`N ≈ 12`, while the same circuit -- same schedule, same native `CP` gates, same
measurement-based ancilla reset -- runs on an interleaved `2N`-site MPS at
sizes the rest of this project's scaling studies operate at.

House-notation parameters are the direct translation of the collaborator note's
recommended operating point (h/K=g_theirs/K=3.4, L/K=5.4, K*tau_m~Uniform[0,0.54]):
J=K=1.0, h=3.4, Delta=2*g_theirs=6.8, g=L=5.4, tau_max=0.54 (K=1 unit). `J`/`g`
are the number-operator (projector) couplings of Eq. \\ref{eq:model} directly,
*not* Pauli-ZZ coefficients -- see `src/native_gate_cooling.jl`'s module
docstring and `ProposalRydbergCooling/notation_translation.md` (issue #675).

Three sections:

1. **Cross-check** at `N = 4, 6`, where both backends are cheap: the same
   ensemble run on ED and on TN, reported side by side with Monte Carlo error
   bars. This is the script-level echo of the `test_native_gate_cooling.jl`
   cross-validation, so a scaling run always carries its own evidence that the
   TN numbers mean the same thing the ED numbers do.
2. **Scaling** across `N`, reporting the cooled relative residual
   `q_M = (E_M - E_0)/(E_init - E_0)`, wall-clock per trajectory, the MPS bond
   dimension actually reached, and the cumulative discarded Schmidt weight
   (`NativeGateTrajectoryDiagnostics`).
3. **Bond-dimension convergence** at the largest `N`, which is what turns a
   large-`N` number into a *claim*: if `q_M` is unchanged between `maxdim`
   values and the reached bond dimension stays below the cap, the truncation is
   not what set the answer.

`E_0` comes from DMRG on the system-only MPO built from the same
`native_projector_system_hamiltonian` definition the trajectory measures
against (`spin_sites = 1:N`), so no second copy of the model exists to drift.
`E_init = ⟨+|^N H_S |+⟩^N = J(N-1)/4 + hN` exactly, since `⟨+|n|+⟩ = 1/2` and
`⟨+|X|+⟩ = 1` on a product state.

Usage:
    julia --project=. scripts/native_gate_tn_scaling.jl
"""

using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using Random

const J_, h_, DELTA_, G_, TAU_MAX = 1.0, 3.4, 6.8, 5.4, 0.54
const R_SLICES = 2
const N_CYCLES = 25

"""Exact `⟨+|^N H_S |+⟩^N`: `⟨n_i n_{i+1}⟩ = 1/4` and `⟨X⟩ = 1` on a product `|+⟩` state."""
initial_energy(N::Int) = J_ * (N - 1) / 4 + h_ * N

"""Ground energy of `H_S` on `N` spins: exact below `ed_limit`, DMRG on the system-only MPO above it."""
function ground_energy(p::NativeGateCircuitParams; ed_limit::Int=12)
    p.N <= ed_limit && return CoolingTNS.find_ground_state(
        native_projector_system_hamiltonian(p, EDBackend()), EDBackend())[1]
    sites = siteinds("S=1/2", p.N)
    H = native_projector_system_hamiltonian(p, TNBackend(), sites, collect(1:p.N))
    return CoolingTNS.find_ground_state(H, TNBackend(), sites)[1]
end

"""Per-cycle ensemble mean and standard error over `n_traj` trajectories of the energies returned by `run`."""
function ensemble(run, n_cycles::Int, n_traj::Int)
    total = zeros(n_cycles)
    total_sq = zeros(n_cycles)
    for _ in 1:n_traj
        E = run()
        total .+= E
        total_sq .+= E .^ 2
    end
    means = total ./ n_traj
    variances = max.(total_sq ./ n_traj .- means .^ 2, 0.0)
    return means, sqrt.(variances ./ n_traj)
end

relative_residual(E, E0, E_init) = (E - E0) / (E_init - E0)

function cross_check(N::Int; n_traj::Int=200, seed::Int=101)
    p = NativeGateCircuitParams(N, J_, h_, DELTA_, G_, R_SLICES)
    sites = siteinds("S=1/2", interleaved_total_sites(N))
    H_ed = native_projector_system_hamiltonian(p, EDBackend())
    H_tn = native_projector_system_hamiltonian(p, TNBackend(), sites)
    E0, E_init = ground_energy(p), initial_energy(N)

    rng_ed = MersenneTwister(seed)
    m_ed, s_ed = ensemble(() -> run_native_gate_trajectory(
        p, N_CYCLES, rng_ed; H_S=H_ed, randomized_tau=true, tau_max=TAU_MAX)[1],
        N_CYCLES, n_traj)
    rng_tn = MersenneTwister(seed + 1)
    m_tn, s_tn = ensemble(() -> run_native_gate_trajectory(
        p, N_CYCLES, rng_tn, TNBackend(), sites; H_S=H_tn, maxdim=1 << N,
        randomized_tau=true, tau_max=TAU_MAX)[1], N_CYCLES, n_traj)

    @printf("  N=%2d  E0=%+9.4f  E_init=%+9.4f  (n_traj=%d)\n", N, E0, E_init, n_traj)
    for c in (5, 10, 20, N_CYCLES)
        z = abs(m_ed[c] - m_tn[c]) / sqrt(s_ed[c]^2 + s_tn[c]^2)
        @printf("    q_%-2d   ED %.4f+/-%.4f   TN %.4f+/-%.4f   z=%.2f\n",
                c, relative_residual(m_ed[c], E0, E_init), s_ed[c] / (E_init - E0),
                relative_residual(m_tn[c], E0, E_init), s_tn[c] / (E_init - E0), z)
    end
end

function tn_run(N::Int; maxdim::Int, n_traj::Int, seed::Int)
    p = NativeGateCircuitParams(N, J_, h_, DELTA_, G_, R_SLICES)
    sites = siteinds("S=1/2", interleaved_total_sites(N))
    H_tn = native_projector_system_hamiltonian(p, TNBackend(), sites)
    rng = MersenneTwister(seed)
    diagnostics = NativeGateTrajectoryDiagnostics()

    # One warm-up trajectory so the reported wall clock is steady-state, not compilation.
    run_native_gate_trajectory(p, 1, MersenneTwister(seed), TNBackend(), sites;
                               H_S=H_tn, maxdim=maxdim, randomized_tau=true, tau_max=TAU_MAX)

    means, errs = nothing, nothing
    elapsed = @elapsed begin
        means, errs = ensemble(() -> run_native_gate_trajectory(
            p, N_CYCLES, rng, TNBackend(), sites; H_S=H_tn, maxdim=maxdim,
            randomized_tau=true, tau_max=TAU_MAX, diagnostics=diagnostics)[1],
            N_CYCLES, n_traj)
    end
    return (
        means=means, errs=errs, seconds_per_trajectory=elapsed / n_traj,
        bond_dim=maximum(diagnostics.bond_dims),
        truncation=maximum(diagnostics.truncation_weights),
    )
end

function main()
    println("=" ^ 78)
    println("Native-gate cooling: TN (MPS) backend scaling")
    @printf("J=%.1f h=%.1f Delta=%.1f g=%.1f  r=%d  tau~U[0,%.2f]  cycles=%d\n",
            J_, h_, DELTA_, G_, R_SLICES, TAU_MAX, N_CYCLES)
    println("=" ^ 78)

    println("\n[1] ED-vs-TN cross-check (both backends feasible)")
    for N in (4, 6)
        cross_check(N)
    end

    println("\n[2] TN scaling (ED is infeasible from N ~ 14 upward)")
    println("  N   qubits   q_25              s/traj   max bond   max cum. trunc.")
    largest = 0
    for (N, n_traj) in ((8, 24), (12, 16), (16, 12), (20, 8))
        p = NativeGateCircuitParams(N, J_, h_, DELTA_, G_, R_SLICES)
        E0, E_init = ground_energy(p), initial_energy(N)
        r = tn_run(N; maxdim=64, n_traj=n_traj, seed=200 + N)
        q = relative_residual(r.means[end], E0, E_init)
        dq = r.errs[end] / (E_init - E0)
        @printf("  %2d   %3d      %.4f +/- %.4f   %7.2f   %8d   %.2e\n",
                N, interleaved_total_sites(N), q, dq, r.seconds_per_trajectory,
                r.bond_dim, r.truncation)
        largest = N
    end

    println("\n[3] Bond-dimension convergence at N=$largest")
    p = NativeGateCircuitParams(largest, J_, h_, DELTA_, G_, R_SLICES)
    E0, E_init = ground_energy(p), initial_energy(largest)
    println("  maxdim   q_25              s/traj   max bond   max cum. trunc.")
    for maxdim in (16, 32, 64, 128)
        r = tn_run(largest; maxdim=maxdim, n_traj=8, seed=999)
        @printf("  %6d   %.4f +/- %.4f   %7.2f   %8d   %.2e\n",
                maxdim, relative_residual(r.means[end], E0, E_init),
                r.errs[end] / (E_init - E0), r.seconds_per_trajectory,
                r.bond_dim, r.truncation)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
