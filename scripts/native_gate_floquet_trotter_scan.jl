"""
    native_gate_floquet_trotter_scan.jl

Floquet-vs-Trotter scan for the hardware-native Rydberg algorithmic-cooling
circuit (see src/native_gate_cooling.jl), at N_sys=N_bath=4 and 6 (8 and 12
total qubits), using MCWF trajectories on the ED backend.

House-notation parameters are the direct translation of the collaborator note's
recommended operating point (h/K=g_theirs/K=3.4, L/K=5.4, K*tau_m~Uniform[0,0.54]):
J=K=1.0, h=3.4, Delta=2*g_theirs=6.8, g=L=5.4, tau_max=0.54 (K=1 unit). `J`/`g`
are the number-operator (projector) couplings of Eq. \ref{eq:model} directly,
*not* Pauli-ZZ coefficients -- see src/native_gate_cooling.jl module docstring
and ProposalRydbergCooling/notation_translation.md (issue #675).

For each r (number of Trotter slices per collision -- r=1 is the coarsest,
most Floquet-kick-like circuit; larger r approaches the continuum limit), reports
the real (graph-colored) gate count / entangling depth, the noiseless cooling
trajectory, and the trajectory under a per-qubit-per-hardware-sublayer
depolarizing channel (one all-qubit pass per graph-colored entangling sublayer
-- the same sublayer schedule the printed depth counts -- plus one
own-register pass per global rotation pulse; see `noise_passes`/`noise_sites`
in src/native_gate_cooling.jl) with the MPQ Sr-88 Rydberg-CZ infidelity
(arXiv:2506.10714, fidelity 0.9945) as the reference error rate -- the
noise-modeling methodology follows
arXiv:2303.08461 (Yang, Christianen, ..., Cirac), which studies exactly this
question (how coarse can a Trotter/Floquet step be before noise, not
discretization error, is the limiting factor).
"""

using CoolingTNS
using LinearAlgebra
using Random
using Printf

const J_, h_, DELTA_, G_, TAU_MAX = 1.0, 3.4, 6.8, 5.4, 0.54
const P_TWO_QUBIT = 1 - 0.9945  # MPQ Rydberg CZ infidelity, arXiv:2506.10714

function relative_residual(E_traj, E0, E_init)
    return (E_traj .- E0) ./ (E_init - E0)
end

function system_energetics(N::Int)
    H_S = native_projector_system_hamiltonian(N, J_, h_)  # keep sparse
    E0, _, _ = CoolingTNS.find_ground_state(H_S, EDBackend())
    plus = ComplexF64[1, 1] / sqrt(2)
    sys_plus = reduce(kron, fill(plus, N))
    E_init = real(dot(sys_plus, H_S * sys_plus))
    return H_S, E0, E_init
end

function run_batch(p::NativeGateCircuitParams, H_S, n_cycles::Int, n_traj::Int, seed::Int; noise_p::Float64=0.0)
    rng = MersenneTwister(seed)
    all_E = zeros(n_traj, n_cycles)
    for t in 1:n_traj
        energies, _ = run_native_gate_trajectory(
            p, n_cycles, rng; H_S=H_S, noise_p=noise_p, randomized_tau=true, tau_max=TAU_MAX,
        )
        all_E[t, :] = energies
    end
    return vec(sum(all_E, dims=1)) ./ n_traj
end

function gate_count_table()
    println("="^70)
    println("Real (computed) gate count / entangling depth vs N, r=2")
    println("="^70)
    for N in (2, 3, 4, 5, 6, 8)
        p = NativeGateCircuitParams(N, J_, h_, DELTA_, G_, 2)
        gc = gate_count_and_depth(p)
        @printf("N=%2d: %3d native 2-qubit gates/round, entangling depth %2d/round (%d sublayers x r=2)\n",
                N, gc.two_qubit_gates_per_round, gc.entangling_depth_per_round, gc.sublayers_per_diag_step)
    end
    println()
end

function floquet_trotter_scan(N::Int; n_cycles::Int=25, n_traj::Int=200, r_values=(1, 2, 3, 4, 8))
    println("="^70)
    println("Floquet-vs-Trotter scan at N=$N ($(2N) total qubits)")
    println("="^70)
    H_S, E0, E_init = system_energetics(N)
    for r in r_values
        p = NativeGateCircuitParams(N, J_, h_, DELTA_, G_, r)
        gc = gate_count_and_depth(p)
        t0 = time()
        E_noiseless = run_batch(p, H_S, n_cycles, n_traj, 100 + r)
        q_noiseless = relative_residual(E_noiseless, E0, E_init)
        E_noisy = run_batch(p, H_S, n_cycles, n_traj, 200 + r; noise_p=P_TWO_QUBIT)
        q_noisy = relative_residual(E_noisy, E0, E_init)
        @printf("r=%2d (gates/round=%3d, depth/round=%2d): noiseless q_final=%.4f  |  with MPQ-CZ-fidelity noise (p=%.4f/qubit/sublayer) q_final=%.4f   [%.1fs]\n",
                r, gc.two_qubit_gates_per_round, gc.entangling_depth_per_round,
                q_noiseless[end], P_TWO_QUBIT, q_noisy[end], time() - t0)
    end
    println()
end

gate_count_table()
floquet_trotter_scan(4; n_cycles=25, n_traj=300, r_values=(1, 2, 3, 4, 8))
floquet_trotter_scan(6; n_cycles=25, n_traj=150, r_values=(1, 2, 3, 4, 8))
