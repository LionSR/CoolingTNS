"""
    test_native_gate_cooling.jl

Correctness tests for src/native_gate_cooling.jl: the hardware-native Rydberg
algorithmic-cooling circuit (ED backend). See notation_translation.md in
ProposalRydbergCooling for the physics/notation background.
"""

using Test
using CoolingTNS
using LinearAlgebra
using Random

@testset "Native-gate cooling circuit" begin

    @testset "ZZ compilation identity" begin
        # exp(-iθZZ) == e^{-iθ} * CP_ij(-4θ) * P_i(2θ) * P_j(2θ), up to the
        # scalar global phase e^{-iθ} (deliberately omitted from the actual
        # gate-application code since global phase never affects observables).
        # This identity targets clean Pauli-ZZ Ising, NOT the projector
        # Hamiltonian this file's circuit actually compiles to (see module
        # docstring / issue #675) -- kept as a documented, independently
        # verified, but deliberately unused-by-default alternate compilation.
        nq = 2
        Z = ComplexF64[1 0; 0 -1]
        ZZ_diag = real.(diag(kron(Z, Z)))
        for θ in (0.13, -0.4, 1.1, 2.7, 0.0)
            compiled = native_zz_evolution_diag(θ, 1, 2, nq) .* cis(-θ)
            exact = exp.(-im .* θ .* ZZ_diag)
            @test isapprox(compiled, exact; atol=1e-10)
        end
    end

    @testset "Gate count / entangling depth (graph-colored, not hardcoded)" begin
        # Matches the collaborator note's headline numbers at N=5, r=2 exactly
        # (18 native two-atom gates, entangling depth 6) via real edge-coloring.
        # Gate count/depth is purely combinatorial (depends only on N and
        # connectivity), so it is independent of J/h/Delta/g's numerical
        # values -- these are the recommended-point couplings (Eq.
        # \ref{eq:recommended}: J=K=1.0, h=3.4, Delta=6.8, g=L=5.4) for
        # documentation, not because this test needs them.
        for (N, expected_gates_r2, expected_depth_r2) in
            ((2, 6, 4), (3, 10, 6), (4, 14, 6), (5, 18, 6), (6, 22, 6), (8, 30, 6))
            p = NativeGateCircuitParams(N, 1.0, 3.4, 6.8, 5.4, 2)
            gc = gate_count_and_depth(p)
            @test gc.two_qubit_gates_per_round == expected_gates_r2
            @test gc.entangling_depth_per_round == expected_depth_r2
        end
        # Depth is N-independent (fixed at 3 sublayers/slice for N>=3): only
        # gate *count* (parallel width) grows with N.
        for N in (3, 4, 5, 6, 8)
            p = NativeGateCircuitParams(N, 1.0, 3.4, 6.8, 5.4, 1)
            @test gate_count_and_depth(p).entangling_depth_per_round == 3
        end
    end

    @testset "B-S-B short block (bsb_collision_layers)" begin
        # Matches the collaborator note's short-block numbers at N=5 exactly
        # (14 native two-atom gates, entangling depth 4), via graph coloring.
        for (N, expected_gates, expected_depth) in ((3, 8, 4), (4, 11, 4), (5, 14, 4), (6, 17, 4))
            p = NativeGateCircuitParams(N, 1.0, 3.4, 6.8, 5.4, 1)
            gc = bsb_gate_count_and_depth(p)
            @test gc.two_qubit_gates == expected_gates
            @test gc.entangling_depth == expected_depth
        end

        N = 3
        J_, h_, Delta_, g_ = 1.0, 3.4, 6.8, 5.4
        sites_total = interleaved_total_sites(N)
        rng = MersenneTwister(0)
        ψ0 = randn(rng, ComplexF64, 2^sites_total)
        ψ0 ./= norm(ψ0)

        # bsb_collision_layers(p, τ) should exactly implement
        # exp(-i(τ/2)V) exp(-iτ H_S^chain) exp(-iτ H_X) exp(-i(τ/2)V) -- NOT
        # exp(-iτ H_full) (it deliberately omits the bath's own field H_bath,
        # matching the collaborator's B-S-B construction), so check it against
        # its own defining decomposition rather than the full propagator.
        p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 1)
        τ = 0.2
        layers = bsb_collision_layers(p, τ)
        @test length(layers) == 5
        state = apply_collision(copy(ψ0), p, layers, sites_total)
        @test isapprox(norm(state), 1.0; atol=1e-10)

        # Physical sanity at small N, few trajectories: B-S-B should cool
        # distinctly less than the recommended two-slice block (matching the
        # collaborator note's q_25=0.288 short-block vs 0.186 recommended,
        # from Benjamin's own separately-retuned B-S-B parameters -- this
        # test only checks the qualitative ordering at the shared recommended
        # point, not those exact retuned values).
        H_S = native_projector_system_hamiltonian(N, J_, h_)
        p_r2 = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2)
        n_cycles, n_traj = 15, 150
        function mean_final_energy(rng_seed; kwargs...)
            rng = MersenneTwister(rng_seed)
            total = 0.0
            for _ in 1:n_traj
                E, _ = run_native_gate_trajectory(p_r2, n_cycles, rng; H_S=H_S, randomized_tau=true, tau_max=0.54, kwargs...)
                total += E[end]
            end
            return total / n_traj
        end
        E_recommended = mean_final_energy(20)
        E_bsb = mean_final_energy(21; layers_fn=bsb_collision_layers)
        @test E_bsb > E_recommended + 1.0
    end

    @testset "Trotter circuit converges to the exact continuum propagator" begin
        N = 3
        J_, h_, Delta_, g_ = 1.0, 3.4, 6.8, 5.4
        sites_total = interleaved_total_sites(N)
        # native_projector_total_hamiltonian: the collaborator note's own
        # projector Hamiltonian H_S+H_A+V (Eq. \ref{eq:model}), NOT
        # `construct_system_bath_hamiltonian` for `IsingModel`+`--coupling
        # ZZ` -- that is a different (clean-ZZ) target that `collision_layers`
        # no longer compiles to (see module docstring / issue #675).
        H_full = Matrix(native_projector_total_hamiltonian(N, J_, h_, Delta_, g_))

        rng = MersenneTwister(0)
        ψ0 = randn(rng, ComplexF64, 2^sites_total)
        ψ0 ./= norm(ψ0)

        τ = 0.3
        exact = exp(-im * τ * H_full) * ψ0
        infidelities = Float64[]
        for r in (1, 2, 4, 8)
            p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, r)
            layers = collision_layers(p, τ)
            state = apply_collision(copy(ψ0), p, layers, sites_total)
            push!(infidelities, 1 - abs(dot(exact, state)))
        end
        # Strictly decreasing with r, and a fine split should be essentially exact.
        @test issorted(infidelities; rev=true)
        @test infidelities[end] < 1e-5
    end

    @testset "Measurement-based reset (process_bath_ed_monte_carlo) reproduces ρ_sys" begin
        N = 3
        p = NativeGateCircuitParams(N, 1.0, 3.4, 6.8, 5.4, 2)
        state = initial_state_plus_cold(N)
        nq = CoolingTNS.n_qubits(p)
        layers = collision_layers(p, 0.4)
        state = apply_collision(state, p, layers, nq)
        state ./= norm(state)

        M = system_bath_matrix(state, N)
        ρ_exact = M * M'
        dim_half = 2^N

        # (a) Exact algebraic identity, no RNG: Σ_b M[:,b]*M[:,b]' = M*M' --
        # summing every bath outcome's unnormalized collapse reproduces ρ_sys,
        # the algebraic reason the measurement-based reset is exact rather
        # than approximate.
        ρ_reconstructed = zeros(ComplexF64, dim_half, dim_half)
        for b in 1:dim_half
            col = M[:, b]
            ρ_reconstructed .+= col * col'
        end
        @test isapprox(ρ_reconstructed, ρ_exact; atol=1e-10)

        # (b)-(d) The production reset path itself (`_measure_and_reset` ->
        # process_bath_ed_monte_carlo -> measure_ed!), not just the identity
        # above: (b) every collapse must return exactly the *normalized* M
        # column of its sampled bath outcome, with no relative-phase
        # corruption (`EDStateVector` normalizes on construction), pinning the
        # interleaved bit-ordering conventions of measure_ed! against
        # system_bath_matrix -- the [s1,b1]-vs-[s1,s2] partial-trace bug
        # class; (c) sampled outcome frequencies must follow the Born weights
        # ‖M[:,b]‖²; (d) the ensemble average of the collapses must reproduce
        # ρ_sys within Monte Carlo error (seeded, so deterministic in CI).
        n_samples = 4000
        rng = MersenneTwister(2)
        counts = zeros(Int, dim_half)
        ρ_emp = zeros(ComplexF64, dim_half, dim_half)
        worst_column_mismatch = 0.0
        for _ in 1:n_samples
            ψ_sys, outcomes = CoolingTNS.process_bath_ed_monte_carlo(
                CoolingTNS.EDStateVector(copy(state), 2N), N, rng)
            b = sum(outcomes[i] << (i - 1) for i in 1:N)  # bath spin i at bit i-1
            counts[b + 1] += 1
            col = M[:, b + 1]
            worst_column_mismatch =
                max(worst_column_mismatch, norm(ψ_sys.data .- col ./ norm(col)))
            v = ψ_sys.data
            ρ_emp .+= (v * v') ./ n_samples
        end
        @test worst_column_mismatch < 1e-10
        born_weights = [sum(abs2, M[:, b]) for b in 1:dim_half]
        @test isapprox(counts ./ n_samples, born_weights; atol=0.05)
        @test opnorm(ρ_emp - ρ_exact) < 0.05
    end

    @testset "Noise clock matches the reported entangling depth (noise_passes/noise_sites)" begin
        # The depolarizing-noise applications must track the same graph-colored
        # sublayer schedule that gate_count_and_depth/bsb_gate_count_and_depth
        # report entangling depth in -- one all-qubit pass per entangling
        # sublayer -- rather than one pass per bookkeeping layer object.
        for N in (2, 3, 5), r in (1, 2)
            p = NativeGateCircuitParams(N, 1.0, 3.4, 6.8, 5.4, r)
            diag_passes = sum(noise_passes(l) for l in collision_layers(p, 0.3) if l isa DiagonalLayer)
            @test diag_passes == gate_count_and_depth(p).entangling_depth_per_round
            bsb_diag_passes = sum(noise_passes(l) for l in bsb_collision_layers(p, 0.3) if l isa DiagonalLayer)
            @test bsb_diag_passes == bsb_gate_count_and_depth(p).entangling_depth
        end
        # Global rotation pulses noise only their own register (system and
        # bath rotations act on disjoint atoms); entangling sublayers expose
        # every atom to the global Rydberg illumination.
        N = 3
        registers = (sys=interleaved_system_sites(N), bath=interleaved_bath_sites(N),
                     all=collect(1:interleaved_total_sites(N)))
        @test noise_sites(SystemRotationLayer(0.1), registers) == registers.sys
        @test noise_sites(BathRotationLayer(0.1), registers) == registers.bath
        @test noise_sites(DiagonalLayer(ones(ComplexF64, 1 << 2N), 3), registers) == registers.all
        @test noise_passes(SystemRotationLayer(0.1)) == 1
        @test noise_passes(BathRotationLayer(0.1)) == 1
        @test noise_passes(DiagonalLayer(ones(ComplexF64, 1 << 2N), 3)) == 3
    end

    @testset "Physical sanity of controls (small N, few trajectories)" begin
        N = 3
        J_, h_, Delta_, g_, tau_max = 1.0, 3.4, 6.8, 5.4, 0.54
        p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2)
        H_S = native_projector_system_hamiltonian(N, J_, h_)
        E0, ψ0, _ = CoolingTNS.find_ground_state(H_S, EDBackend())

        n_cycles, n_traj = 15, 150
        function mean_final_energy(rng_seed; kwargs...)
            rng = MersenneTwister(rng_seed)
            total = 0.0
            for _ in 1:n_traj
                E, _ = run_native_gate_trajectory(p, n_cycles, rng; H_S=H_S, randomized_tau=true, tau_max=tau_max, kwargs...)
                total += E[end]
            end
            return total / n_traj
        end

        E_default = mean_final_energy(1)
        E_nosandwich = mean_final_energy(2; reset_bath=ZeroReset())
        E_maxmixed = mean_final_energy(3; initial_sys=MaximallyMixedState())

        # No-sandwich (reset straight to |0>, not cold |X-> for the X-field
        # bath) must cool distinctly worse than the reset sandwich.
        @test E_nosandwich > E_default + 1.0

        # Maximally-mixed input should converge close to the same cooled
        # attractor as the hot |+>^N input -- the cooling mechanism is a real
        # attractor, not an artifact of the specific initial state.
        @test isapprox(E_maxmixed, E_default; atol=3.0)

        # Both defaults must cool below the system's own ground energy's
        # trivial upper bound (E_init), i.e. actually cool at all.
        plus = ComplexF64[1, 1] / sqrt(2)
        sys_plus = reduce(kron, fill(plus, N))
        E_init = real(dot(sys_plus, H_S * sys_plus))
        @test E_default < E_init
        @test E0 <= E_default
    end

    @testset "Exact-continuous reference is consistent with the Trotterized circuit" begin
        N = 3
        J_, h_, Delta_, g_, tau_max = 1.0, 3.4, 6.8, 5.4, 0.3
        p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 4)
        H_S = native_projector_system_hamiltonian(N, J_, h_)
        evals, evecs = exact_collision_operator(p)

        n_cycles, n_traj = 10, 100
        function mean_final_energy(rng, runner)
            total = 0.0
            for _ in 1:n_traj
                E, _ = runner(rng)
                total += E[end]
            end
            return total / n_traj
        end
        E_trotter = mean_final_energy(MersenneTwister(5),
            rng -> run_native_gate_trajectory(p, n_cycles, rng; H_S=H_S, randomized_tau=true, tau_max=tau_max))
        E_exact = mean_final_energy(MersenneTwister(6),
            rng -> run_exact_continuous_trajectory(p, n_cycles, rng; H_S=H_S, randomized_tau=true, tau_max=tau_max, evals=evals, evecs=evecs))

        # At r=4 the Trotter error should already be small relative to the
        # overall cooling scale.
        @test isapprox(E_trotter, E_exact; atol=2.0)
    end

    @testset "Noise path (apply_depolarizing_ed via apply_collision)" begin
        N = 3
        J_, h_, Delta_, g_, tau_max = 1.0, 3.4, 6.8, 5.4, 0.54
        p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2)
        H_S = native_projector_system_hamiltonian(N, J_, h_)

        # Reproducibility: apply_depolarizing_ed is now RNG-threaded (was the
        # global RNG before), so two runs seeded identically must match exactly.
        rng_a = MersenneTwister(7)
        Ea, _ = run_native_gate_trajectory(p, 10, rng_a; H_S=H_S, randomized_tau=true, tau_max=tau_max, noise_p=0.05)
        rng_b = MersenneTwister(7)
        Eb, _ = run_native_gate_trajectory(p, 10, rng_b; H_S=H_S, randomized_tau=true, tau_max=tau_max, noise_p=0.05)
        @test Ea == Eb

        # Physical sanity: noise should degrade cooling relative to the
        # noiseless case, on ensemble average.
        n_cycles, n_traj = 15, 150
        function mean_final_energy(seed; kwargs...)
            rng = MersenneTwister(seed)
            total = 0.0
            for _ in 1:n_traj
                E, _ = run_native_gate_trajectory(p, n_cycles, rng; H_S=H_S, randomized_tau=true, tau_max=tau_max, kwargs...)
                total += E[end]
            end
            return total / n_traj
        end
        E_noiseless = mean_final_energy(30)
        E_noisy = mean_final_energy(31; noise_p=0.05)
        @test E_noisy > E_noiseless
    end

    @testset "Exact-continuous fidelity tracking" begin
        N = 3
        J_, h_, Delta_, g_, tau_max = 1.0, 3.4, 6.8, 5.4, 0.3
        p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 1)
        H_S = native_projector_system_hamiltonian(N, J_, h_)
        E0, ψ0, _ = CoolingTNS.find_ground_state(H_S, EDBackend())
        gs = ComplexF64.(ψ0.data)
        rng = MersenneTwister(8)
        E, F = run_exact_continuous_trajectory(p, 10, rng; H_S=H_S, randomized_tau=true, tau_max=tau_max, ground_state=gs)
        @test F !== nothing
        @test all(0.0 .<= F .<= 1.0 + 1e-9)
        # Fidelity should trend upward as the system cools toward the ground state.
        @test F[end] > F[1]
    end

    @testset "Residual local-phase (residual_alpha) sensitivity" begin
        # residual_alpha mirrors the collaborator note's residual_alpha_per_pulse
        # in reproduce_native_note.py's native_diagonal_phase: an uncompensated
        # single-atom phase left over per real Rydberg pulse after imperfect
        # calibration/tracking of the global P(alpha) phase.
        N = 3
        J_, h_, Delta_, g_, tau_max = 1.0, 3.4, 6.8, 5.4, 0.54
        H_S = native_projector_system_hamiltonian(N, J_, h_)

        # Regression: the 6-argument constructor (no residual_alpha) must be
        # indistinguishable from the 7-argument form with residual_alpha=0.0 --
        # same parameters, same compiled circuit layers, same trajectory.
        p_default = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2)
        p_explicit_zero = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2, 0.0)
        @test p_default == p_explicit_zero

        τ = 0.3
        layers_default = collision_layers(p_default, τ)
        layers_explicit_zero = collision_layers(p_explicit_zero, τ)
        @test length(layers_default) == length(layers_explicit_zero)
        for (a, b) in zip(layers_default, layers_explicit_zero)
            @test typeof(a) == typeof(b)
            if a isa DiagonalLayer
                @test a.phase == b.phase
            end
        end

        rng_a = MersenneTwister(40)
        Ea, _ = run_native_gate_trajectory(p_default, 10, rng_a; H_S=H_S, randomized_tau=true, tau_max=tau_max)
        rng_b = MersenneTwister(40)
        Eb, _ = run_native_gate_trajectory(p_explicit_zero, 10, rng_b; H_S=H_S, randomized_tau=true, tau_max=tau_max)
        @test Ea == Eb

        # Physical sanity: cooling quality should measurably degrade (higher
        # ensemble-averaged final energy) once |residual_alpha| grows large
        # enough, analogous to the collaborator note's q_25(alpha)/q_20(alpha)
        # sensitivity curve (scripts/native_phase_sensitivity_scan.jl).
        n_cycles, n_traj = 15, 150
        function mean_final_energy(p, seed)
            rng = MersenneTwister(seed)
            total = 0.0
            for _ in 1:n_traj
                E, _ = run_native_gate_trajectory(p, n_cycles, rng; H_S=H_S, randomized_tau=true, tau_max=tau_max)
                total += E[end]
            end
            return total / n_traj
        end

        p_zero = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2, 0.0)
        p_small = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2, 0.05)
        p_large = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2, 0.2)
        p_large_neg = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2, -0.2)

        E_zero = mean_final_energy(p_zero, 50)
        E_small = mean_final_energy(p_small, 51)
        E_large = mean_final_energy(p_large, 52)
        E_large_neg = mean_final_energy(p_large_neg, 53)

        @test E_large > E_small
        # No monotonic "small residual_alpha strictly hurts" assertion here:
        # at this file's corrected (issue #675) recommended-point coupling
        # (J=K=1.0, g=L=5.4 -- 4x stronger than the pre-fix J=K/4, g=L/4 this
        # test was originally tuned against), a *small* residual phase
        # (0.05 rad) measurably HELPS cooling on ensemble average rather than
        # hurting it -- verified at high statistics (n_traj=3000):
        # E_small=-5.23+/-0.05 vs E_zero=-4.67+/-0.05, opposite of the naive
        # "any miscalibration hurts" expectation. Only a large enough residual
        # phase (0.2 rad here) reliably degrades cooling, in either sign
        # direction (the phase enters through
        # cis(n_pulses*residual_alpha*n_total), not an odd function of alpha
        # alone once combined with the rest of the circuit, so exact +/-
        # symmetry isn't asserted -- only that both directions hurt cooling
        # relative to the perfectly-calibrated circuit once alpha is large).
        @test E_large_neg > E_zero + 1.0
        @test E_large > E_zero + 1.0
    end

    @testset "Purity diagnostic (compute_purity, opt-in)" begin
        N = 3
        J_, h_, Delta_, g_, tau_max = 1.0, 3.4, 6.8, 5.4, 0.54
        p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2)
        H_S = native_projector_system_hamiltonian(N, J_, h_)

        @testset "Unentangled product state has purity 1 (exact/deterministic)" begin
            # No collision has been applied yet: system and bath are still an
            # exact product state, so ρ_sys is pure.
            state = initial_state_plus_cold(N)
            M = system_bath_matrix(state, N)
            @test isapprox(purity_from_matrix(M), 1.0; atol=1e-10)

            # ‖M*M'‖_F² and ‖M'*M‖_F² must agree exactly (M square ⟹ M*M'
            # and M'*M share the same eigenvalues), confirming
            # purity_from_matrix's M'*M shortcut is equivalent to Tr(ρ_sys²).
            @test isapprox(real(sum(abs2, M * M')), real(sum(abs2, M' * M)); atol=1e-10)
        end

        @testset "A real collision creates genuine entanglement (purity < 1)" begin
            rng = MersenneTwister(42)
            state = initial_state_plus_cold(N)
            nq = CoolingTNS.n_qubits(p)
            layers = collision_layers(p, tau_max)
            state = apply_collision(state, p, layers, nq)
            state ./= norm(state)
            M = system_bath_matrix(state, N)
            pur = purity_from_matrix(M)
            @test -1e-10 <= pur <= 1.0 + 1e-10
            @test pur < 1.0 - 1e-6
        end

        @testset "compute_purity=false (default) returns nothing" begin
            rng = MersenneTwister(9)
            E, F, Pur = run_native_gate_trajectory(p, 5, rng; H_S=H_S, randomized_tau=true, tau_max=tau_max)
            @test Pur === nothing
            E2, F2, Pur2 = run_exact_continuous_trajectory(p, 5, rng; H_S=H_S, randomized_tau=true, tau_max=tau_max)
            @test Pur2 === nothing
        end

        @testset "compute_purity=true: bounds and cooling-cycle trend (ensemble)" begin
            n_cycles, n_traj = 15, 150
            purity_sums = zeros(n_cycles)
            rng = MersenneTwister(11)
            for _ in 1:n_traj
                E, F, Pur = run_native_gate_trajectory(
                    p, n_cycles, rng; H_S=H_S, randomized_tau=true, tau_max=tau_max, compute_purity=true,
                )
                @test Pur !== nothing
                @test length(Pur) == n_cycles
                @test all(-1e-8 .<= Pur .<= 1.0 + 1e-8)
                purity_sums .+= Pur
            end
            purity_mean = purity_sums ./ n_traj
            println("\n  Native-gate purity evolution (ensemble mean, n_traj=$n_traj): $(round.(purity_mean, digits=6))")

            # Trend, verified empirically (not assumed) across several RNG
            # seeds at these parameters: purity dips sharply on the very
            # first collision (the hot, maximally-symmetric |+⟩^N system gets
            # strongly entangled with the bath -- confirms the channel is
            # genuinely non-unital/mixing, the collaborator note's point in
            # introducing this diagnostic), then *recovers upward* on
            # ensemble average across subsequent cycles as the system is
            # driven toward its pure ground-state attractor. This is the
            # same direction as the "Exact-continuous fidelity tracking"
            # testset above (ground-state fidelity F trends upward,
            # `F[end] > F[1]`, as cooling proceeds): since the cooling target
            # is a pure state, fidelity -> 1 necessarily drags purity -> 1
            # alongside it, so a monotonic *decrease* in purity across full
            # cooling cycles would be inconsistent with successful cooling,
            # not a sign of it -- entropy being "extracted" from the system
            # means the system purity goes up, not down. See CLAUDE.md's
            # "Debugging Best Practices": this direction was confirmed
            # numerically (see purity_trend_check2.jl-style scan across
            # seeds 1/11/99/12345, all showing the same ~0.14-0.15 recovery
            # gap) rather than assumed from the issue text's paraphrase.
            @test purity_mean[1] < 0.9
            half = n_cycles ÷ 2
            early_mean = sum(purity_mean[1:half]) / half
            late_mean = sum(purity_mean[(end - half + 1):end]) / half
            @test late_mean > early_mean + 0.05
        end
    end

    @testset "N=5 recommended point matches the collaborator's headline q_20/q_25 (issue #675)" begin
        # Regression test for issue #675: reproduces AlgoCool2026.tex Table 1's
        # noiseless benchmark at the recommended operating point (Eq.
        # \ref{eq:recommended}: N=5, J=K=1.0, h=3.4, Delta=6.8, g=L=5.4,
        # tau_max=0.54, r=2) -- E0/K=-16.13167, E_init/K=18, digital
        # q_20=0.207/q_25=0.186, exact-continuous q_20=0.194/q_25=0.164.
        #
        # Root cause of the originally-reported ~1.5-2x mismatch (see
        # notation_translation.md §6 and the module docstring of
        # src/native_gate_cooling.jl): this file used to compile/measure
        # against the *clean Pauli-ZZ* Ising Hamiltonian instead of the
        # collaborator's own *projector* Ising Hamiltonian (`H_S=J·Σn_in_{i+1}
        # +h·ΣX_i`) -- fixed by `native_projector_system_hamiltonian`/
        # `native_projector_total_hamiltonian` and the uncompensated
        # `CP_ij(-Jt)` compilation in `native_chain_diagonal`/`native_pair_diagonal`.
        #
        # This check uses the EXACT Gauss-Legendre-quadrature-averaged channel
        # over the randomized collision time (matching the collaborator's own
        # `reproduce_native_note.py` methodology: `leggauss`/`averaged_kraus`/
        # `trajectory`), built from this file's own production primitives
        # (`collision_layers`, `apply_collision`, `exact_collision_operator`),
        # rather than finite-trajectory MCWF sampling -- deterministic, so it
        # can be checked to a much tighter tolerance than a Monte Carlo average.
        N, J_, h_, Delta_, g_, tau_max, r = 5, 1.0, 3.4, 6.8, 5.4, 0.54, 2
        p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, r)
        nq = CoolingTNS.n_qubits(p)
        dim_half = 1 << N

        H_S = native_projector_system_hamiltonian(p)
        E0, ψ0_gs, _ = CoolingTNS.find_ground_state(H_S, EDBackend())
        plus = ComplexF64[1, 1] / sqrt(2)
        sys_plus = reduce(kron, fill(plus, N))
        E_init = real(dot(sys_plus, H_S * sys_plus))
        @test isapprox(E0, -16.13167; atol=1e-3)
        @test isapprox(E_init, 18.0; atol=1e-9)

        # Gauss-Legendre quadrature on [0, tau_max] via the Golub-Welsch
        # eigendecomposition of the Legendre Jacobi matrix (avoids adding a
        # quadrature-package dependency just for this test).
        nquad = 10
        β = [k / sqrt(4k^2 - 1) for k in 1:(nquad - 1)]
        legendre_vals, legendre_vecs = eigen(SymTridiagonal(zeros(nquad), β))
        legendre_weights = 2 .* (legendre_vecs[1, :] .^ 2)
        perm = sortperm(legendre_vals)
        taus = tau_max .* (legendre_vals[perm] .+ 1) ./ 2
        weights = legendre_weights[perm] ./ 2

        bath0 = bath_ground_state_product(N)

        # Kraus operators of one collision: propagate every system
        # computational basis state (with a fresh cold bath attached) through
        # `propagate`, then read off the reduced system-bath amplitude
        # matrix -- the Stinespring-dilation construction of the bath-traced
        # channel (`system_bath_matrix`'s columns *are* the Kraus operators,
        # indexed by final bath outcome).
        function collision_kraus_ops(propagate::Function, τ::Float64)
            Ks = [zeros(ComplexF64, dim_half, dim_half) for _ in 1:dim_half]
            for s in 0:(dim_half - 1)
                sys_amp = zeros(ComplexF64, dim_half)
                sys_amp[s + 1] = 1
                state = propagate(build_interleaved_state(sys_amp, bath0, N), τ)
                M = system_bath_matrix(state, N)
                for b in 1:dim_half
                    Ks[b][:, s + 1] = M[:, b]
                end
            end
            return Ks
        end

        function quadrature_channel(propagate::Function)
            Ks = Matrix{ComplexF64}[]
            for (τ, ω) in zip(taus, weights), K in collision_kraus_ops(propagate, τ)
                push!(Ks, sqrt(ω) .* K)
            end
            return Ks
        end

        function apply_channel(Ks::Vector{Matrix{ComplexF64}}, rho::Matrix{ComplexF64})
            out = zeros(ComplexF64, dim_half, dim_half)
            for K in Ks
                out .+= K * rho * K'
            end
            return (out .+ out') ./ 2
        end

        function q_trajectory(Ks::Vector{Matrix{ComplexF64}}; n_cycles::Int=25)
            rho = sys_plus * sys_plus'
            q = zeros(n_cycles)
            for m in 1:n_cycles
                rho = apply_channel(Ks, rho)
                E = real(tr(rho * H_S))
                q[m] = (E - E0) / (E_init - E0)
            end
            return q
        end

        digital_propagate(state, τ) = apply_collision(state, p, collision_layers(p, τ), nq)
        q_digital = q_trajectory(quadrature_channel(digital_propagate))
        @test isapprox(q_digital[20], 0.207; rtol=0.05)
        @test isapprox(q_digital[25], 0.186; rtol=0.05)

        evals, evecs = exact_collision_operator(p)
        exact_propagate(state, τ) = CoolingTNS._exact_collision_propagate(state, τ, evals, evecs)
        q_exact = q_trajectory(quadrature_channel(exact_propagate))
        @test isapprox(q_exact[20], 0.194; rtol=0.05)
        @test isapprox(q_exact[25], 0.164; rtol=0.05)
    end
end
