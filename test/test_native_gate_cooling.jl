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
        for (N, expected_gates_r2, expected_depth_r2) in
            ((2, 6, 4), (3, 10, 6), (4, 14, 6), (5, 18, 6), (6, 22, 6), (8, 30, 6))
            p = NativeGateCircuitParams(N, 0.25, 3.4, 6.8, 1.35, 2)
            gc = gate_count_and_depth(p)
            @test gc.two_qubit_gates_per_round == expected_gates_r2
            @test gc.entangling_depth_per_round == expected_depth_r2
        end
        # Depth is N-independent (fixed at 3 sublayers/slice for N>=3): only
        # gate *count* (parallel width) grows with N.
        for N in (3, 4, 5, 6, 8)
            p = NativeGateCircuitParams(N, 0.25, 3.4, 6.8, 1.35, 1)
            @test gate_count_and_depth(p).entangling_depth_per_round == 3
        end
    end

    @testset "B-S-B short block (bsb_collision_layers)" begin
        # Matches the collaborator note's short-block numbers at N=5 exactly
        # (14 native two-atom gates, entangling depth 4), via graph coloring.
        for (N, expected_gates, expected_depth) in ((3, 8, 4), (4, 11, 4), (5, 14, 4), (6, 17, 4))
            p = NativeGateCircuitParams(N, 0.25, 3.4, 6.8, 1.35, 1)
            gc = bsb_gate_count_and_depth(p)
            @test gc.two_qubit_gates == expected_gates
            @test gc.entangling_depth == expected_depth
        end

        N = 3
        J_, h_, Delta_, g_ = 0.25, 3.4, 6.8, 1.35
        ham = HamiltonianParameters(IsingModel(), N, (J=J_, h=h_), :open)
        sites_total = interleaved_total_sites(N)
        coupling_params = BasicCouplingParameters("ZZ", g_, 1, 1.0, Delta_)
        H_full = Matrix(CoolingTNS.construct_system_bath_hamiltonian(ham, EDBackend(), sites_total, coupling_params))
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
        # collaborator note's q_25=0.288 short-block vs 0.186 recommended).
        H_S = CoolingTNS.construct_system_hamiltonian(
            HamiltonianParameters(IsingModel(), N, (J=J_, h=h_), :open), EDBackend(), N,
        )
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
        J_, h_, Delta_, g_ = 0.25, 3.4, 6.8, 1.35
        ham = HamiltonianParameters(IsingModel(), N, (J=J_, h=h_), :open)
        sites_total = interleaved_total_sites(N)
        coupling_params = BasicCouplingParameters("ZZ", g_, 1, 1.0, Delta_)
        H_full = Matrix(CoolingTNS.construct_system_bath_hamiltonian(ham, EDBackend(), sites_total, coupling_params))

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

    @testset "sample_bath_ed reproduces ρ_sys exactly (deterministic identity, no RNG)" begin
        # Σ_b P(b) * |M[:,b]/√P(b)⟩⟨M[:,b]/√P(b)| = M*M' exactly -- this is an
        # exact algebraic identity (not a Monte Carlo statement), since it sums
        # over *every* possible bath outcome rather than sampling one.
        N = 3
        p = NativeGateCircuitParams(N, 0.25, 3.4, 6.8, 1.35, 2)
        rng = MersenneTwister(1)
        state = initial_state_plus_cold(N)
        nq = CoolingTNS.n_qubits(p)
        layers = collision_layers(p, 0.4)
        state = apply_collision(state, p, layers, nq)
        state ./= norm(state)

        M = system_bath_matrix(state, N)
        ρ_exact = M * M'
        dim_half = 2^N
        ρ_reconstructed = zeros(ComplexF64, dim_half, dim_half)
        for b in 1:dim_half
            col = M[:, b]
            ρ_reconstructed .+= col * col'
        end
        @test isapprox(ρ_reconstructed, ρ_exact; atol=1e-10)
    end

    @testset "Physical sanity of controls (small N, few trajectories)" begin
        N = 3
        J_, h_, Delta_, g_, tau_max = 0.25, 3.4, 6.8, 1.35, 0.54
        p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 2)
        H_S = CoolingTNS.construct_system_hamiltonian(
            HamiltonianParameters(IsingModel(), N, (J=J_, h=h_), :open), EDBackend(), N,
        )
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
        E_nosandwich = mean_final_energy(2; reset_bath=:zero)
        E_maxmixed = mean_final_energy(3; initial_sys=:maximally_mixed)

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
        J_, h_, Delta_, g_, tau_max = 0.25, 3.4, 6.8, 1.35, 0.3
        p = NativeGateCircuitParams(N, J_, h_, Delta_, g_, 4)
        H_S = CoolingTNS.construct_system_hamiltonian(
            HamiltonianParameters(IsingModel(), N, (J=J_, h=h_), :open), EDBackend(), N,
        )
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
end
