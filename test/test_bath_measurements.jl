using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra

@isdefined(test_mpo_to_matrix) || include("test_helpers.jl")

@testset "Bath Magnetization Convention" begin
    tn_mc_state = CoolingTNS.QuantumState(
        CoolingTNS.TNBackend(),
        CoolingTNS.MonteCarloWavefunction(),
        CoolingTNS.TrotterEvolution(),
        nothing,
    )
    ed_mc_state = CoolingTNS.QuantumState(
        CoolingTNS.EDBackend(),
        CoolingTNS.MonteCarloWavefunction(),
        CoolingTNS.ContinuousEvolution(),
        nothing,
    )
    ed_dm_state = CoolingTNS.QuantumState(
        CoolingTNS.EDBackend(),
        CoolingTNS.DensityMatrix(),
        CoolingTNS.ContinuousEvolution(),
        nothing,
    )
    tn_dm_state = CoolingTNS.QuantumState(
        CoolingTNS.TNBackend(),
        CoolingTNS.DensityMatrix(),
        CoolingTNS.TrotterEvolution(),
        nothing,
    )

    @testset "Monte Carlo sample maps" begin
        # TN samples are ITensor site indices: 1 = Up, 2 = Dn.
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_mc_state, [1, 1], 2
        ) ≈ 1.0
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_mc_state, [2, 2], 2
        ) ≈ -1.0
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_mc_state, [1, 2], 2
        ) ≈ 0.0
        @test_throws ArgumentError CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_mc_state, [0], 1
        )
        @test_throws ArgumentError CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_mc_state, [1], 2
        )
        @test_throws ArgumentError CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_mc_state, [1, 2], 0
        )

        # ED samples are computational bits: 0 = Up, 1 = Dn.
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.EDBackend(), ed_mc_state, [0, 0], 2
        ) ≈ 1.0
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.EDBackend(), ed_mc_state, [1, 1], 2
        ) ≈ -1.0
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.EDBackend(), ed_mc_state, [0, 1], 2
        ) ≈ 0.0
        @test_throws ArgumentError CoolingTNS.compute_bath_magnetization(
            CoolingTNS.EDBackend(), ed_mc_state, [2], 1
        )
        @test_throws ArgumentError CoolingTNS.compute_bath_magnetization(
            CoolingTNS.EDBackend(), ed_mc_state, [0], 2
        )
        @test_throws ArgumentError CoolingTNS.compute_bath_magnetization(
            CoolingTNS.EDBackend(), ed_mc_state, [0, 1], 0
        )
    end

    @testset "Density matrix maps" begin
        ρ_up = ComplexF64[1 0; 0 0]
        ρ_dn = ComplexF64[0 0; 0 1]
        ρ_mix = ComplexF64[0.5 0; 0 0.5]

        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.EDBackend(), ed_dm_state, ρ_up, 1
        ) ≈ 1.0
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.EDBackend(), ed_dm_state, ρ_dn, 1
        ) ≈ -1.0
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.EDBackend(), ed_dm_state, ρ_mix, 1
        ) ≈ 0.0

        sites_bath = siteinds("S=1/2", 2)
        ψ_up_dn = MPS(sites_bath, ["Up", "Dn"])
        ψ_dn_dn = MPS(sites_bath, ["Dn", "Dn"])
        ρ_up_dn = outer(ψ_up_dn', ψ_up_dn)
        ρ_dn_dn = outer(ψ_dn_dn', ψ_dn_dn)

        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_dm_state, ρ_up_dn, sites_bath
        ) ≈ 0.0
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_dm_state, ρ_dn_dn, sites_bath
        ) ≈ -1.0
    end

    @testset "TN reduced MPO canonicalization" begin
        sites_bath = siteinds("S=1/2", 1)
        s = sites_bath[1]
        ρ_tensor = ITensor(ComplexF64, prime(s), s)
        ρ_tensor[prime(s) => 1, s => 1] = 0.6 + 1e-12im
        ρ_tensor[prime(s) => 1, s => 2] = 0.2 + 0.3im
        ρ_tensor[prime(s) => 2, s => 1] = -0.1 + 0.4im
        ρ_tensor[prime(s) => 2, s => 2] = 0.4 - 1e-12im
        ρ_raw = MPO([ρ_tensor])
        raw_matrix = test_mpo_to_matrix(ρ_raw)

        ρ_canonical = CoolingTNS._canonical_reduced_density_mpo(ρ_raw)
        canonical_matrix = test_mpo_to_matrix(ρ_canonical)
        expected_matrix = 0.5 * (raw_matrix + raw_matrix')
        expected_matrix /= tr(expected_matrix)

        @test !ishermitian(raw_matrix)
        @test ishermitian(canonical_matrix)
        @test tr(canonical_matrix) ≈ 1.0 atol=1e-14
        @test canonical_matrix ≈ expected_matrix atol=1e-14
    end

    @testset "TN bath sampling uses the same convention" begin
        N = 2
        sites_sys = siteinds("S=1/2", N)
        sites_sb = siteinds("S=1/2", 2N)
        ψ_sys = MPS(sites_sys, "Up")
        ψ_sb = CoolingTNS.appendzeros_MPS(ψ_sys, sites_sb, "XX")

        bath_sample, _ = CoolingTNS.sample_bath(ψ_sb)

        @test bath_sample == fill(2, N)
        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_mc_state, bath_sample, N
        ) ≈ -1.0
    end

    @testset "TN density-matrix bath trace is measured" begin
        N = 2
        sites_sb = siteinds("S=1/2", CoolingTNS.interleaved_total_sites(N))
        sites_sys = CoolingTNS.interleaved_system_indices(sites_sb, N)
        sites_bath = CoolingTNS.interleaved_bath_indices(sites_sb, N)

        ρ_sys = MPO(sites_sys, "Id") / 2.0^N
        ρ_sb = CoolingTNS.appendzeros_MPO(ρ_sys, sites_sb, "XX")
        ρ_bath = CoolingTNS._reduced_bath_density_mpo(ρ_sb, sites_sb, N)

        tn_dm_state = CoolingTNS.QuantumState(
            CoolingTNS.TNBackend(),
            CoolingTNS.DensityMatrix(),
            CoolingTNS.TrotterEvolution(),
            nothing,
        )

        @test CoolingTNS.compute_bath_magnetization(
            CoolingTNS.TNBackend(), tn_dm_state, ρ_bath, sites_bath
        ) ≈ -1.0
    end

    @testset "TN reduced density entry points share system convention" begin
        N = 2
        backend = CoolingTNS.TNBackend()
        ham_params = CoolingTNS.IsingParameters(N, 1.0, -2.0)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 0.2, 1.0)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.TrotterEvolution();
            Dmax=20,
            cutoff=1e-10,
            tau=0.1,
        )

        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        state0 = CoolingTNS.setup_initial_state(problem, sim_params, "identity", 0.0)
        ρ_sb = CoolingTNS.prepare_combined_state(problem, state0)
        sites = problem.extra.sites
        sites_bath = CoolingTNS.interleaved_bath_indices(sites, N)

        expected_system = CoolingTNS._reduced_system_density_mpo(ρ_sb, sites, N)
        expected_system_matrix = test_mpo_to_matrix(expected_system)

        state1, bath_mag = CoolingTNS.process_bath_and_update(
            problem,
            ρ_sb,
            state0,
            sim_params,
        )
        processed_system, bath_info = CoolingTNS.process_bath(
            backend,
            CoolingTNS.DensityMatrix(),
            ρ_sb,
            N,
            N,
        )
        traced_system = CoolingTNS.trace_out_bath(backend, ρ_sb, N, N)
        expected_bath = CoolingTNS._reduced_bath_density_mpo(ρ_sb, sites, N)
        expected_bath_mag = CoolingTNS.compute_bath_magnetization(
            backend,
            state0,
            expected_bath,
            sites_bath,
        )

        @test test_mpo_to_matrix(state1.state) ≈ expected_system_matrix atol=1e-12
        @test test_mpo_to_matrix(processed_system) ≈ expected_system_matrix atol=1e-12
        @test test_mpo_to_matrix(traced_system) ≈ expected_system_matrix atol=1e-12
        @test bath_info === nothing
        @test bath_mag ≈ expected_bath_mag atol=1e-12
        @test ishermitian(expected_system_matrix)
        @test tr(expected_system_matrix) ≈ 1.0 atol=1e-12
    end

    @testset "TN density-matrix cooling populates bath_mag_list" begin
        N = 2
        backend = CoolingTNS.TNBackend()
        ham_params = CoolingTNS.IsingParameters(N, 1.0, -2.0)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 0.2, 1.0)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.TrotterEvolution();
            Dmax=20,
            cutoff=1e-10,
            tau=0.1,
        )

        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        state0 = CoolingTNS.setup_initial_state(problem, sim_params, "identity", 0.0)
        results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)

        @test haskey(results, CoolingTNS.RESULT_BATH_MAGNETIZATION)
        @test results[CoolingTNS.RESULT_BATH_MAGNETIZATION][2] ≈ -1.0 atol=1e-10
    end

    @testset "ED Monte Carlo cooling populates bath_mag_list" begin
        N = 2
        backend = CoolingTNS.EDBackend()
        ham_params = CoolingTNS.IsingParameters(N, 1.0, -2.0)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 0.0, 1.0)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0,
        )

        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
        results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)

        @test haskey(results, CoolingTNS.RESULT_BATH_MAGNETIZATION)
        # With g=0 and te=0, the bath is sampled from the deterministic XX
        # ground state; this is not a statistical MCWF assertion.
        @test results[CoolingTNS.RESULT_BATH_MAGNETIZATION][2] ≈ -1.0 atol=1e-12
    end

    @testset "ED density-matrix cooling returns system state and bath_mag_list" begin
        N = 2
        backend = CoolingTNS.EDBackend()
        ham_params = CoolingTNS.IsingParameters(N, 1.0, -2.0)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 0.2, 1.0)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0,
        )

        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        state0 = CoolingTNS.setup_initial_state(problem, sim_params, "identity", 0.0)
        ρ_sb = CoolingTNS.prepare_combined_state(problem, state0)
        ρ_evolved = CoolingTNS.evolve_cooling_step(
            problem,
            ρ_sb,
            coupling_params.te,
            sim_params,
            ham_params,
        )
        state1, bath_mag = CoolingTNS.process_bath_and_update(
            problem,
            ρ_evolved,
            state0,
            sim_params,
        )
        results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)

        @test state1.state.n_qubits == N
        @test bath_mag ≈ -1.0 atol=1e-12
        @test haskey(results, CoolingTNS.RESULT_BATH_MAGNETIZATION)
        @test results[CoolingTNS.RESULT_BATH_MAGNETIZATION][2] ≈ -1.0 atol=1e-12
    end
end
