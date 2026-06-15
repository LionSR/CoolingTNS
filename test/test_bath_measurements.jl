using Test
using CoolingTNS
using ITensors
using ITensorMPS

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
        sites_sb = siteinds("S=1/2", 2N)
        sites_sys = sites_sb[1:2:2N-1]
        sites_bath = sites_sb[2:2:2N]

        ρ_sys = MPO(sites_sys, "Id") / 2.0^N
        ρ_sb = CoolingTNS.appendzeros_MPO(ρ_sys, sites_sb, "XX")
        ρ_bath = CoolingTNS.partial_trace_system(ρ_sb, sites_sb, sites_bath)
        ρ_bath /= tr(ρ_bath)

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
end
