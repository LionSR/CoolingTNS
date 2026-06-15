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
end
