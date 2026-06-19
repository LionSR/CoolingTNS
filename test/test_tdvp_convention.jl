using Test

@isdefined(test_mpo_to_matrix) || include("test_helpers.jl")
include(joinpath(@__DIR__, "..", "scripts", "diagnostics", "tdvp_convention.jl"))

@testset "TN TDVP real-time convention" begin
    @test CoolingTNS._tdvp_real_time(0.5) == -0.5im
    @test CoolingTNS._tdvp_step_count(0.0, 0.5) == 0
    @test CoolingTNS._tdvp_step_count(0.19, 0.5) == 1
    @test CoolingTNS._tdvp_step_count(0.7, 0.3) == 3
    @test_throws ArgumentError CoolingTNS._tdvp_real_time(-0.1)
    @test_throws ArgumentError CoolingTNS._tdvp_step_count(-0.1, 0.5)
    @test_throws ArgumentError CoolingTNS._tdvp_step_count(0.1, 0.0)

    result = tdvp_convention_check(verbose=false)

    @test result.overlap_real > 1 - 1e-8
    @test abs(result.energy_evolved - result.energy_exact) < 1e-8
    @test result.norm_error < 1e-10
    @test result.overlap_nonunitary < 0.95
end

@testset "TDVP Krylov expansion evolves interleaved Ising terms" begin
    N = 2
    ham_params = CoolingTNS.IsingParameters(N, 1.0, -1.05, :open)
    coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.3, 1, 0.4, 0.5)
    sites = siteinds("S=1/2", CoolingTNS.interleaved_total_sites(N))
    H = CoolingTNS.construct_system_bath_hamiltonian(
        ham_params,
        CoolingTNS.TNBackend(),
        sites,
        coupling_params,
    )

    sites_sys = CoolingTNS.interleaved_system_indices(sites, N)
    ψ_sys = CoolingTNS._theta_product_mps(sites_sys, 0.0)
    ψ0 = CoolingTNS.appendzeros_MPS(ψ_sys, sites, "XX")
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.MonteCarloWavefunction(),
        CoolingTNS.ContinuousEvolution();
        Dmax=64,
        cutoff=1e-13,
        tau=0.05,
    )

    H_matrix = test_mpo_to_matrix(H)
    ψ0_vector = _tdvp_test_vector(ψ0, sites)
    ψ0_vector ./= norm(ψ0_vector)
    ψ_budgeted = CoolingTNS._tdvp_expand_state(H, copy(ψ0), 2, 1e-13, 1)
    budgeted_vector = _tdvp_test_vector(ψ_budgeted, sites)
    budgeted_vector ./= norm(budgeted_vector)
    @test maxlinkdim(ψ_budgeted) <= 2
    @test abs(dot(ψ0_vector, budgeted_vector)) > 1 - 1e-12

    zero_expanded = CoolingTNS._tdvp_expand_state(0.0 * H, copy(ψ0), 64, 1e-13, 1)
    zero_vector = _tdvp_test_vector(zero_expanded, sites)
    zero_vector ./= norm(zero_vector)
    @test all(isfinite, abs.(zero_vector))
    @test abs(dot(ψ0_vector, zero_vector)) > 1 - 1e-12

    capped_gate = exp(-0.37im * op("Z", sites[1]) * op("Z", sites[2]))
    ψ_partially_capped = apply(
        [capped_gate],
        MPS(sites, "X+");
        cutoff=1e-14,
        maxdim=2,
        move_sites_back=true,
    )
    @test maxlinkdim(ψ_partially_capped) == 2
    ψ_cap_expanded = CoolingTNS._tdvp_expand_state(H, copy(ψ_partially_capped), 2, 1e-13, 1)
    @test maxlinkdim(ψ_cap_expanded) > maxlinkdim(ψ_partially_capped)
    @test abs(inner(ψ_partially_capped, ψ_cap_expanded) /
              (norm(ψ_partially_capped) * norm(ψ_cap_expanded))) > 1 - 1e-12

    ψ_exact = exp(CoolingTNS._tdvp_real_time(coupling_params.te) * H_matrix) * ψ0_vector
    ψ_exact ./= norm(ψ_exact)

    ψ_unexpanded = CoolingTNS.evolve_state(
        ham_params,
        sim_params,
        CoolingTNS.TNBackend(),
        H,
        ψ0,
        coupling_params.te,
        sites;
        tdvp_expand_krylovdim=0,
    )
    unexpanded_vector = _tdvp_test_vector(ψ_unexpanded, sites)
    unexpanded_vector ./= norm(unexpanded_vector)

    ψ_expanded = CoolingTNS.evolve_state(
        ham_params,
        sim_params,
        CoolingTNS.TNBackend(),
        H,
        ψ0,
        coupling_params.te,
        sites,
    )
    expanded_vector = _tdvp_test_vector(ψ_expanded, sites)
    expanded_vector ./= norm(expanded_vector)

    @test abs(dot(ψ_exact, unexpanded_vector)) < 0.99
    @test abs(dot(ψ_exact, expanded_vector)) > 1 - 1e-10
    @test norm(ψ_exact - expanded_vector) < 1e-5
end
