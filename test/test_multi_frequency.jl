using Test
using CoolingTNS

@testset "Multi-frequency Cooling" begin
    backend = CoolingTNS.EDBackend()
    N = 3
    ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5)

    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.DensityMatrix(),
        CoolingTNS.ContinuousEvolution();
        pe=0.0,
    )

    # Compute resonant gap via standard single-frequency setup
    coupling_basic = CoolingTNS.BasicCouplingParameters("XX", 0.1, 2, 1.0, nothing)
    problem_basic = CoolingTNS.setup_problem(backend, ham_params, coupling_basic, sim_params)
    gap = problem_basic.extra.coupling_params.delta
    @test gap !== nothing
    @test gap > 0

    # Multi-frequency protocol with a single Δ value (should match single-frequency behavior)
    mf_params = CoolingTNS.MultiFrequencyCouplingParameters(
        "XX",
        0.1,
        2,
        1.0,
        [gap];
        randomize_times=false,
        schedule=:round_robin,
    )

    problem_mf = CoolingTNS.setup_problem(backend, ham_params, mf_params, sim_params)
    initial_state = CoolingTNS.setup_initial_state(problem_mf, sim_params, "product", 0.0)

    results = CoolingTNS.run_cooling(problem_mf, initial_state, mf_params, sim_params, ham_params)

    @test haskey(results, "E_list")
    @test haskey(results, "GS_overlap_list")
    @test haskey(results, "delta_list")
    @test haskey(results, "te_list")

    @test length(results["E_list"]) == mf_params.steps + 1
    @test length(results["delta_list"]) == mf_params.steps + 1

    @test isnan(results["delta_list"][1])
    @test all(isfinite, results["delta_list"][2:end])
    @test all(isfinite, results["te_list"][2:end])

    # Cooling should reduce the system energy on average
    @test results["E_list"][end] <= results["E_list"][1] + 1e-10
end
