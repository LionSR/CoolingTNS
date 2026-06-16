using Test
using CoolingTNS
using Random

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

    @test haskey(results, CoolingTNS.RESULT_ENERGY)
    @test haskey(results, CoolingTNS.RESULT_GROUND_STATE_OVERLAP)
    @test haskey(results, CoolingTNS.RESULT_DELTA_LIST)
    @test haskey(results, CoolingTNS.RESULT_TE_LIST)

    @test length(results[CoolingTNS.RESULT_ENERGY]) == mf_params.steps + 1
    @test length(results[CoolingTNS.RESULT_DELTA_LIST]) == mf_params.steps + 1

    @test isnan(results[CoolingTNS.RESULT_DELTA_LIST][1])
    @test all(isfinite, results[CoolingTNS.RESULT_DELTA_LIST][2:end])
    @test all(isfinite, results[CoolingTNS.RESULT_TE_LIST][2:end])

    # Cooling should reduce the system energy on average
    @test results[CoolingTNS.RESULT_ENERGY][end] <= results[CoolingTNS.RESULT_ENERGY][1] + 1e-10

    @testset "TN DM+Trotter supports multi-frequency" begin
        Random.seed!(1)

        backend_tn = CoolingTNS.TNBackend()
        N_tn = 4
        ham_params_tn = CoolingTNS.NiIsingParameters(N_tn, 1.0, -1.05, 0.5)

        sim_params_tn = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.TrotterEvolution();
            Dmax=10,
            cutoff=1e-6,
            tau=0.2,
            pe=0.0,
        )

        # Compute resonant gap via standard single-frequency setup
        coupling_basic_tn = CoolingTNS.BasicCouplingParameters("XX", 0.1, 1, 0.5, nothing)
        problem_basic_tn = CoolingTNS.setup_problem(backend_tn, ham_params_tn, coupling_basic_tn, sim_params_tn)
        gap_tn = problem_basic_tn.extra.coupling_params.delta
        @test gap_tn !== nothing
        @test gap_tn > 0

        mf_params_tn = CoolingTNS.MultiFrequencyCouplingParameters(
            "XX",
            0.1,
            2,
            0.5,
            [gap_tn];
            randomize_times=true,
            schedule=:round_robin,
        )

        problem_mf_tn = CoolingTNS.setup_problem(backend_tn, ham_params_tn, mf_params_tn, sim_params_tn)
        state_tn = CoolingTNS.setup_initial_state(problem_mf_tn, sim_params_tn, "product", 0.0)

        results_tn = CoolingTNS.run_cooling(problem_mf_tn, state_tn, mf_params_tn, sim_params_tn, ham_params_tn)

        @test haskey(results_tn, CoolingTNS.RESULT_ENERGY)
        @test haskey(results_tn, CoolingTNS.RESULT_DELTA_LIST)
        @test haskey(results_tn, CoolingTNS.RESULT_TE_LIST)
        @test length(results_tn[CoolingTNS.RESULT_ENERGY]) == mf_params_tn.steps + 1
        @test all(isfinite, results_tn[CoolingTNS.RESULT_ENERGY])
    end
end
