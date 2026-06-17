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
    initial_state_explicit_nothing = CoolingTNS.setup_initial_state(problem_mf, sim_params, "product", 0.0)
    results_explicit_nothing = CoolingTNS.run_cooling(
        problem_mf,
        initial_state_explicit_nothing,
        mf_params,
        sim_params,
        ham_params;
        step_observer=nothing,
    )

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
    @test results_explicit_nothing[CoolingTNS.RESULT_ENERGY] == results[CoolingTNS.RESULT_ENERGY]
    @test isequal(
        results_explicit_nothing[CoolingTNS.RESULT_DELTA_LIST],
        results[CoolingTNS.RESULT_DELTA_LIST],
    )
    @test isequal(
        results_explicit_nothing[CoolingTNS.RESULT_TE_LIST],
        results[CoolingTNS.RESULT_TE_LIST],
    )
    @test results_explicit_nothing[CoolingTNS.RESULT_GROUND_STATE_OVERLAP] ==
          results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP]

    @testset "Spectral detunings respect finite ED spectra" begin
        spectral_values = CoolingTNS.spectral_delta_values(
            ham_params, backend; R=5
        )

        @test length(spectral_values) == 5
        @test issorted(spectral_values)
        # These generic non-integrable parameters lift the simple Ising degeneracies.
        @test all(>(0), spectral_values)

        krylov_capped_values = CoolingTNS.spectral_delta_values(
            ham_params, backend; R=5, num_excitations=10, krylovdim=7
        )
        @test length(krylov_capped_values) == 5

        @test_throws ArgumentError CoolingTNS.spectral_delta_values(
            ham_params, backend; R=5, num_excitations=4
        )
        @test_throws ArgumentError CoolingTNS.spectral_delta_values(
            ham_params, backend; R=1, num_excitations=0
        )
        @test_throws ArgumentError CoolingTNS.compute_excitation_gaps(
            ham_params, backend; num_excitations=2^N
        )
        @test_throws ArgumentError CoolingTNS.compute_excitation_gaps(
            ham_params, backend; num_excitations=5, krylovdim=6
        )
        @test_throws ArgumentError CoolingTNS.spectral_delta_values(
            ham_params, backend; R=5, num_excitations=10, krylovdim=6
        )
        @test_throws ArgumentError CoolingTNS.spectral_delta_values(
            ham_params, backend; R=2^N
        )
    end

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

    @testset "step observer sees evolved TN state before bath processing" begin
        for (label, sim_method) in (
            ("MCWF", CoolingTNS.MonteCarloWavefunction()),
            ("MPO", CoolingTNS.DensityMatrix()),
        )
            @testset "$label observer contract" begin
                Random.seed!(2)

                backend_tn = CoolingTNS.TNBackend()
                N_tn = 4
                ham_params_tn = CoolingTNS.NiIsingParameters(N_tn, 1.0, -1.05, 0.5)

                sim_params_tn = CoolingTNS.UnifiedSimulationParameters(
                    sim_method,
                    CoolingTNS.TrotterEvolution();
                    Dmax=6,
                    cutoff=1e-6,
                    tau=0.2,
                    pe=0.0,
                )

                coupling_basic_tn = CoolingTNS.BasicCouplingParameters("XX", 0.1, 1, 0.5, nothing)
                problem_basic_tn = CoolingTNS.setup_problem(
                    backend_tn,
                    ham_params_tn,
                    coupling_basic_tn,
                    sim_params_tn,
                )
                gap_tn = problem_basic_tn.extra.coupling_params.delta

                mf_params_tn = CoolingTNS.MultiFrequencyCouplingParameters(
                    "XX",
                    0.1,
                    1,
                    0.5,
                    [gap_tn];
                    randomize_times=false,
                    schedule=:round_robin,
                )

                problem_mf_tn = CoolingTNS.setup_problem(backend_tn, ham_params_tn, mf_params_tn, sim_params_tn)
                state_tn = CoolingTNS.setup_initial_state(problem_mf_tn, sim_params_tn, "product", 0.0)

                observed = NamedTuple[]
                observer = info -> push!(
                    observed,
                    (
                        stage=info.stage,
                        step=info.step,
                        state_length=length(info.state.state),
                        evolved_length=info.evolved_state === nothing ? 0 : length(info.evolved_state),
                    ),
                )

                results_tn = CoolingTNS.run_cooling(
                    problem_mf_tn,
                    state_tn,
                    mf_params_tn,
                    sim_params_tn,
                    ham_params_tn;
                    step_observer=observer,
                )

                @test haskey(results_tn, CoolingTNS.RESULT_ENERGY)
                @test [x.stage for x in observed] == [:initial, :evolved, :updated]
                @test [x.step for x in observed] == [1, 2, 2]
                @test observed[1].state_length == N_tn
                @test observed[1].evolved_length == 0
                @test observed[2].state_length == N_tn
                @test observed[2].evolved_length == 2N_tn
                @test observed[3].state_length == N_tn
                @test observed[3].evolved_length == 0
            end
        end
    end
end
