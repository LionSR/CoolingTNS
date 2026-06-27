using Test
using CoolingTNS
using LinearAlgebra
using Random

# Keep this file directly runnable, matching the other cross-backend tests that
# use the shared dense MPO converter outside `test/runtests.jl`.
@isdefined(test_mpo_to_matrix) || include("test_helpers.jl")

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

    @testset "cycle sequence helper fixes the protocol convention" begin
        @test CoolingTNS.MULTI_FREQUENCY_SCHEDULES == (:round_robin, :descending, :random)
        @test CoolingTNS.parse_multi_frequency_schedule(:descending) == :descending
        @test CoolingTNS.parse_multi_frequency_schedule("descending") == :descending
        @test CoolingTNS.multi_frequency_schedule_token(:round_robin) == "rr"
        @test CoolingTNS.multi_frequency_schedule_token("descending") == "desc"
        @test CoolingTNS.multi_frequency_schedule_token(:random) == "rand"
        @test_throws ArgumentError CoolingTNS.parse_multi_frequency_schedule("bad")

        for (schedule, token) in ((:round_robin, "rr"), (:descending, "desc"), (:random, "rand"))
            filename_params = CoolingTNS.MultiFrequencyCouplingParameters(
                "XX",
                0.1,
                5,
                1.25,
                [0.5, 1.0, 1.5];
                randomize_times=false,
                schedule=schedule,
            )
            filename = CoolingTNS.create_filename(
                ham_params,
                filename_params,
                sim_params,
                backend,
            )
            @test occursin("sched$token", filename)
        end

        sequence_params = CoolingTNS.MultiFrequencyCouplingParameters(
            "XX",
            0.1,
            5,
            1.25,
            [0.5, 1.0, 1.5];
            randomize_times=false,
            schedule=:round_robin,
        )
        sequence = CoolingTNS.multi_frequency_cycle_sequence(sequence_params)
        @test sequence.delta_indices == [1, 2, 3, 1, 2]
        @test isequal(sequence.delta_list, [NaN, 0.5, 1.0, 1.5, 0.5, 1.0])
        @test isequal(sequence.te_list, [NaN, 1.25, 1.25, 1.25, 1.25, 1.25])

        fourth_choice = CoolingTNS.multi_frequency_cycle_choice(sequence_params, 4)
        @test fourth_choice == (delta_index=1, delta=0.5, te=1.25)
        @test_throws ArgumentError CoolingTNS.multi_frequency_cycle_choice(sequence_params, 0)

        descending_params = CoolingTNS.MultiFrequencyCouplingParameters(
            "XX",
            0.1,
            5,
            1.25,
            [0.5, 1.0, 1.5];
            randomize_times=false,
            schedule=:descending,
        )
        descending_sequence = CoolingTNS.multi_frequency_cycle_sequence(descending_params)
        @test descending_sequence.delta_indices == [3, 2, 1, 3, 2]
        @test isequal(descending_sequence.delta_list, [NaN, 1.5, 1.0, 0.5, 1.5, 1.0])
        @test isequal(descending_sequence.te_list, sequence.te_list)
        @test CoolingTNS.multi_frequency_cycle_choice(descending_params, 4) ==
              (delta_index=3, delta=1.5, te=1.25)

        random_params = CoolingTNS.MultiFrequencyCouplingParameters(
            "XX",
            0.1,
            4,
            0.75,
            [0.5, 1.0, 1.5];
            randomize_times=true,
            schedule=:random,
        )
        random_sequence_a = CoolingTNS.multi_frequency_cycle_sequence(
            random_params; rng=MersenneTwister(17)
        )
        random_sequence_b = CoolingTNS.multi_frequency_cycle_sequence(
            random_params; rng=MersenneTwister(17)
        )
        @test random_sequence_a.delta_indices == random_sequence_b.delta_indices
        @test isequal(random_sequence_a.delta_list, random_sequence_b.delta_list)
        @test isequal(random_sequence_a.te_list, random_sequence_b.te_list)
        @test all(i -> 1 <= i <= length(random_params.delta_values), random_sequence_a.delta_indices)
        @test all(t -> 0 <= t <= 2 * random_params.te, random_sequence_a.te_list[2:end])

        zero_step_params = CoolingTNS.MultiFrequencyCouplingParameters(
            "XX",
            0.1,
            0,
            0.75,
            [0.5, 1.0];
            randomize_times=true,
            schedule=:random,
        )
        zero_step_sequence = CoolingTNS.multi_frequency_cycle_sequence(zero_step_params)
        @test isempty(zero_step_sequence.delta_indices)
        @test isequal(zero_step_sequence.delta_list, [NaN])
        @test isequal(zero_step_sequence.te_list, [NaN])
    end

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
    @test results[CoolingTNS.RESULT_REQUESTED_STEPS] == mf_params.steps
    @test results[CoolingTNS.RESULT_COMPLETED_STEPS] == mf_params.steps
    @test results[CoolingTNS.RESULT_STOP_REASON] == ""

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

    descending_run_params = CoolingTNS.MultiFrequencyCouplingParameters(
        "XX",
        0.1,
        3,
        0.5,
        [0.5, 1.0, 1.5];
        randomize_times=false,
        schedule=:descending,
    )
    descending_problem = CoolingTNS.setup_problem(
        backend, ham_params, descending_run_params, sim_params)
    descending_initial = CoolingTNS.setup_initial_state(
        descending_problem, sim_params, "product", 0.0)
    descending_results = CoolingTNS.run_cooling(
        descending_problem,
        descending_initial,
        descending_run_params,
        sim_params,
        ham_params,
    )
    @test descending_results[CoolingTNS.RESULT_COMPLETED_STEPS] ==
          descending_run_params.steps
    @test isequal(descending_results[CoolingTNS.RESULT_DELTA_LIST],
                  [NaN, 1.5, 1.0, 0.5])
    @test all(isfinite, descending_results[CoolingTNS.RESULT_ENERGY])

    rng_mf_params = CoolingTNS.MultiFrequencyCouplingParameters(
        "XX",
        0.1,
        2,
        1.0,
        [gap, 2 * gap];
        randomize_times=true,
        schedule=:random,
    )
    rng_problem = CoolingTNS.setup_problem(backend, ham_params, rng_mf_params, sim_params)
    rng_state = CoolingTNS.setup_initial_state(rng_problem, sim_params, "product", 0.0)
    expected_sequence = CoolingTNS.multi_frequency_cycle_sequence(
        rng_mf_params; rng=MersenneTwister(23)
    )
    rng_results = CoolingTNS.run_cooling(
        rng_problem,
        rng_state,
        rng_mf_params,
        sim_params,
        ham_params;
        rng=MersenneTwister(23),
    )
    @test isequal(rng_results[CoolingTNS.RESULT_DELTA_LIST], expected_sequence.delta_list)
    @test isequal(rng_results[CoolingTNS.RESULT_TE_LIST], expected_sequence.te_list)

    stopped_state = CoolingTNS.setup_initial_state(problem_mf, sim_params, "product", 0.0)
    stopped_results = CoolingTNS.run_cooling(
        problem_mf,
        stopped_state,
        mf_params,
        sim_params,
        ham_params;
        stop_condition=info -> info.step == 2 ? "unit_test_stop" : nothing,
    )
    @test stopped_results[CoolingTNS.RESULT_REQUESTED_STEPS] == mf_params.steps
    @test stopped_results[CoolingTNS.RESULT_COMPLETED_STEPS] == 1
    @test stopped_results[CoolingTNS.RESULT_STOP_REASON] == "unit_test_stop"
    @test length(stopped_results[CoolingTNS.RESULT_ENERGY]) == 2
    @test length(stopped_results[CoolingTNS.RESULT_DELTA_LIST]) == 2
    @test isnan(stopped_results[CoolingTNS.RESULT_DELTA_LIST][1])
    @test isfinite(stopped_results[CoolingTNS.RESULT_DELTA_LIST][2])

    final_stop_state = CoolingTNS.setup_initial_state(problem_mf, sim_params, "product", 0.0)
    final_stop_results = CoolingTNS.run_cooling(
        problem_mf,
        final_stop_state,
        mf_params,
        sim_params,
        ham_params;
        stop_condition=info -> info.step == mf_params.steps + 1 ? :final_cycle_stop : nothing,
    )
    @test final_stop_results[CoolingTNS.RESULT_REQUESTED_STEPS] == mf_params.steps
    @test final_stop_results[CoolingTNS.RESULT_COMPLETED_STEPS] == mf_params.steps
    @test final_stop_results[CoolingTNS.RESULT_STOP_REASON] == "final_cycle_stop"
    @test length(final_stop_results[CoolingTNS.RESULT_ENERGY]) == mf_params.steps + 1

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
        @test problem_mf_tn.extra.coupling_params === mf_params_tn
        @test haskey(problem_mf_tn.extra, :gates_cache)
        @test haskey(problem_mf_tn.extra, :trotter_step_gates_cache)

        problem_mf_tn_reused = CoolingTNS.setup_tn_multifrequency_problem_from_system(
            backend_tn,
            ham_params_tn,
            mf_params_tn,
            sim_params_tn,
            problem_basic_tn.extra.sites,
            problem_basic_tn.H_sys,
            gap_tn,
            problem_basic_tn.e₀,
            problem_basic_tn.ϕ₀,
        )
        problem_mf_tn_reused_ref = CoolingTNS.setup_tn_multifrequency_problem_from_system(
            backend_tn,
            ham_params_tn,
            mf_params_tn,
            sim_params_tn,
            problem_basic_tn.extra.sites,
            problem_basic_tn.H_sys,
            gap_tn,
            problem_basic_tn.e₀,
            problem_basic_tn.ϕ₀,
        )
        @test problem_mf_tn_reused.H_sys === problem_basic_tn.H_sys
        @test problem_mf_tn_reused.ϕ₀ === problem_basic_tn.ϕ₀
        @test problem_mf_tn_reused.extra.sites === problem_basic_tn.extra.sites
        @test problem_mf_tn_reused.e₀ == problem_basic_tn.e₀
        @test problem_mf_tn_reused.extra.gap == gap_tn
        @test problem_mf_tn_reused.extra.coupling_params === mf_params_tn
        @test problem_mf_tn_reused_ref.extra.gates_cache !== problem_mf_tn_reused.extra.gates_cache
        @test problem_mf_tn_reused_ref.extra.trotter_step_gates_cache !==
            problem_mf_tn_reused.extra.trotter_step_gates_cache

        state_tn = CoolingTNS.setup_initial_state(problem_mf_tn_reused_ref, sim_params_tn, "product", 0.0)
        state_tn_reused = CoolingTNS.setup_initial_state(problem_mf_tn_reused, sim_params_tn, "product", 0.0)

        Random.seed!(3)
        results_tn = CoolingTNS.run_cooling(
            problem_mf_tn_reused_ref,
            state_tn,
            mf_params_tn,
            sim_params_tn,
            ham_params_tn,
        )
        Random.seed!(3)
        results_tn_reused = CoolingTNS.run_cooling(
            problem_mf_tn_reused,
            state_tn_reused,
            mf_params_tn,
            sim_params_tn,
            ham_params_tn,
        )

        @test haskey(results_tn, CoolingTNS.RESULT_ENERGY)
        @test haskey(results_tn, CoolingTNS.RESULT_DELTA_LIST)
        @test haskey(results_tn, CoolingTNS.RESULT_TE_LIST)
        @test length(results_tn[CoolingTNS.RESULT_ENERGY]) == mf_params_tn.steps + 1
        @test all(isfinite, results_tn[CoolingTNS.RESULT_ENERGY])
        @test isapprox(
            results_tn_reused[CoolingTNS.RESULT_ENERGY],
            results_tn[CoolingTNS.RESULT_ENERGY];
            rtol=1e-10,
            atol=1e-10,
        )
        @test isapprox(
            results_tn_reused[CoolingTNS.RESULT_GROUND_STATE_OVERLAP],
            results_tn[CoolingTNS.RESULT_GROUND_STATE_OVERLAP];
            rtol=1e-8,
            atol=1e-8,
        )
        @test isequal(
            results_tn_reused[CoolingTNS.RESULT_DELTA_LIST],
            results_tn[CoolingTNS.RESULT_DELTA_LIST],
        )
        @test isequal(
            results_tn_reused[CoolingTNS.RESULT_TE_LIST],
            results_tn[CoolingTNS.RESULT_TE_LIST],
        )

        # The TDVP branch is checked structurally here. A trajectory comparison
        # would make this small multi-frequency unit test substantially slower;
        # the helper contract needed by the validation driver is that it reuses
        # the same system objects and installs the H_cache form of `extra`.
        sim_params_tdvp_tn = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            Dmax=6,
            cutoff=1e-6,
            tau=0.2,
            pe=0.0,
        )
        mf_params_tdvp_tn = CoolingTNS.MultiFrequencyCouplingParameters(
            "XX",
            0.1,
            1,
            0.5,
            [gap_tn];
            randomize_times=false,
            schedule=:round_robin,
        )
        problem_mf_tdvp_reused = CoolingTNS.setup_tn_multifrequency_problem_from_system(
            backend_tn,
            ham_params_tn,
            mf_params_tdvp_tn,
            sim_params_tdvp_tn,
            problem_basic_tn.extra.sites,
            problem_basic_tn.H_sys,
            gap_tn,
            problem_basic_tn.e₀,
            problem_basic_tn.ϕ₀,
        )
        @test problem_mf_tdvp_reused.H_sys === problem_basic_tn.H_sys
        @test problem_mf_tdvp_reused.ϕ₀ === problem_basic_tn.ϕ₀
        @test haskey(problem_mf_tdvp_reused.extra, :H_cache)
        @test !haskey(problem_mf_tdvp_reused.extra, :gates_cache)
        @test problem_mf_tdvp_reused.extra.coupling_params === mf_params_tdvp_tn

        sites_sys = CoolingTNS.interleaved_system_indices(problem_basic_tn.extra.sites, N_tn)
        @test_throws ArgumentError CoolingTNS.setup_tn_multifrequency_problem_from_system(
            backend_tn,
            ham_params_tn,
            mf_params_tdvp_tn,
            sim_params_tdvp_tn,
            sites_sys,
            problem_basic_tn.H_sys,
            gap_tn,
            problem_basic_tn.e₀,
            problem_basic_tn.ϕ₀,
        )
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
                @test [x.stage for x in observed] == [:initial, :prepared, :evolved, :updated]
                @test [x.step for x in observed] == [1, 2, 2, 2]
                @test observed[1].state_length == N_tn
                @test observed[1].evolved_length == 0
                @test observed[2].state_length == N_tn
                @test observed[2].evolved_length == 2N_tn
                @test observed[3].state_length == N_tn
                @test observed[3].evolved_length == 2N_tn
                @test observed[4].state_length == N_tn
                @test observed[4].evolved_length == 0
            end
        end
    end

    @testset "TDVP evolution kwargs reach multi-frequency cooling" begin
        Random.seed!(4)

        backend_tn = CoolingTNS.TNBackend()
        N_tn = 2
        ham_params_tn = CoolingTNS.NiIsingParameters(N_tn, 1.0, -1.05, 0.5)
        sim_params_tn = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(),
            CoolingTNS.ContinuousEvolution();
            Dmax=4,
            cutoff=1e-6,
            tau=0.2,
            pe=0.0,
        )

        coupling_basic_tn = CoolingTNS.BasicCouplingParameters("XX", 0.1, 1, 0.2, nothing)
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
            0.2,
            [gap_tn];
            randomize_times=false,
            schedule=:round_robin,
        )

        problem_mf_tn = CoolingTNS.setup_problem(backend_tn, ham_params_tn, mf_params_tn, sim_params_tn)
        state_tn = CoolingTNS.setup_initial_state(problem_mf_tn, sim_params_tn, "product", 0.0)

        sweeps = Int[]
        observer = CoolingTNS.tdvp_sweep_observer((; sweep, kwargs...) -> begin
            push!(sweeps, sweep)
            return nothing
        end)

        results_tn = CoolingTNS.run_cooling(
            problem_mf_tn,
            state_tn,
            mf_params_tn,
            sim_params_tn,
            ham_params_tn;
            evolution_kwargs=(tdvp_sweep_observer! = observer,),
        )

        @test haskey(results_tn, CoolingTNS.RESULT_ENERGY)
        @test sweeps == [1]
    end

    @testset "dynamic density-channel detunings match ED and TN" begin
        # Setting J=0 leaves only disjoint local system-bath pairs.  The ED and
        # TN Trotter channels should then agree to roundoff; nonzero J would
        # introduce ordinary Trotter-splitting error, which is not this test's
        # target.
        ham_match = CoolingTNS.NiIsingParameters(2, 0.0, -0.7, 0.3)
        mf_match = CoolingTNS.MultiFrequencyCouplingParameters(
            "XZ",
            0.23,
            2,
            0.4,
            [0.4, 0.9];
            randomize_times=false,
            schedule=:descending,
        )
        sim_ed_match = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.TrotterEvolution();
            tau=0.4,
            pe=0.0,
        )
        sim_tn_match = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.TrotterEvolution();
            Dmax=64,
            cutoff=1e-14,
            tau=0.4,
            pe=0.0,
        )

        problem_ed_match = CoolingTNS.setup_problem(
            CoolingTNS.EDBackend(),
            ham_match,
            mf_match,
            sim_ed_match,
        )
        problem_tn_match = CoolingTNS.setup_problem(
            CoolingTNS.TNBackend(),
            ham_match,
            mf_match,
            sim_tn_match,
        )
        @test problem_ed_match.e₀ ≈ problem_tn_match.e₀ atol=1e-12

        final_ed_state = Ref{Any}(nothing)
        final_tn_state = Ref{Any}(nothing)
        capture_ed = info -> begin
            info.stage == :updated && (final_ed_state[] = info.state.state)
            return nothing
        end
        capture_tn = info -> begin
            info.stage == :updated && (final_tn_state[] = info.state.state)
            return nothing
        end

        results_ed_match = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                CoolingTNS.run_cooling(
                    problem_ed_match,
                    CoolingTNS.setup_initial_state(problem_ed_match, sim_ed_match, "theta", -0.2),
                    mf_match,
                    sim_ed_match,
                    ham_match;
                    step_observer=capture_ed,
                )
            end
        end
        results_tn_match = redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                CoolingTNS.run_cooling(
                    problem_tn_match,
                    CoolingTNS.setup_initial_state(problem_tn_match, sim_tn_match, "theta", -0.2),
                    mf_match,
                    sim_tn_match,
                    ham_match;
                    step_observer=capture_tn,
                )
            end
        end

        @test isequal(
            results_ed_match[CoolingTNS.RESULT_DELTA_LIST],
            [NaN, 0.9, 0.4],
        )
        @test isequal(
            results_tn_match[CoolingTNS.RESULT_DELTA_LIST],
            results_ed_match[CoolingTNS.RESULT_DELTA_LIST],
        )
        @test isapprox(
            results_tn_match[CoolingTNS.RESULT_ENERGY],
            results_ed_match[CoolingTNS.RESULT_ENERGY];
            atol=1e-12,
            rtol=1e-12,
        )

        @test final_ed_state[] !== nothing
        @test final_tn_state[] !== nothing
        ρ_ed = final_ed_state[].data
        ρ_tn = test_mpo_to_matrix(final_tn_state[])
        @test ρ_tn ≈ ρ_ed atol=1e-12
        @test tr(ρ_ed) ≈ 1.0 + 0.0im atol=1e-12
        @test tr(ρ_tn) ≈ 1.0 + 0.0im atol=1e-12
    end
end
