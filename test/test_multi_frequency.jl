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
end
