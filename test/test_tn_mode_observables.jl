using Test
using CoolingTNS
using ITensors
using ITensorMPS

@testset "TN Mode Observables" begin
    @testset "MPS h_k agrees with ED for X+ product state" begin
        N = 4
        J, h = 1.0, 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        sites = siteinds("S=1/2", N)

        ψ_tn = MPS(sites, "X+")
        ψ_ed = CoolingTNS.create_theta_state_ed(N, "theta", 0.0)

        px_tn = measure_state_parity(ψ_tn, N)
        px_ed = measure_state_parity(ψ_ed, N)
        @test px_tn ≈ 1.0 atol=1e-10
        @test px_tn ≈ px_ed atol=1e-10

        gF = fermionic_bc(:periodic, 1)
        ks_tn, hk_tn, εk_tn = measure_all_mode_energies(ψ_tn, ham_params; gF=gF)
        ks_ed, hk_ed, εk_ed = measure_all_mode_energies(ψ_ed, ham_params; gF=gF)

        @test ks_tn == ks_ed
        @test εk_tn ≈ εk_ed atol=1e-12
        @test hk_tn ≈ hk_ed atol=1e-10

        for k in ks_tn
            @test measure_hk(ψ_tn, k, ham_params) ≈ measure_hk(ψ_ed, k, ham_params) atol=1e-10
        end
    end

    @testset "MPS h_k agrees with ED for all-up product state on explicit grids" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        sites = siteinds("S=1/2", N)

        ψ_tn = MPS(sites, "Up")
        ψ_ed = CoolingTNS.product_state_ed(N, 0)

        @test abs(measure_state_parity(ψ_tn, N)) < 1e-10
        @test abs(measure_state_parity(ψ_ed, N)) < 1e-10

        for gF in [-1, 1]
            ks_tn, hk_tn, εk_tn = measure_all_mode_energies(ψ_tn, ham_params; gF=gF)
            ks_ed, hk_ed, εk_ed = measure_all_mode_energies(ψ_ed, ham_params; gF=gF)

            @test ks_tn == ks_ed
            @test εk_tn ≈ εk_ed atol=1e-12
            @test hk_tn ≈ hk_ed atol=1e-10
        end
    end

    @testset "MPS mode observables reject open spin boundaries" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :open)
        sites = siteinds("S=1/2", N)
        ψ_tn = MPS(sites, "X+")

        @test_throws ArgumentError measure_hk(ψ_tn, 1//2, ham_params)
    end

    @testset "TN cooling records MPS mode observables" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 0, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(MonteCarloWavefunction(), ContinuousEvolution(); maxiter=20)

        problem = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state0 = setup_initial_state(problem, sim_params, "theta", 0.0)

        results = redirect_stdout(devnull) do
            run_cooling(problem, state0, coupling_params, sim_params, ham_params; measure_modes=true)
        end

        ks_expected, hk_expected, ε_expected =
            measure_all_mode_energies(state0.state, ham_params; gF=results[RESULT_MODE_GF])

        @test results[RESULT_MODE_K_INDICES] == ks_expected
        @test results[RESULT_MODE_ENERGIES] ≈ ε_expected atol=1e-12
        @test results[RESULT_MODE_HK][1, :] ≈ hk_expected atol=1e-10
        @test results[RESULT_MODE_NK][1, :] ≈ mode_occupation_from_hk(hk_expected) atol=1e-10
        @test results[RESULT_MODE_NK] ≈ mode_occupation_from_hk(results[RESULT_MODE_HK]) atol=1e-12
    end

    @testset "TN mode measurements skip MPS length mismatches" begin
        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 0, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(MonteCarloWavefunction(), ContinuousEvolution(); maxiter=20)

        problem = setup_problem(TNBackend(), ham_params, coupling_params, sim_params)
        state0 = setup_initial_state(problem, sim_params, "theta", 0.0)
        measurements = CoolingTNS.initialize_measurements(
            problem,
            state0,
            1;
            measure_modes=true,
            ham_params=ham_params,
        )

        CoolingTNS.perform_backend_measurements!(measurements, 1, problem, state0, ham_params, nothing)
        previous_hk = copy(measurements[RESULT_MODE_HK][1, :])
        previous_nk = copy(measurements[RESULT_MODE_NK][1, :])

        bad_sites = siteinds("S=1/2", N + 1)
        bad_state = QuantumState(
            TNBackend(),
            MonteCarloWavefunction(),
            ContinuousEvolution(),
            MPS(bad_sites, "X+"),
        )

        @test_logs(
            (:warn, "Dimension mismatch in measurements"),
            (:warn, "Skipping measurement due to dimension mismatch"),
            (:warn, "Skipping mode measurement due to dimension mismatch"),
            CoolingTNS.perform_backend_measurements!(
                measurements,
                2,
                problem,
                bad_state,
                ham_params,
                nothing,
            ),
        )
        @test measurements[RESULT_MODE_HK][2, :] == previous_hk
        @test measurements[RESULT_MODE_NK][2, :] == previous_nk
    end
end
