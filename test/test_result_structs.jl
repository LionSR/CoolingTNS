using Test
using CoolingTNS

@testset "Result Struct Serialization" begin
    E = [1.0, 0.5]
    overlap = [0.2, 0.7]
    purity = [1.0, 0.9]
    bath_mag = [0.0, -1.0]
    bath_sample_mag = [-1.0, 0.0]

    @testset "DensityMatrixResults" begin
        results = CoolingTNS.DensityMatrixResults(E, overlap, purity)
        data = CoolingTNS.to_dict(results)

        @test data[CoolingTNS.RESULT_ENERGY] === E
        @test data[CoolingTNS.RESULT_GROUND_STATE_OVERLAP] === overlap
        @test data[CoolingTNS.RESULT_PURITY] === purity
        @test !haskey(data, CoolingTNS.RESULT_BATH_MAGNETIZATION)
        @test !haskey(data, "energy_list")
        @test !haskey(data, "gs_overlap_list")

        results_with_bath = CoolingTNS.DensityMatrixResults(
            E,
            overlap,
            purity;
            bath_magnetization_list=bath_mag,
        )
        data_with_bath = CoolingTNS.to_dict(results_with_bath)

        @test data_with_bath[CoolingTNS.RESULT_BATH_MAGNETIZATION] === bath_mag
        @test !haskey(data_with_bath, "bath_magnetization_list")
    end

    @testset "MonteCarloResults" begin
        E_traj = [1.0 0.8; 0.5 0.4]
        overlap_traj = [0.2 0.3; 0.7 0.8]
        E_std = [0.1, 0.05]
        overlap_std = [0.02, 0.03]
        results = CoolingTNS.MonteCarloResults(
            E,
            overlap,
            purity,
            E_traj,
            overlap_traj,
            2,
            E_std,
            overlap_std,
        )
        data = CoolingTNS.to_dict(results)

        @test data[CoolingTNS.RESULT_ENERGY] === E
        @test data[CoolingTNS.RESULT_GROUND_STATE_OVERLAP] === overlap
        @test data[CoolingTNS.RESULT_PURITY] === purity
        @test data[CoolingTNS.RESULT_N_TRAJECTORIES] == 2
        @test data[CoolingTNS.RESULT_ENERGY_TRAJECTORIES] === E_traj
        @test data[CoolingTNS.RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES] === overlap_traj
        @test data[CoolingTNS.RESULT_ENERGY_STD] === E_std
        @test data[CoolingTNS.RESULT_GROUND_STATE_OVERLAP_STD] === overlap_std
        @test !haskey(data, CoolingTNS.RESULT_BATH_MAGNETIZATION)
        @test !haskey(data, CoolingTNS.RESULT_BATH_SAMPLE_MAGNETIZATION)

        results_with_bath = CoolingTNS.MonteCarloResults(
            E,
            overlap,
            purity,
            E_traj,
            overlap_traj,
            2,
            E_std,
            overlap_std;
            bath_magnetization_list=bath_mag,
            bath_sample_magnetization_list=bath_sample_mag,
        )
        data_with_bath = CoolingTNS.to_dict(results_with_bath)

        @test data_with_bath[CoolingTNS.RESULT_BATH_MAGNETIZATION] === bath_mag
        @test data_with_bath[CoolingTNS.RESULT_BATH_SAMPLE_MAGNETIZATION] === bath_sample_mag
        @test !haskey(data_with_bath, "bath_magnetization_list")
        @test !haskey(data_with_bath, "bath_sample_magnetization_list")
    end

    @testset "TensorNetworkResults" begin
        results = CoolingTNS.TensorNetworkResults(E, overlap, bath_mag, nothing)
        data = CoolingTNS.to_dict(results)

        @test data[CoolingTNS.RESULT_ENERGY] === E
        @test data[CoolingTNS.RESULT_GROUND_STATE_OVERLAP] === overlap
        @test data[CoolingTNS.RESULT_PURITY] == ones(length(E))
        @test data[CoolingTNS.RESULT_BATH_MAGNETIZATION] === bath_mag
        @test !haskey(data, CoolingTNS.RESULT_BATH_SAMPLE_MAGNETIZATION)
        @test !haskey(data, CoolingTNS.RESULT_BOND_DIMS)
        @test !haskey(data, CoolingTNS.RESULT_TRUNCATION_ERRORS)
        @test data[CoolingTNS.RESULT_TRUNCATION_ERROR_HISTORY_STATUS] ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED
        @test !haskey(data, CoolingTNS.RESULT_RENYI_ENTROPY)
        @test !haskey(data, "energy_list")
        @test !haskey(data, "gs_overlap_list")
        @test !haskey(data, "bath_magnetization_list")
        @test !haskey(data, "bath_sample_magnetization_list")
        @test !haskey(data, "final_state")

        bond_dims = [[1], [1, 2]]
        truncation_errors = [0.0, 1e-8]
        renyi_entropy = [[0.0], [0.1]]
        final_state = "sentinel"
        full_results = CoolingTNS.TensorNetworkResults(
            E,
            overlap,
            purity,
            bond_dims,
            truncation_errors,
            renyi_entropy,
            bath_mag,
            final_state,
            bath_sample_magnetization_list=bath_sample_mag,
        )
        full_data = CoolingTNS.to_dict(full_results)

        @test full_data[CoolingTNS.RESULT_BOND_DIMS] === bond_dims
        @test full_data[CoolingTNS.RESULT_TRUNCATION_ERRORS] === truncation_errors
        @test full_data[CoolingTNS.RESULT_TRUNCATION_ERROR_HISTORY_STATUS] ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_MEASURED
        @test full_data[CoolingTNS.RESULT_RENYI_ENTROPY] === renyi_entropy
        @test full_data[CoolingTNS.RESULT_BATH_MAGNETIZATION] === bath_mag
        @test full_data[CoolingTNS.RESULT_BATH_SAMPLE_MAGNETIZATION] === bath_sample_mag
        @test full_data[CoolingTNS.RESULT_FINAL_STATE] === final_state
        @test !haskey(full_data, "bath_sample_magnetization_list")

        sample_only_results = CoolingTNS.TensorNetworkResults(
            E,
            overlap,
            purity,
            bond_dims,
            truncation_errors,
            renyi_entropy,
            nothing,
            nothing;
            bath_sample_magnetization_list=bath_sample_mag,
        )
        sample_only_data = CoolingTNS.to_dict(sample_only_results)

        @test !haskey(sample_only_data, CoolingTNS.RESULT_BATH_MAGNETIZATION)
        @test sample_only_data[CoolingTNS.RESULT_BATH_SAMPLE_MAGNETIZATION] === bath_sample_mag

        no_bath_results = CoolingTNS.TensorNetworkResults(
            E,
            overlap,
            purity,
            bond_dims,
            truncation_errors,
            renyi_entropy,
            nothing,
            nothing,
        )
        no_bath_data = CoolingTNS.to_dict(no_bath_results)

        @test !haskey(no_bath_data, CoolingTNS.RESULT_BATH_MAGNETIZATION)
        @test !haskey(no_bath_data, CoolingTNS.RESULT_BATH_SAMPLE_MAGNETIZATION)
    end

    @testset "Live density-matrix bath schema is representable" begin
        backend = CoolingTNS.EDBackend()
        ham_params = CoolingTNS.IsingParameters(2, 1.0, -2.0)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.0, 1, 0.2, 1.0)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(),
            CoolingTNS.ContinuousEvolution();
            pe=0.0,
        )

        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
        live_data = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)

        @test haskey(live_data, CoolingTNS.RESULT_BATH_MAGNETIZATION)

        struct_data = CoolingTNS.to_dict(CoolingTNS.DensityMatrixResults(
            Float64.(live_data[CoolingTNS.RESULT_ENERGY]),
            Float64.(live_data[CoolingTNS.RESULT_GROUND_STATE_OVERLAP]),
            Float64.(live_data[CoolingTNS.RESULT_PURITY]);
            bath_magnetization_list=Float64.(live_data[CoolingTNS.RESULT_BATH_MAGNETIZATION]),
        ))

        for key in (
            CoolingTNS.RESULT_ENERGY,
            CoolingTNS.RESULT_GROUND_STATE_OVERLAP,
            CoolingTNS.RESULT_PURITY,
            CoolingTNS.RESULT_BATH_MAGNETIZATION,
        )
            @test haskey(struct_data, key)
            @test struct_data[key] == live_data[key]
        end
    end

    @testset "Result schema constants include diagnostics" begin
        for key in (
            CoolingTNS.RESULT_BATH_MAGNETIZATION,
            CoolingTNS.RESULT_BATH_SAMPLE_MAGNETIZATION,
            CoolingTNS.RESULT_MODE_HK,
            CoolingTNS.RESULT_MODE_NK,
            CoolingTNS.RESULT_MODE_K_INDICES,
            CoolingTNS.RESULT_MODE_ENERGIES,
            CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES,
            CoolingTNS.RESULT_MODE_GF,
            CoolingTNS.RESULT_MODE_GF_SOURCE,
            CoolingTNS.RESULT_MODE_HK_TRAJECTORIES,
            CoolingTNS.RESULT_MODE_NK_TRAJECTORIES,
            CoolingTNS.RESULT_MODE_HK_STDERR,
            CoolingTNS.RESULT_MODE_NK_STDERR,
            CoolingTNS.RESULT_TE,
            CoolingTNS.RESULT_INIT_STATE,
            CoolingTNS.RESULT_INIT_THETA,
            CoolingTNS.RESULT_ENERGY_TRAJECTORIES,
            CoolingTNS.RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES,
            CoolingTNS.RESULT_ENERGY_STD,
            CoolingTNS.RESULT_GROUND_STATE_OVERLAP_STD,
            CoolingTNS.RESULT_BOND_DIMS,
            CoolingTNS.RESULT_TRUNCATION_ERRORS,
            CoolingTNS.RESULT_TRUNCATION_ERROR_HISTORY_STATUS,
            CoolingTNS.RESULT_RENYI_ENTROPY,
            CoolingTNS.RESULT_FINAL_STATE,
        )
            @test key in CoolingTNS.RESULT_KEYS
        end
        @test CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED == "not_recorded"
        @test CoolingTNS.TRUNCATION_ERROR_HISTORY_LEGACY_MISSING == "legacy_missing"
        @test CoolingTNS.TRUNCATION_ERROR_HISTORY_MEASURED == "measured"
        @test CoolingTNS.TRUNCATION_ERROR_HISTORY_EMPTY == "empty"
        @test CoolingTNS.TRUNCATION_ERROR_HISTORY_STATUSES == (
            CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED,
            CoolingTNS.TRUNCATION_ERROR_HISTORY_LEGACY_MISSING,
            CoolingTNS.TRUNCATION_ERROR_HISTORY_MEASURED,
            CoolingTNS.TRUNCATION_ERROR_HISTORY_EMPTY,
        )
        @test CoolingTNS.require_truncation_error_history_status_label("measured") ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_MEASURED
        @test CoolingTNS.truncation_error_history_status_label(Float64[]) ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED
        @test CoolingTNS.truncation_error_history_status_label(Float64[]; recorded=true) ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_EMPTY
        @test CoolingTNS.truncation_error_history_status_label([1e-12]; recorded=true) ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_MEASURED
        @test_throws ArgumentError CoolingTNS.require_truncation_error_history_status_label(
            "estimated",
        )
        @test all(key -> key in CoolingTNS.RESULT_KEYS,
                  CoolingTNS.RESULT_MODE_OBSERVABLE_PAYLOAD_KEYS)
    end
end
