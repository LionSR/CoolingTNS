using Test
using CoolingTNS

@testset "Result Struct Serialization" begin
    E = [1.0, 0.5]
    overlap = [0.2, 0.7]
    purity = [1.0, 0.9]

    @testset "DensityMatrixResults" begin
        results = CoolingTNS.DensityMatrixResults(E, overlap, purity)
        data = CoolingTNS.to_dict(results)

        @test data[CoolingTNS.RESULT_ENERGY] === E
        @test data[CoolingTNS.RESULT_GROUND_STATE_OVERLAP] === overlap
        @test data[CoolingTNS.RESULT_PURITY] === purity
        @test !haskey(data, "energy_list")
        @test !haskey(data, "gs_overlap_list")
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
        @test data["E_trajectories"] === E_traj
        @test data["GS_trajectories"] === overlap_traj
        @test data["E_std"] === E_std
        @test data["GS_std"] === overlap_std
    end

    @testset "TensorNetworkResults" begin
        bath_mag = [1.0, -1.0]
        results = CoolingTNS.TensorNetworkResults(E, overlap, bath_mag, nothing)
        data = CoolingTNS.to_dict(results)

        @test data[CoolingTNS.RESULT_ENERGY] === E
        @test data[CoolingTNS.RESULT_GROUND_STATE_OVERLAP] === overlap
        @test data[CoolingTNS.RESULT_PURITY] == ones(length(E))
        @test data[CoolingTNS.RESULT_BATH_MAGNETIZATION] === bath_mag
        @test !haskey(data, "energy_list")
        @test !haskey(data, "gs_overlap_list")
        @test !haskey(data, "bath_magnetization_list")
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
        )
        full_data = CoolingTNS.to_dict(full_results)

        @test full_data["bond_dims"] === bond_dims
        @test full_data["truncation_errors"] === truncation_errors
        @test full_data["renyi_entropy"] === renyi_entropy
        @test full_data["final_state"] === final_state
    end
end
