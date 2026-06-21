using Test
using CoolingTNS

include(joinpath(@__DIR__, "..", "scripts", "validation", "largeN_scaling_helpers.jl"))

@testset "Large-N scaling helper functions" begin
    @test LARGE_N_TRAJECTORY_SEED_RULE ==
          "trajectory_seed = base_seed + 1_000_000*N + 10_000*R + trajectory; " *
          "valid for 1 <= R < 100 and 1 <= trajectory < 10000"
    @test largeN_trajectory_seed(20260617, 64, 10, 1) == 84360618
    @test largeN_trajectory_seed(7, 2, 1, 3) == 2010010
    @test_throws ArgumentError largeN_trajectory_seed(7, 2, 0, 1)
    @test_throws ArgumentError largeN_trajectory_seed(7, 2, 100, 1)
    @test_throws ArgumentError largeN_trajectory_seed(7, 2, 1, 0)
    @test_throws ArgumentError largeN_trajectory_seed(7, 2, 1, 10_000)

    gap_protocol = largeN_detuning_protocol(0.5; delta_max_factor=6.0)
    @test gap_protocol.source == "gap_scaled_range"
    @test gap_protocol.reference_gap == 0.5
    @test gap_protocol.delta_min == 0.5
    @test gap_protocol.delta_max == 3.0
    @test gap_protocol.fixed_across_dmax == false
    @test largeN_delta_values(gap_protocol, 1) == [0.5]
    @test largeN_delta_values(gap_protocol, 3) == [0.5, 1.75, 3.0]

    fixed_protocol = largeN_detuning_protocol(
        0.75; delta_min=0.25, delta_max=1.25, delta_max_factor=6.0
    )
    @test fixed_protocol.source == "fixed_range"
    @test fixed_protocol.reference_gap == 0.75
    @test fixed_protocol.delta_min == 0.25
    @test fixed_protocol.delta_max == 1.25
    @test isnan(fixed_protocol.delta_max_factor)
    @test fixed_protocol.fixed_across_dmax == true
    @test largeN_delta_values(fixed_protocol, 2) == [0.25, 1.25]

    @test largeN_method_kind_from_name("mcwf") === :mcwf
    @test largeN_method_kind_from_name("MPO") === :mpo
    @test_throws ArgumentError largeN_method_kind_from_name("ed")

    fixed_protocol_negative_reference = largeN_detuning_protocol(
        -0.035; delta_min=0.5, delta_max=0.5, delta_max_factor=6.0
    )
    @test fixed_protocol_negative_reference.source == "fixed_range"
    @test fixed_protocol_negative_reference.reference_gap == -0.035
    @test fixed_protocol_negative_reference.delta_min == 0.5
    @test fixed_protocol_negative_reference.delta_max == 0.5

    @test_throws ArgumentError largeN_detuning_protocol(0.0; delta_max_factor=6.0)
    @test_throws ArgumentError largeN_detuning_protocol(0.5; delta_max_factor=0.5)
    @test_throws ArgumentError largeN_detuning_protocol(0.5; delta_min=0.5)
    @test_throws ArgumentError largeN_detuning_protocol(0.5; delta_min=1.0, delta_max=0.5)
    @test_throws ArgumentError largeN_detuning_protocol(NaN; delta_min=0.5, delta_max=0.5)
    @test_throws ArgumentError largeN_delta_values(fixed_protocol, 0)

    @test tn_method_maxdim(MonteCarloWavefunction(), 12) == 12
    @test tn_method_maxdim(DensityMatrix(), 12) == 48
    @test tn_trotter_maxdim(MonteCarloWavefunction(), 12) == 12
    @test tn_trotter_maxdim(DensityMatrix(), 12) == 48

    @test first_bond_saturation_cycle([1, 11, 50, 159], 320) == 0
    @test first_bond_saturation_cycle([1, 11, 50, 320], 320) == 3
    @test first_bond_saturation_cycle([320, 11, 50], 320) == 0
    @test first_bond_saturation_cycle([1, 320, 320], 320) == 1

    @test first_recorded_saturation_cycle([0, 0, 0]) == 0
    @test first_recorded_saturation_cycle([0, 4, 2]) == 2
    @test first_recorded_saturation_cycle([3, 1, 2]) == 1

    @test saturation_cycle_label(0) == "none"
    @test saturation_cycle_label(4) == "4"
    @test saturation_cycle_label(missing) == "n/a"
    @test bond_cap_status(0, 0) == "no_cap_hit"
    @test bond_cap_status(3, 0) == "not_converged_system_cap"
    @test bond_cap_status(0, 2) == "not_converged_evolved_cap"
    @test bond_cap_status(3, 2) == "not_converged_system_and_evolved_cap"
    @test bond_cap_status(0, 0, 4) == "not_converged_tdvp_sweep_cap"
    @test bond_cap_status(3, 0, 4) == "not_converged_system_and_tdvp_sweep_cap"
    @test bond_cap_status(0, 2, 4) == "not_converged_evolved_and_tdvp_sweep_cap"
    @test bond_cap_status(3, 2, 4) ==
          "not_converged_system_and_evolved_and_tdvp_sweep_cap"
    @test effective_bond_dimension_label(48, 0, 64) == "48"
    @test effective_bond_dimension_label(64, 2, 64) == ">=64"
    @test effective_bond_dimension_label(72, 2, 64) == ">=72"

    @test bond_dimension_quantiles([10, 20, 30, 40], [0.5, 0.75]) ≈ [25.0, 32.5]
    @test all(isnan, bond_dimension_quantiles(Int[], [0.5, 0.9]))

    @test bond_dimension_fraction_at_least([10, 20, 30, 40], 25) == 0.5
    @test isnan(bond_dimension_fraction_at_least(Int[], 25))
    @test bond_dimension_threshold_fractions([10, 20, 30, 40], 40, [0.5, 0.75]) ==
          [0.75, 0.5]

    system_max = [1 1; 6 9; 12 10]
    system_mean = [1.0 1.0; 4.0 5.0; 8.0 5.0]
    evolved_max = [99 99; 12 6; 8 14]
    evolved_mean = [NaN NaN; 8.0 7.0; 9.0 11.0]
    @test final_system_max_bond(system_max) == 12
    @test final_system_mean_bond(system_mean) == 6.5
    @test peak_evolved_max_bond(evolved_max) == 14
    @test peak_evolved_mean_bond(evolved_mean) == 10.0
end
