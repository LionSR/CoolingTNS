using Test
using CoolingTNS

include(joinpath(@__DIR__, "..", "scripts", "validation", "largeN_scaling_helpers.jl"))

@testset "Large-N scaling helper functions" begin
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
