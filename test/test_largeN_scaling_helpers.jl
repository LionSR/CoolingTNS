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
end
