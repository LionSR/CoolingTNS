using Test

include(joinpath(@__DIR__, "..", "scripts", "diagnostics", "tdvp_convention.jl"))

@testset "TN TDVP real-time convention" begin
    @test CoolingTNS._tdvp_real_time(0.5) == -0.5im
    @test CoolingTNS._tdvp_step_count(0.0, 0.5) == 0
    @test CoolingTNS._tdvp_step_count(0.19, 0.5) == 1
    @test CoolingTNS._tdvp_step_count(0.7, 0.3) == 3
    @test_throws ArgumentError CoolingTNS._tdvp_real_time(-0.1)
    @test_throws ArgumentError CoolingTNS._tdvp_step_count(-0.1, 0.5)
    @test_throws ArgumentError CoolingTNS._tdvp_step_count(0.1, 0.0)

    result = tdvp_convention_check(verbose=false)

    @test result.overlap_real > 1 - 1e-8
    @test abs(result.energy_evolved - result.energy_exact) < 1e-8
    @test result.norm_error < 1e-10
    @test result.overlap_nonunitary < 0.95
end
