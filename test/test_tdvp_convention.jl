using Test

include(joinpath(@__DIR__, "..", "scripts", "diagnostics", "tdvp_convention.jl"))

@testset "TN TDVP real-time convention" begin
    result = tdvp_convention_check(verbose=false)

    @test result.overlap_real > 1 - 1e-8
    @test abs(result.energy_evolved - result.energy_exact) < 1e-8
    @test result.norm_error < 1e-10
    @test result.overlap_nonunitary < 0.95
end
