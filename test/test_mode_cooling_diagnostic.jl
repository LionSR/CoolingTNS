using Test
using CoolingTNS

module ModeCoolingDiagnosticScript
using CoolingTNS
using Printf

include(joinpath(@__DIR__, "..", "scripts", "diagnostics", "mode_cooling_diagnostic.jl"))
end

@testset "Mode cooling diagnostic validates stored occupations" begin
    occupation_from_results = ModeCoolingDiagnosticScript._mode_occupation_from_diagnostic_results
    mode_hk = [-1.0 0.0; 1.0 -0.5]
    mode_nk = CoolingTNS.mode_occupation_from_hk(mode_hk)

    stored = Dict{String,Any}(
        CoolingTNS.RESULT_MODE_HK => mode_hk,
        CoolingTNS.RESULT_MODE_NK => copy(mode_nk),
    )
    @test occupation_from_results(stored) == mode_nk

    legacy = Dict{String,Any}(
        CoolingTNS.RESULT_MODE_HK => mode_hk,
    )
    @test occupation_from_results(legacy) == mode_nk

    bad_nk = copy(mode_nk)
    bad_nk[1, 1] = 0.25
    inconsistent = Dict{String,Any}(
        CoolingTNS.RESULT_MODE_HK => mode_hk,
        CoolingTNS.RESULT_MODE_NK => bad_nk,
    )
    @test_throws ArgumentError occupation_from_results(inconsistent)
end
