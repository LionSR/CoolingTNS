using Test
using CoolingTNS

module ModeCoolingDiagnosticScript
using CoolingTNS
using Printf

include(joinpath(@__DIR__, "..", "scripts", "diagnostics", "mode_cooling_diagnostic.jl"))
end

@testset "Mode cooling diagnostic parses short exact controls" begin
    parse_args = ModeCoolingDiagnosticScript.parse_mode_cooling_diagnostic_args
    defaults = ModeCoolingDiagnosticScript.DEFAULT_MODE_COOLING_DIAGNOSTIC_CONFIG

    @test defaults.N == 6
    @test defaults.steps == 50
    @test defaults.bc == :periodic
    @test defaults.coupling == "XX"

    cfg = parse_args([
        "--N", "4",
        "--steps", "3",
        "--J", "1.0",
        "--h", "-1.05",
        "--bc", "antiperiodic",
        "--coupling", "ZZ",
        "--g", "0.2",
        "--te", "1.0",
        "--init-angle", string(pi / 6),
        "--plot",
    ])
    @test cfg.N == 4
    @test cfg.steps == 3
    @test cfg.J == 1.0
    @test cfg.h == -1.05
    @test cfg.bc == :antiperiodic
    @test cfg.coupling == "ZZ"
    @test cfg.g == 0.2
    @test cfg.te == 1.0
    @test cfg.init_angle ≈ pi / 6
    @test cfg.theta_code ≈ CoolingTNS.theta_code_from_initial_product_angle(pi / 6)
    @test cfg.do_plot

    theta_cfg = parse_args(["--theta-code", "0.0"])
    @test theta_cfg.theta_code == 0.0
    @test theta_cfg.init_angle ≈ pi / 4

    io = IOBuffer()
    @test parse_args(["--help"]; io=io) === nothing
    @test occursin("--steps INT", String(take!(io)))

    @test_throws ArgumentError parse_args(["--N", "5"])
    @test_throws ArgumentError parse_args(["--steps", "0"])
    @test_throws ArgumentError parse_args(["--te", "-1.0"])
    @test_throws ArgumentError parse_args(["--bc", "open"])
    @test_throws ArgumentError parse_args(["--coupling", "bad"])
    @test_throws ArgumentError parse_args(["--init-angle", "0.1", "--theta-code", "0.2"])
    @test_throws ArgumentError parse_args(["--N"])
    @test_throws ArgumentError parse_args(["--coupling"])
    @test_throws ArgumentError parse_args(["--unknown"])
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
