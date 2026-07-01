using Test

module PhysicsInvestigationReportScript
using CoolingTNS
using LinearAlgebra
using Random
using Statistics
using Printf
using Test

include(joinpath(@__DIR__, "..", "scripts", "diagnostics", "physics_investigation_report.jl"))
end

@testset "Physics investigation report parses reproducible short controls" begin
    parse_args = PhysicsInvestigationReportScript.parse_physics_investigation_report_args
    defaults = PhysicsInvestigationReportScript.DEFAULT_PHYSICS_INVESTIGATION_REPORT_CONFIG

    @test !PhysicsInvestigationReportScript._physics_report_should_execute()
    @test defaults.sections == (:small_ed, :ed_trotter, :mc, :ed_scan)
    @test defaults.n_traj == 200
    @test defaults.ed_scan_N == 4
    @test defaults.ed_scan_steps == 50

    cfg = parse_args([
        "--only", "ed-scan",
        "--ed-scan-N", "2",
        "--ed-scan-steps", "3",
        "--traj", "5",
    ])
    @test cfg.sections == (:ed_scan,)
    @test cfg.ed_scan_N == 2
    @test cfg.ed_scan_steps == 3
    @test cfg.n_traj == 5

    multi = parse_args(["--only", "mc,ed-scan"])
    @test multi.sections == (:mc, :ed_scan)

    all_sections = parse_args(["--only", "all"])
    @test all_sections.sections == defaults.sections

    withenv("COOLINGTNS_TRAJ" => "7") do
        @test parse_args(String[]).n_traj == 7
    end

    io = IOBuffer()
    @test parse_args(["--help"]; io=io) === nothing
    usage = String(take!(io))
    @test occursin("--only LIST", usage)
    @test occursin("--ed-scan-steps INT", usage)
    @test occursin("COOLINGTNS_TRAJ", usage)

    withenv("COOLINGTNS_TRAJ" => "not-an-int") do
        help_io = IOBuffer()
        @test parse_args(["--help"]; io=help_io) === nothing
        @test occursin("COOLINGTNS_TRAJ", String(take!(help_io)))
        @test parse_args(["--only", "ed-scan"]).sections == (:ed_scan,)
        @test parse_args(["--traj", "5"]).n_traj == 5
        @test_throws ArgumentError parse_args(String[])
        @test_throws ArgumentError parse_args(["--only", "mc"])
    end

    @test_throws ArgumentError parse_args(["--only"])
    @test_throws ArgumentError parse_args(["--only", "bad"])
    @test_throws ArgumentError parse_args(["--traj", "0"])
    @test_throws ArgumentError parse_args(["--ed-scan-N", "1"])
    @test_throws ArgumentError parse_args(["--ed-scan-steps", "0"])
    @test_throws ArgumentError parse_args(["--unknown"])
end
