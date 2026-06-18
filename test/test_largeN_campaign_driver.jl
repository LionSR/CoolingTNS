using Test

include(joinpath(@__DIR__, "..", "scripts", "validation",
                 "run_largeN_multifrequency_tn_scaling.jl"))

@testset "Large-N campaign driver Dmax ladder" begin
    cfg = parse_args([
        "--Ns", "64",
        "--R-values", "1,2,5,10",
        "--methods", "mcwf",
        "--steps", "4",
        "--Dmax-values", "160,320,640",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--outdir", tempdir(),
    ])

    @test campaign_dmax_values(cfg) == [160, 320, 640]
    cfgs = campaign_dmax_configs(cfg)
    @test [c["Dmax"] for c in cfgs] == [160, 320, 640]
    @test all(c -> c["Dmax_values"] == [160, 320, 640], cfgs)

    paths = output_path.(cfgs)
    @test length(unique(paths)) == 3
    @test occursin("Dmax160", paths[1])
    @test occursin("Dmax320", paths[2])
    @test occursin("Dmax640", paths[3])

    single_cfg = parse_args(["--Dmax", "80"])
    @test campaign_dmax_values(single_cfg) == [80]

    @test_throws ErrorException parse_args(["--Dmax-values", "160,0"])
    @test_throws ErrorException parse_args([
        "--Dmax-values", "160,320",
        "--output", joinpath(tempdir(), "one_file.h5"),
    ])
end
