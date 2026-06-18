using Test
using HDF5

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

    @test cfg["evolution_method"] == "trotter"
    @test sim_params_for("mcwf", cfg).evolution_method isa CoolingTNS.TrotterEvolution
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

    continuous_cfg = parse_args([
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--M-mcwf", "3",
        "--outdir", tempdir(),
    ])
    sim_params_continuous = sim_params_for("mcwf", continuous_cfg)
    @test sim_params_continuous.evolution_method isa CoolingTNS.ContinuousEvolution
    @test sim_params_continuous.n_trajectories == 3
    @test occursin("mcwf_continuous_steps", output_path(continuous_cfg))

    @test_throws ErrorException parse_args(["--evolution-method", "bad"])
    @test_throws ErrorException parse_args([
        "--methods", "mpo",
        "--evolution-method", "continuous",
    ])
    @test_throws ErrorException parse_args(["--Dmax-values", "160,0"])
    @test_throws ErrorException parse_args(["--Dmax-values", "320,320"])
    @test_throws ErrorException parse_args(["--Dmax-values", "160,320"])
    @test_throws ErrorException parse_args([
        "--Dmax-values", "160,320",
        "--delta-min", "0.5",
        "--delta-max", "3.0",
        "--output", joinpath(tempdir(), "one_file.h5"),
    ])

    path = tempname() * ".h5"
    try
        protocol = largeN_detuning_protocol(0.75; delta_min=0.5, delta_max=3.0)
        traj_rows = [
            Dict{String,Any}(
                "E" => [-2.0, -1.5, -1.0],
                "overlap" => [0.1, 0.2, 0.3],
                "purity" => [1.0, 1.0, 1.0],
                "sys_maxbond" => [1, 4, 8],
                "sys_meanbond" => [1.0, 3.0, 6.0],
                "evolved_maxbond" => [0, 7, 9],
                "evolved_meanbond" => [NaN, 5.0, 7.0],
                "delta_list" => [NaN, 0.5, 3.0],
                "final_bond_dims" => [4, 8],
                "elapsed" => 1.25,
            ),
        ]

        h5open(path, "w") do f
            write_run_group(f, "R2", traj_rows, -2.0, 8, protocol, [0.5, 3.0])
            gap_protocol = largeN_detuning_protocol(0.75; delta_max_factor=4.0)
            write_run_group(
                f, "R3_gap", traj_rows, -2.0, 8,
                gap_protocol, largeN_delta_values(gap_protocol, 3)
            )
        end
        h5open(path, "r") do f
            g = f["R2"]
            @test read(g["delta_values"]) == [0.5, 3.0]
            @test read(g["detuning_protocol_source"]) == "fixed_range"
            @test read(g["detuning_reference_gap"]) == 0.75
            @test read(g["detuning_delta_min"]) == 0.5
            @test read(g["detuning_delta_max"]) == 3.0
            @test isnan(read(g["detuning_delta_max_factor"]))
            @test read(g["detuning_fixed_across_dmax"]) == true

            g_gap = f["R3_gap"]
            @test read(g_gap["delta_values"]) == [0.75, 1.875, 3.0]
            @test read(g_gap["detuning_protocol_source"]) == "gap_scaled_range"
            @test read(g_gap["detuning_reference_gap"]) == 0.75
            @test read(g_gap["detuning_delta_min"]) == 0.75
            @test read(g_gap["detuning_delta_max"]) == 3.0
            @test read(g_gap["detuning_delta_max_factor"]) == 4.0
            @test read(g_gap["detuning_fixed_across_dmax"]) == false
        end
    finally
        rm(path; force=true)
    end
end

@testset "Large-N campaign driver TDVP setup path" begin
    cfg = parse_args([
        "--Ns", "4",
        "--R-values", "1",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "1",
        "--Dmax", "6",
        "--cutoff", "1e-6",
        "--tau", "0.2",
        "--delta-min", "0.5",
        "--delta-max", "0.5",
        "--outdir", tempdir(),
    ])

    backend = CoolingTNS.TNBackend()
    ham_params = CoolingTNS.NiIsingParameters(4, cfg["J"], cfg["hx"], cfg["hz"])
    sim_params = sim_params_for("mcwf", cfg)
    cp_base = CoolingTNS.BasicCouplingParameters(cfg["coupling"], cfg["g"], 1, cfg["te"], nothing)
    base_problem = CoolingTNS.setup_problem(backend, ham_params, cp_base, sim_params)
    gap = Float64(base_problem.extra.coupling_params.delta)

    detuning_protocol = largeN_detuning_protocol(gap, cfg)
    mf_params = CoolingTNS.MultiFrequencyCouplingParameters(
        cfg["coupling"],
        cfg["g"],
        cfg["steps"],
        cfg["te"],
        largeN_delta_values(detuning_protocol, 1);
        randomize_times=false,
        schedule=cfg["schedule_symbol"],
    )
    problem = CoolingTNS.setup_tn_multifrequency_problem_from_system(
        backend,
        ham_params,
        mf_params,
        sim_params,
        base_problem.extra.sites,
        base_problem.H_sys,
        gap,
        base_problem.e₀,
        base_problem.ϕ₀,
    )

    @test problem.H_sys === base_problem.H_sys
    @test problem.ϕ₀ === base_problem.ϕ₀
    @test haskey(problem.extra, :H_cache)
    @test !haskey(problem.extra, :gates_cache)
    @test problem.extra.coupling_params === mf_params
end
