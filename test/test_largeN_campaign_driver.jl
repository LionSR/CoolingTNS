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

    progress_path = joinpath(tempdir(), "largeN_progress.csv")
    progress_cfg = parse_args(["--progress-csv", progress_path])
    @test progress_cfg["progress_csv"] == progress_path

    tdvp_progress_cfg = parse_args([
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--progress-csv", progress_path,
        "--tdvp-outputlevel", "1",
        "--tdvp-sweep-progress",
    ])
    @test tdvp_progress_cfg["tdvp_outputlevel"] == 1
    @test tdvp_progress_cfg["tdvp_sweep_progress"] == true

    parallel_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "2,5",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "5",
        "--Dmax-values", "96,128",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--outdir", tempdir(),
        "--progress-csv", joinpath(tempdir(), "tdvp_progress.csv"),
        "--tdvp-sweep-progress",
        "--print-parallel-plan",
    ])
    @test parallel_cfg["print_parallel_plan"] == true
    parallel_jobs = parallel_plan_configs(parallel_cfg)
    @test length(parallel_jobs) == 4
    @test all(job -> job["Dmax_values"] === nothing, parallel_jobs)
    @test all(job -> length(job["Ns"]) == 1, parallel_jobs)
    @test all(job -> length(job["R_values"]) == 1, parallel_jobs)
    @test length(unique(output_path.(parallel_jobs))) == length(parallel_jobs)
    @test length(unique(job["progress_csv"] for job in parallel_jobs)) == length(parallel_jobs)
    @test all(job -> occursin("Dmax$(job["Dmax"])", output_path(job)), parallel_jobs)
    @test all(job -> occursin("_R$(only(job["R_values"]))_", job["progress_csv"]), parallel_jobs)

    parallel_commands = parallel_plan_commands(parallel_cfg)
    @test length(parallel_commands) == 4
    @test all(command -> occursin("--tdvp-sweep-progress", command), parallel_commands)
    @test all(command -> occursin("--evolution-method continuous", command), parallel_commands)
    @test all(command -> occursin("--delta-min 0.5051167496264384", command), parallel_commands)
    @test !any(command -> occursin("--Dmax-values", command), parallel_commands)
    parallel_plan_text = sprint(io -> print_parallel_plan(parallel_cfg; io=io))
    @test all(
        line -> startswith(line, "#") || startswith(line, "julia "),
        split(chomp(parallel_plan_text), '\n'),
    )
    @test shell_word("~/tdvp data") == "'~/tdvp data'"
    @test shell_word("~/tdvp_data") == "~/tdvp_data"

    mode_cfg = parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--Ns", "64",
        "--R-values", "1,2",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "5",
        "--Dmax", "32",
        "--h", "-0.75",
        "--measure-modes",
        "--outdir", tempdir(),
    ])
    @test mode_cfg["measure_modes"] == true
    @test mode_cfg["model"] == "ising"
    @test mode_cfg["bc"] == "periodic"
    @test mode_cfg["h"] == -0.75
    mode_ham = campaign_hamiltonian_parameters(64, mode_cfg)
    @test mode_ham.model isa CoolingTNS.IsingModel
    @test mode_ham.bc == :periodic
    @test mode_ham.params.h == -0.75
    @test CoolingTNS.supports_ising_fourier_observables(mode_ham)
    @test occursin("ising_bcperiodic", output_path(mode_cfg))
    mode_command = join(command_args_for_config(mode_cfg), " ")
    @test occursin("--model ising", mode_command)
    @test occursin("--bc periodic", mode_command)
    @test occursin("--h -0.75", mode_command)
    @test !occursin("--hx", mode_command)
    @test !occursin("--hz", mode_command)
    @test occursin("--measure-modes", mode_command)

    @test parse_args([
        "--model", "ising",
        "--bc", "antiperiodic",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--measure_modes",
    ])["measure_modes"]
    @test_throws ErrorException parse_args(["--measure-modes"])
    @test_throws ErrorException parse_args([
        "--model", "ising",
        "--bc", "open",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--measure-modes",
    ])
    @test_throws ErrorException parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--Ns", "5",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--measure-modes",
    ])
    @test_throws ErrorException parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--methods", "mpo",
        "--evolution-method", "trotter",
        "--measure-modes",
    ])

    single_output_plan = parse_args([
        "--methods", "mcwf",
        "--R-values", "5",
        "--output", joinpath(tempdir(), "single.h5"),
        "--print-parallel-plan",
    ])
    @test length(parallel_plan_configs(single_output_plan)) == 1
    @test_throws ErrorException parallel_plan_configs(parse_args([
        "--methods", "mcwf",
        "--R-values", "2,5",
        "--output", joinpath(tempdir(), "collision.h5"),
        "--print-parallel-plan",
    ]))

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
    @test_throws ErrorException parse_args(["--h", "0.5"])
    @test_throws ErrorException parse_args(["--bc", "periodic"])
    @test_throws ErrorException parse_args([
        "--model", "ising",
        "--bc", "antiperiodic",
        "--methods", "mcwf",
        "--evolution-method", "trotter",
    ])
    @test_throws ErrorException parse_args(["--tdvp-sweep-progress"])
    @test_throws ErrorException parse_args([
        "--progress-csv", progress_path,
        "--tdvp-sweep-progress",
    ])
    @test_throws ErrorException parse_args(["--Dmax-values", "160,0"])
    @test_throws ErrorException parse_args(["--Dmax-values", "320,320"])
    @test_throws ErrorException parse_args(["--Ns", "64,64"])
    @test_throws ErrorException parse_args(["--R-values", "2,2"])
    @test_throws ErrorException parse_args(["--methods", "mcwf,mcwf"])
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

    mode_path = tempname() * ".h5"
    try
        protocol = largeN_detuning_protocol(0.75; delta_min=0.5, delta_max=1.5)
        mode_hk_1 = [-1.0 0.0; -0.5 0.5; 0.0 1.0]
        mode_hk_2 = [-0.8 0.2; -0.4 0.6; 0.2 0.8]
        mode_nk_1 = CoolingTNS.mode_occupation_from_hk(mode_hk_1)
        mode_nk_2 = CoolingTNS.mode_occupation_from_hk(mode_hk_2)
        base_row = Dict{String,Any}(
            "E" => [-2.0, -1.5, -1.0],
            "overlap" => [0.1, 0.2, 0.3],
            "purity" => [1.0, 1.0, 1.0],
            "sys_maxbond" => [1, 4, 8],
            "sys_meanbond" => [1.0, 3.0, 6.0],
            "evolved_maxbond" => [0, 7, 9],
            "evolved_meanbond" => [NaN, 5.0, 7.0],
            "delta_list" => [NaN, 0.5, 1.5],
            "final_bond_dims" => [4, 8],
            "elapsed" => 1.25,
            CoolingTNS.RESULT_MODE_K_INDICES => [1//2, 3//2],
            CoolingTNS.RESULT_MODE_ENERGIES => [0.4, 0.8],
            CoolingTNS.RESULT_MODE_GF => -1,
            CoolingTNS.RESULT_MODE_GF_SOURCE => "state",
        )
        traj_rows = [
            merge(copy(base_row), Dict{String,Any}(
                CoolingTNS.RESULT_MODE_HK => mode_hk_1,
                CoolingTNS.RESULT_MODE_NK => mode_nk_1,
            )),
            merge(copy(base_row), Dict{String,Any}(
                "E" => [-2.0, -1.75, -1.25],
                CoolingTNS.RESULT_MODE_HK => mode_hk_2,
                CoolingTNS.RESULT_MODE_NK => mode_nk_2,
                "elapsed" => 1.5,
            )),
        ]

        h5open(mode_path, "w") do f
            write_run_group(f, "R_modes", traj_rows, -2.0, 8, protocol, [0.5, 1.5])
        end
        h5open(mode_path, "r") do f
            g = f["R_modes"]
            @test read(g[CoolingTNS.RESULT_MODE_HK]) ≈ (mode_hk_1 .+ mode_hk_2) ./ 2
            @test read(g[CoolingTNS.RESULT_MODE_NK]) ≈ (mode_nk_1 .+ mode_nk_2) ./ 2
            @test read(g[CoolingTNS.RESULT_MODE_K_INDICES]) == Float64.([1//2, 3//2])
            @test read(g[CoolingTNS.RESULT_MODE_ENERGIES]) == [0.4, 0.8]
            @test read(g[CoolingTNS.RESULT_MODE_GF]) == -1
            @test read(g[CoolingTNS.RESULT_MODE_GF_SOURCE]) == "state"
            @test size(read(g["mode_hk_trajectories"])) == (3, 2, 2)
            @test read(g["mode_hk_trajectories"])[:, :, 1] ≈ mode_hk_1
            @test read(g["mode_hk_trajectories"])[:, :, 2] ≈ mode_hk_2
            @test size(read(g["mode_nk_stderr"])) == (3, 2)
        end
    finally
        rm(mode_path; force=true)
    end
end

@testset "Large-N campaign progress CSV" begin
    context = (
        method="mcwf",
        evolution="continuous",
        R=2,
        trajectory=1,
        seed=123,
        Dmax=24,
        cutoff=1e-6,
        tau=0.2,
    )
    ham_params = CoolingTNS.NiIsingParameters(4, 1.0, -1.05, 0.5)
    measurements = Dict{String,Any}(
        CoolingTNS.RESULT_ENERGY => [-4.0, -1.0],
        CoolingTNS.RESULT_GROUND_STATE_OVERLAP => [0.1, 0.2],
    )
    info = (
        stage=:updated,
        step=2,
        measurements=measurements,
        delta=0.5,
        te=2.0,
    )
    row = progress_row(
        context,
        info,
        ham_params,
        -4.0,
        (max=8, mean=6.5),
        (max=13, mean=9.25),
        3.5,
    )

    @test row["method"] == "mcwf"
    @test row["evolution"] == "continuous"
    @test row["cycle"] == 1
    @test row["energy_per_site"] == -0.25
    @test row["relative_energy"] == relative_energy(-1.0, -4.0)
    @test row["system_max_bond"] == 8
    @test row["evolved_max_bond"] == 13

    initial_info = merge(info, (stage=:initial, step=1, delta=NaN, te=NaN))
    initial_row = progress_row(
        context,
        initial_info,
        ham_params,
        -4.0,
        (max=1, mean=1.0),
        (max=NaN, mean=NaN),
        0.1,
    )
    @test initial_row["cycle"] == 0
    @test initial_row["energy_per_site"] == -1.0
    @test isnan(initial_row["delta"])
    @test isnan(initial_row["evolved_max_bond"])

    prepared_info = merge(info, (stage=:prepared,))
    prepared_row = progress_row(
        context,
        prepared_info,
        ham_params,
        -4.0,
        (max=8, mean=6.5),
        (max=11, mean=7.25),
        2.0,
    )
    @test prepared_row["cycle"] == 1
    @test isnan(prepared_row["energy_per_site"])
    @test isnan(prepared_row["relative_energy"])
    @test isnan(prepared_row["overlap"])
    @test prepared_row["system_max_bond"] == 8
    @test prepared_row["evolved_max_bond"] == 11

    evolved_info = merge(info, (stage=:evolved,))
    evolved_row = progress_row(
        context,
        evolved_info,
        ham_params,
        -4.0,
        (max=8, mean=6.5),
        (max=13, mean=9.25),
        2.5,
    )
    @test evolved_row["cycle"] == 1
    @test isnan(evolved_row["energy_per_site"])
    @test isnan(evolved_row["relative_energy"])
    @test isnan(evolved_row["overlap"])
    @test evolved_row["system_max_bond"] == 8
    @test evolved_row["evolved_max_bond"] == 13
    @test isnan(evolved_row["tdvp_sweep"])
    @test isnan(evolved_row["tdvp_time"])

    tdvp_context = (step=2, delta=0.5, te=2.0, sys_bs=(max=8, mean=6.5))
    sweep_row = tdvp_sweep_progress_row(
        context,
        tdvp_context,
        3,
        -0.6im,
        ham_params,
        (max=21, mean=14.5),
        4.5,
    )
    @test sweep_row["stage"] == "tdvp_sweep"
    @test sweep_row["cycle"] == 1
    @test isnan(sweep_row["energy_per_site"])
    @test sweep_row["system_max_bond"] == 8
    @test sweep_row["evolved_max_bond"] == 21
    @test sweep_row["tdvp_sweep"] == 3
    @test sweep_row["tdvp_time"] == 0.6

    path = tempname() * ".csv"
    try
        append_progress_csv_row(path, row)
        append_progress_csv_row(path, merge(row, Dict{String,Any}(
            "timestamp" => "contains,comma",
            "stage" => "evolved",
        )))
        lines = readlines(path)
        @test lines[1] == join(PROGRESS_CSV_COLUMNS, ",")
        @test length(lines) == 3
        @test occursin("\"contains,comma\"", lines[3])
        @test count(==(','), lines[1]) == length(PROGRESS_CSV_COLUMNS) - 1
    finally
        rm(path; force=true)
    end

    stale_path = tempname() * ".csv"
    try
        write(stale_path, "timestamp,N\nold,4\n")
        @test_throws ArgumentError append_progress_csv_row(stale_path, row)
    finally
        rm(stale_path; force=true)
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
