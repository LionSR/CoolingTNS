using Test
using HDF5

include(joinpath(@__DIR__, "..", "scripts", "validation",
                 "run_largeN_multifrequency_tn_scaling.jl"))

@testset "Large-N bond-cap stop rule" begin
    @test isnothing(bond_cap_stop_reason(2, 10, [1, 5], [0, 6], [0, 9]))
    @test bond_cap_stop_reason(2, 10, [1, 10], [0, 6], [0, 9]) == "bond_cap"
    @test bond_cap_stop_reason(2, 10, [1, 5], [0, 10], [0, 9]) == "bond_cap"
    @test bond_cap_stop_reason(2, 10, [1, 5], [0, 6], [0, 10]) == "bond_cap"
end

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

    te1_cfg = parse_args(["--te", "1.0", "--outdir", tempdir()])
    te05_cfg = parse_args(["--te", "0.5", "--outdir", tempdir()])
    @test output_path(te1_cfg) != output_path(te05_cfg)
    @test occursin("_te1_", output_path(te1_cfg))
    @test occursin("_te0.5_", output_path(te05_cfg))

    default_init_cfg = parse_args(["--outdir", tempdir()])
    @test default_init_cfg["init_state"] == "product"
    @test default_init_cfg["theta"] == 0.0
    default_init_command = join(command_args_for_config(default_init_cfg), " ")
    @test occursin("--init-state product", default_init_command)
    @test occursin("--theta 0.0", default_init_command)

    theta_init_cfg = parse_args([
        "--init-state", "theta",
        "--theta", "0.0",
        "--outdir", tempdir(),
    ])
    @test theta_init_cfg["init_state"] == "theta"
    @test theta_init_cfg["theta"] == 0.0
    @test output_path(theta_init_cfg) != output_path(default_init_cfg)
    @test occursin("_inittheta_theta0_", output_path(theta_init_cfg))
    theta_init_command = join(command_args_for_config(theta_init_cfg), " ")
    @test occursin("--init-state theta", theta_init_command)
    @test occursin("--theta 0.0", theta_init_command)

    identity_init_cfg = parse_args([
        "--methods", "mpo",
        "--init-state", "identity",
    ])
    @test identity_init_cfg["init_state"] == "identity"
    @test occursin("_initidentity_", output_path(identity_init_cfg))
    @test !occursin("_initidentity_theta", output_path(identity_init_cfg))
    @test_throws ErrorException parse_args(["--init-state", "bad"])
    @test_throws ErrorException parse_args([
        "--methods", "mcwf",
        "--init-state", "identity",
    ])

    random_schedule_cfg = parse_args([
        "--schedule", "random",
        "--outdir", tempdir(),
    ])
    round_robin_cfg = parse_args([
        "--schedule", "round_robin",
        "--outdir", tempdir(),
    ])
    @test output_path(random_schedule_cfg) != output_path(round_robin_cfg)
    @test occursin("_schedrandom_", output_path(random_schedule_cfg))
    @test !occursin("_schedround_robin_", output_path(round_robin_cfg))

    random_time_cfg = parse_args([
        "--randomize-times",
        "--outdir", tempdir(),
    ])
    fixed_time_cfg = parse_args(["--outdir", tempdir()])
    @test random_time_cfg["randomize_times"] == true
    @test fixed_time_cfg["randomize_times"] == false
    @test output_path(random_time_cfg) != output_path(fixed_time_cfg)
    @test occursin("_randtime_", output_path(random_time_cfg))
    random_time_command = join(command_args_for_config(random_time_cfg), " ")
    @test occursin("--randomize-times", random_time_command)
    @test parse_args(["--randomize_times"])["randomize_times"] == true

    progress_path = joinpath(tempdir(), "largeN_progress.csv")
    progress_cfg = parse_args(["--progress-csv", progress_path])
    @test progress_cfg["progress_csv"] == progress_path

    tdvp_progress_cfg = parse_args([
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--progress-csv", progress_path,
        "--tdvp-outputlevel", "1",
        "--tdvp-sweep-progress",
        "--stop-on-bond-cap",
    ])
    @test tdvp_progress_cfg["tdvp_outputlevel"] == 1
    @test tdvp_progress_cfg["tdvp_sweep_progress"] == true
    @test tdvp_progress_cfg["stop_on_bond_cap"] == true

    tdvp_hdf5_sweep_cfg = parse_args([
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--tdvp-sweep-progress",
    ])
    @test tdvp_hdf5_sweep_cfg["progress_csv"] === nothing
    @test tdvp_hdf5_sweep_cfg["tdvp_sweep_progress"] == true

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
    @test !any(command -> occursin("--stop-on-bond-cap", command), parallel_commands)
    parallel_plan_text = sprint(io -> print_parallel_plan(parallel_cfg; io=io))
    @test all(
        line -> startswith(line, "#") || startswith(line, "julia "),
        split(chomp(parallel_plan_text), '\n'),
    )
    @test shell_word("~/tdvp data") == "'~/tdvp data'"
    @test shell_word("~/tdvp_data") == "~/tdvp_data"

    thread_plan_args = [
        "--Ns", "64",
        "--R-values", "2",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "5",
        "--Dmax", "96",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--print-parallel-plan",
    ]
    unthreaded_command = only(parallel_plan_commands(parse_args(thread_plan_args)))
    @test startswith(unthreaded_command, "julia ")
    @test !occursin("_NUM_THREADS", unthreaded_command)

    julia_threaded_command = only(parallel_plan_commands(parse_args(
        vcat(thread_plan_args, ["--plan-julia-threads", "2"])
    )))
    @test startswith(julia_threaded_command, "JULIA_NUM_THREADS=2 julia ")
    @test !occursin("OPENBLAS_NUM_THREADS", julia_threaded_command)

    blas_threaded_command = only(parallel_plan_commands(parse_args(
        vcat(thread_plan_args, ["--plan-blas-threads", "1"])
    )))
    @test startswith(
        blas_threaded_command,
        "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 julia ",
    )
    @test !occursin("JULIA_NUM_THREADS", blas_threaded_command)

    threaded_plan_cfg = parse_args(vcat(
        thread_plan_args,
        ["--plan-julia-threads", "2", "--plan-blas-threads", "1"],
    ))
    threaded_commands = parallel_plan_commands(threaded_plan_cfg)
    @test length(threaded_commands) == 1
    @test startswith(
        only(threaded_commands),
        "JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 julia ",
    )
    @test !occursin("--plan-julia-threads", only(threaded_commands))
    @test !occursin("--plan-blas-threads", only(threaded_commands))

    random_time_plan_cfg = parse_args(vcat(
        thread_plan_args,
        ["--randomize-times"],
    ))
    random_time_plan_command = only(parallel_plan_commands(random_time_plan_cfg))
    @test occursin("--randomize-times", random_time_plan_command)
    @test occursin("_randtime_", output_path(random_time_plan_cfg))

    stop_on_cap_cfg = parse_args([
        "--methods", "mcwf",
        "--R-values", "5",
        "--evolution-method", "continuous",
        "--stop-on-bond-cap",
        "--print-parallel-plan",
    ])
    stop_on_cap_command = only(parallel_plan_commands(stop_on_cap_cfg))
    @test occursin("--stop-on-bond-cap", stop_on_cap_command)
    @test occursin("_stopcap_steps", output_path(stop_on_cap_cfg))
    full_cap_cfg = parse_args([
        "--methods", "mcwf",
        "--R-values", "5",
        "--evolution-method", "continuous",
    ])
    @test output_path(stop_on_cap_cfg) != output_path(full_cap_cfg)
    explicit_stop_path = joinpath(tempdir(), "explicit_stop.h5")
    explicit_stop_cfg = parse_args([
        "--methods", "mcwf",
        "--R-values", "5",
        "--evolution-method", "continuous",
        "--stop-on-bond-cap",
        "--output", explicit_stop_path,
    ])
    @test output_path(explicit_stop_cfg) == explicit_stop_path

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
    mode_reference = campaign_base_detuning_reference(mode_ham, mode_cfg)
    @test mode_reference.source == "ising_mode_pair_reference"
    @test mode_reference.delta ≈ CoolingTNS.ising_mode_detuning_reference(mode_ham)
    @test mode_reference.delta > 0
    @test campaign_mode_detuning_preserves_px("XX")
    @test !campaign_mode_detuning_preserves_px("XY")
    @test !campaign_mode_detuning_preserves_px("ZZ")
    nonpreserving_mode_cfg = parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--Ns", "64",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--coupling", "ZZ",
        "--measure-modes",
        "--outdir", tempdir(),
    ])
    @test_throws ErrorException campaign_base_detuning_reference(mode_ham, nonpreserving_mode_cfg)
    explicit_nonpreserving_mode_cfg = parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--Ns", "64",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--coupling", "ZZ",
        "--delta-min", "0.5",
        "--delta-max", "1.0",
        "--measure-modes",
        "--outdir", tempdir(),
    ])
    explicit_nonpreserving_ref = campaign_base_detuning_reference(
        mode_ham,
        explicit_nonpreserving_mode_cfg,
    )
    @test explicit_nonpreserving_ref.source == "setup_gap"
    @test isnothing(explicit_nonpreserving_ref.delta)
    generic_cfg = parse_args(["--outdir", tempdir()])
    generic_reference = campaign_base_detuning_reference(
        campaign_hamiltonian_parameters(64, generic_cfg),
        generic_cfg,
    )
    @test generic_reference.source == "setup_gap"
    @test isnothing(generic_reference.delta)
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
    @test_throws ErrorException parse_args(["--plan-julia-threads", "0"])
    @test_throws ErrorException parse_args(["--plan-blas-threads", "0"])
    @test_throws ErrorException parse_args(["--plan-julia-threads", "1"])
    @test_throws ErrorException parse_args(["--plan-blas-threads", "1"])
    @test_throws ErrorException parse_args([
        "--methods", "mcwf",
        "--M-mcwf", "2",
        "--stop-on-bond-cap",
    ])
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
                "tdvp_sweep_maxbond" => [0, 6, 10],
                "delta_list" => [NaN, 0.5, 3.0],
                "te_list" => [NaN, 1.0, 1.25],
                "final_bond_dims" => [4, 8],
                "elapsed" => 1.25,
                "requested_steps" => 2,
                "completed_steps" => 2,
                "stop_reason" => "",
            ),
        ]

        h5open(path, "w") do f
            write_run_group(f, "R2", traj_rows, -2.0, 8, protocol, [0.5, 3.0])
            traj_rows_noncommon_te = [
                traj_rows[1],
                merge(copy(traj_rows[1]), Dict{String,Any}(
                    "te_list" => [NaN, 0.25, 1.75],
                    "elapsed" => 1.5,
                )),
            ]
            write_run_group(
                f, "R2_noncommon_te", traj_rows_noncommon_te,
                -2.0, 8, protocol, [0.5, 3.0],
            )
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
            @test read(g["requested_steps"]) == [2]
            @test read(g["completed_steps"]) == [2]
            @test read(g["stop_reasons"]) == [""]
            @test vec(read(g["tdvp_sweep_max_bond"])) == [0, 6, 10]
            @test read(g["tdvp_sweep_saturation_cycle"]) == [2]
            @test read(g["te_list_is_common"]) == true
            @test isequal(read(g["te_list"]), [NaN, 1.0, 1.25])
            @test isequal(vec(read(g["te_lists"])), [NaN, 1.0, 1.25])

            g_noncommon_te = f["R2_noncommon_te"]
            @test read(g_noncommon_te["te_list_is_common"]) == false
            @test !haskey(g_noncommon_te, "te_list")
            @test isequal(read(g_noncommon_te["te_list_first_trajectory"]), [NaN, 1.0, 1.25])
            @test isequal(read(g_noncommon_te["te_lists"])[:, 2], [NaN, 0.25, 1.75])

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
            "tdvp_sweep_maxbond" => [0, 6, 10],
            "delta_list" => [NaN, 0.5, 1.5],
            "te_list" => [NaN, 1.0, 1.0],
            "final_bond_dims" => [4, 8],
            "elapsed" => 1.25,
            "requested_steps" => 2,
            "completed_steps" => 2,
            "stop_reason" => "",
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

@testset "Large-N campaign mode measurement smoke run" begin
    mktempdir() do dir
        output = joinpath(dir, "mode_smoke.h5")
        cfg = parse_args([
            "--model", "ising",
            "--bc", "periodic",
            "--Ns", "2",
            "--R-values", "1",
            "--methods", "mcwf",
            "--evolution-method", "continuous",
            "--steps", "1",
            "--Dmax", "6",
            "--cutoff", "1e-6",
            "--tau", "0.2",
            "--h", "0.5",
            "--g", "0.0",
            "--te", "0.0",
            "--init-state", "theta",
            "--theta", "0.0",
            "--measure-modes",
            "--outdir", dir,
            "--output", output,
        ])

        path, summaries = redirect_stdout(devnull) do
            run_campaign(cfg)
        end

        @test path == output
        @test length(summaries) == 1
        @test isfile(output)

        h5open(output, "r") do f
            @test read(f["model"]) == "ising"
            @test read(f["bc"]) == "periodic"
            @test read(f["measure_modes"]) == true
            @test read(f["init_state"]) == "theta"
            @test read(f["theta"]) == 0.0
            @test read(f["randomize_times"]) == false
            @test read(f["h"]) == 0.5
            @test isnan(read(f["hx"]))
            @test isnan(read(f["hz"]))

            gm = f["N2/mcwf"]
            reference = CoolingTNS.ising_mode_detuning_reference(
                CoolingTNS.IsingParameters(2, 1.0, 0.5, :periodic)
            )
            @test read(gm["gap"]) ≈ reference
            @test read(gm["detuning_reference_gap_source"]) == "ising_mode_pair_reference"
            @test read(gm["detuning_protocol_source"]) == "gap_scaled_range"
            @test read(gm["detuning_reference_gap"]) ≈ reference
            @test read(gm["detuning_delta_min"]) ≈ reference
            @test read(gm["detuning_delta_max"]) ≈ 6.0 * reference
            @test read(gm["detuning_delta_max_factor"]) == 6.0
            @test read(gm["detuning_fixed_across_dmax"]) == false

            g = f["N2/mcwf/R1"]
            mode_hk = read(g[CoolingTNS.RESULT_MODE_HK])
            mode_nk = read(g[CoolingTNS.RESULT_MODE_NK])
            @test size(mode_hk) == (2, 2)
            @test size(mode_nk) == (2, 2)
            @test mode_nk ≈ CoolingTNS.mode_occupation_from_hk(mode_hk)
            @test all(isfinite, mode_hk)
            @test all(n -> -1e-12 <= n <= 1 + 1e-12, mode_nk)
            @test read(g[CoolingTNS.RESULT_MODE_GF]) == -1
            @test read(g[CoolingTNS.RESULT_MODE_GF_SOURCE]) == "state"
            @test read(g[CoolingTNS.RESULT_MODE_K_INDICES]) == Float64.([-1//2, 1//2])
            @test length(read(g[CoolingTNS.RESULT_MODE_ENERGIES])) == 2
            @test all(isfinite, read(g[CoolingTNS.RESULT_MODE_ENERGIES]))
            @test size(read(g["mode_hk_trajectories"])) == (2, 2, 1)
            @test size(read(g["mode_nk_trajectories"])) == (2, 2, 1)
            @test read(g["mode_hk_stderr"]) == zeros(2, 2)
            @test read(g["mode_nk_stderr"]) == zeros(2, 2)
            @test !haskey(g, CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION)
        end
    end
end

@testset "Large-N campaign TDVP sweep HDF5 without CSV" begin
    mktempdir() do dir
        output = joinpath(dir, "tdvp_sweep_hdf5.h5")
        cfg = parse_args([
            "--Ns", "2",
            "--R-values", "1",
            "--methods", "mcwf",
            "--evolution-method", "continuous",
            "--steps", "1",
            "--Dmax", "4",
            "--cutoff", "1e-6",
            "--tau", "0.2",
            "--te", "0.05",
            "--delta-min", "0.5",
            "--delta-max", "0.5",
            "--tdvp-sweep-progress",
            "--outdir", dir,
            "--output", output,
        ])
        @test cfg["progress_csv"] === nothing

        redirect_stdout(devnull) do
            run_campaign(cfg)
        end

        h5open(output, "r") do f
            @test read(f["tdvp_sweep_progress"]) == true
            @test read(f["randomize_times"]) == false
            g = f["N2/mcwf/R1"]
            tdvp_sweep_max_bond = read(g["tdvp_sweep_max_bond"])
            @test size(tdvp_sweep_max_bond) == (2, 1)
            @test tdvp_sweep_max_bond[1, 1] == 0
            @test tdvp_sweep_max_bond[2, 1] >= 1
            @test size(read(g["tdvp_sweep_saturation_cycle"])) == (1,)
        end
    end
end

@testset "Large-N campaign randomized-time HDF5" begin
    mktempdir() do dir
        output = joinpath(dir, "randomized_time.h5")
        cfg = parse_args([
            "--Ns", "2",
            "--R-values", "1",
            "--methods", "mcwf",
            "--evolution-method", "continuous",
            "--steps", "1",
            "--Dmax", "4",
            "--cutoff", "1e-6",
            "--tau", "0.2",
            "--te", "0.0",
            "--delta-min", "0.5",
            "--delta-max", "0.5",
            "--randomize-times",
            "--outdir", dir,
            "--output", output,
        ])
        @test cfg["randomize_times"] == true

        redirect_stdout(devnull) do
            run_campaign(cfg)
        end

        h5open(output, "r") do f
            @test read(f["init_state"]) == "product"
            @test read(f["theta"]) == 0.0
            @test read(f["randomize_times"]) == true
            g = f["N2/mcwf/R1"]
            @test read(g["te_list_is_common"]) == true
            @test isequal(read(g["te_list"]), [NaN, 0.0])
            @test isequal(vec(read(g["te_lists"])), [NaN, 0.0])
        end
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
