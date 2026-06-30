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

@testset "Large-N campaign driver method defaults" begin
    @test parse_method_name(" MCWF ") == "mcwf"
    @test parse_evolution_method_name(" TROTTER ") == "trotter"
    @test_throws ErrorException parse_evolution_method_name("mcwf")
    @test_throws ErrorException parse_args(["--evolution-method", "mcwf"])
    @test_throws ErrorException parse_args(["--methods", "continuous"])
    @test_throws ErrorException parse_args(["--Ns", ""])
    @test_throws ErrorException parse_args(["--R-values", ","])
    @test_throws ErrorException parse_args(["--methods", ""])

    default_cfg = parse_args(["--outdir", tempdir()])
    @test default_cfg["Ns"] == [64]
    @test default_cfg["methods"] == ["mcwf"]
    @test basename(default_output_filename(default_cfg)) ==
          "largeN_multifrequency_tn_N64_R1-2-5-10_mcwf_trotter_steps40_Dmax40_g0.3_te2_tau0.2_seed20260617.h5"
    @test occursin("_mcwf_trotter_steps", output_path(default_cfg))
    @test occursin("_g0.3_te2_", output_path(default_cfg))
    weak_coupling_cfg = parse_args([
        "--g", "0.1",
        "--outdir", tempdir(),
    ])
    @test occursin("_g0.1_te2_", output_path(weak_coupling_cfg))
    @test basename(default_output_filename(weak_coupling_cfg)) !=
          basename(default_output_filename(default_cfg))
    default_command = join(command_args_for_config(default_cfg), " ")
    @test occursin("--methods mcwf", default_command)
    @test !occursin("--methods mpo,mcwf", default_command)

    quick_cfg = parse_args(["--quick", "--outdir", tempdir()])
    @test quick_cfg["Ns"] == [8]
    @test quick_cfg["methods"] == ["mpo", "mcwf"]
    @test quick_cfg["evolution_method"] == "trotter"
    @test occursin("_mpo-mcwf_trotter_steps", output_path(quick_cfg))
    quick_command = join(command_args_for_config(quick_cfg), " ")
    @test occursin("--methods mpo,mcwf", quick_command)

    continuous_then_quick_cfg = parse_args([
        "--evolution-method", "continuous",
        "--tdvp-sweep-progress",
        "--tdvp-outputlevel", "1",
        "--quick",
        "--outdir", tempdir(),
    ])
    @test continuous_then_quick_cfg["methods"] == ["mpo", "mcwf"]
    @test continuous_then_quick_cfg["evolution_method"] == "trotter"
    @test continuous_then_quick_cfg["evolution_method_values"] === nothing
    @test continuous_then_quick_cfg["tdvp_sweep_progress"] == false
    @test continuous_then_quick_cfg["tdvp_outputlevel"] == 0

    paired_then_quick_cfg = parse_args([
        "--evolution-method-values", "trotter,continuous",
        "--print-parallel-plan",
        "--tdvp-sweep-progress",
        "--quick",
        "--outdir", tempdir(),
    ])
    @test paired_then_quick_cfg["evolution_method"] == "trotter"
    @test paired_then_quick_cfg["evolution_method_values"] === nothing
    @test campaign_evolution_method_values(paired_then_quick_cfg) == ["trotter"]
    @test paired_then_quick_cfg["tdvp_sweep_progress"] == false

    quick_then_mpo_cfg = parse_args([
        "--quick",
        "--methods", "mpo",
        "--outdir", tempdir(),
    ])
    @test quick_then_mpo_cfg["methods"] == ["mpo"]

    mpo_then_quick_cfg = parse_args([
        "--methods", "mpo",
        "--quick",
        "--outdir", tempdir(),
    ])
    @test mpo_then_quick_cfg["methods"] == ["mpo", "mcwf"]

    quick_then_continuous_cfg = parse_args([
        "--quick",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--tdvp-sweep-progress",
        "--tdvp-outputlevel", "1",
        "--outdir", tempdir(),
    ])
    @test quick_then_continuous_cfg["methods"] == ["mcwf"]
    @test quick_then_continuous_cfg["evolution_method"] == "continuous"
    @test quick_then_continuous_cfg["tdvp_sweep_progress"] == true
    @test quick_then_continuous_cfg["tdvp_outputlevel"] == 1

    explicit_mpo_cfg = parse_args(["--methods", "mpo", "--outdir", tempdir()])
    @test explicit_mpo_cfg["methods"] == ["mpo"]
    @test occursin("_mpo_trotter_steps", output_path(explicit_mpo_cfg))
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
    @test campaign_ladder_configs(cfg) == cfgs

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

    te_ladder_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "2,5",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "40",
        "--Dmax", "96",
        "--te-values", "0.5,1.0",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--outdir", tempdir(),
    ])
    @test campaign_te_values(te_ladder_cfg) == [0.5, 1.0]
    te_cfgs = campaign_ladder_configs(te_ladder_cfg)
    @test [c["te"] for c in te_cfgs] == [0.5, 1.0]
    @test all(c -> c["Dmax"] == 96, te_cfgs)
    @test length(unique(output_path.(te_cfgs))) == 2
    @test occursin("_te0.5_", output_path(te_cfgs[1]))
    @test occursin("_te1_", output_path(te_cfgs[2]))
    @test campaign_te_values(parse_args(["--te", "0.25"])) == [0.25]

    g_ladder_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "10",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "20",
        "--Dmax", "64",
        "--te", "1.0",
        "--g-values", "0.05,0.1",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--outdir", tempdir(),
    ])
    @test campaign_g_values(g_ladder_cfg) == [0.05, 0.1]
    g_cfgs = campaign_ladder_configs(g_ladder_cfg)
    @test [c["g"] for c in g_cfgs] == [0.05, 0.1]
    @test all(c -> c["Dmax"] == 64, g_cfgs)
    @test length(unique(output_path.(g_cfgs))) == 2
    @test occursin("_g0.05_te1_", output_path(g_cfgs[1]))
    @test occursin("_g0.1_te1_", output_path(g_cfgs[2]))
    @test campaign_g_values(parse_args(["--g", "0.2"])) == [0.2]

    default_init_cfg = parse_args(["--outdir", tempdir()])
    @test default_init_cfg["init_state"] == "product"
    @test default_init_cfg["theta"] == 0.0
    default_init_command = join(command_args_for_config(default_init_cfg), " ")
    @test occursin("--init-state product", default_init_command)
    @test occursin("--theta 0.0", default_init_command)

    theta_init_cfg = parse_args([
        "--init-state", " Theta ",
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
    legacy_theta_init_cfg = parse_args([
        "--init_state", " Theta ",
        "--theta", "0.25",
    ])
    @test legacy_theta_init_cfg["init_state"] == "theta"
    @test legacy_theta_init_cfg["theta"] == 0.25
    legacy_theta_init_command = join(command_args_for_config(legacy_theta_init_cfg), " ")
    @test occursin("--init-state theta", legacy_theta_init_command)

    identity_init_cfg = parse_args([
        "--methods", "mpo",
        "--init-state", " Identity ",
    ])
    @test identity_init_cfg["init_state"] == "identity"
    @test occursin("_initidentity_", output_path(identity_init_cfg))
    @test !occursin("_initidentity_theta", output_path(identity_init_cfg))
    identity_init_command = join(command_args_for_config(identity_init_cfg), " ")
    @test occursin("--init-state identity", identity_init_command)
    ground_init_cfg = parse_args([
        "--methods", "mcwf",
        "--init-state", " Ground ",
        "--outdir", tempdir(),
    ])
    @test ground_init_cfg["init_state"] == "ground"
    @test occursin("_initground_", output_path(ground_init_cfg))
    @test !occursin("_initground_theta", output_path(ground_init_cfg))
    ground_init_command = join(command_args_for_config(ground_init_cfg), " ")
    @test occursin("--init-state ground", ground_init_command)
    @test_throws ErrorException parse_args(["--init-state", "bad"])
    @test_throws ErrorException parse_args([
        "--methods", "mcwf",
        "--init-state", " Identity ",
    ])

    random_schedule_cfg = parse_args([
        "--schedule", "random",
        "--outdir", tempdir(),
    ])
    round_robin_cfg = parse_args([
        "--schedule", "round_robin",
        "--outdir", tempdir(),
    ])
    descending_schedule_cfg = parse_args([
        "--schedule", "descending",
        "--outdir", tempdir(),
    ])
    @test output_path(random_schedule_cfg) != output_path(round_robin_cfg)
    @test output_path(descending_schedule_cfg) != output_path(round_robin_cfg)
    @test output_path(descending_schedule_cfg) != output_path(random_schedule_cfg)
    @test occursin("_schedrand_", output_path(random_schedule_cfg))
    @test occursin("_scheddesc_", output_path(descending_schedule_cfg))
    @test !occursin("_schedround_robin_", output_path(round_robin_cfg))
    descending_command = join(command_args_for_config(descending_schedule_cfg), " ")
    @test occursin("--schedule descending", descending_command)
    @test_throws ErrorException parse_args(["--schedule", "bad"])

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
    @test all(job -> job["te_values"] === nothing, parallel_jobs)
    @test all(job -> length(job["Ns"]) == 1, parallel_jobs)
    @test all(job -> length(job["R_values"]) == 1, parallel_jobs)
    @test length(unique(output_path.(parallel_jobs))) == length(parallel_jobs)
    @test length(unique(job["progress_csv"] for job in parallel_jobs)) == length(parallel_jobs)
    @test all(job -> occursin("Dmax$(job["Dmax"])", output_path(job)), parallel_jobs)
    @test all(job -> occursin("_R$(only(job["R_values"]))_", job["progress_csv"]), parallel_jobs)
    @test all(
        job -> occursin(
            splitext(default_output_filename(job))[1],
            basename(job["progress_csv"]),
        ),
        parallel_jobs,
    )

    single_progress_path = joinpath(tempdir(), "single_tdvp_progress.csv")
    single_job_plan_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "2",
        "--methods", "mcwf",
        "--Dmax", "96",
        "--outdir", tempdir(),
        "--progress-csv", single_progress_path,
        "--print-parallel-plan",
    ])
    single_job = only(parallel_plan_configs(single_job_plan_cfg))
    @test single_job["progress_csv"] == single_progress_path

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
    @test occursin("This driver does not launch jobs concurrently", parallel_plan_text)
    @test occursin("external process scheduler", parallel_plan_text)
    @test occursin("append the HDF5 protocol stem", parallel_plan_text)
    @test occursin("requested CSV stem", parallel_plan_text)
    single_plan_text = sprint(io -> print_parallel_plan(single_job_plan_cfg; io=io))
    @test occursin("kept unchanged", single_plan_text)
    @test shell_word("~/tdvp data") == "'~/tdvp data'"
    @test shell_word("~/tdvp_data") == "~/tdvp_data"

    te_plan_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "2",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "40",
        "--Dmax", "96",
        "--te-values", "0.5,1.0",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--outdir", tempdir(),
        "--progress-csv", joinpath(tempdir(), "te_progress.csv"),
        "--print-parallel-plan",
    ])
    te_jobs = parallel_plan_configs(te_plan_cfg)
    @test length(te_jobs) == 2
    @test [job["te"] for job in te_jobs] == [0.5, 1.0]
    @test all(job -> job["te_values"] === nothing, te_jobs)
    @test length(unique(output_path.(te_jobs))) == 2
    @test length(unique(job["progress_csv"] for job in te_jobs)) == 2
    @test occursin("_te0.5_", output_path(te_jobs[1]))
    @test occursin("_te1_", output_path(te_jobs[2]))
    te_commands = parallel_plan_commands(te_plan_cfg)
    @test count(command -> occursin("--te 0.5", command), te_commands) == 1
    @test count(command -> occursin("--te 1.0", command), te_commands) == 1
    @test !any(command -> occursin("--te-values", command), te_commands)
    te_plan_text = sprint(io -> print_parallel_plan(te_plan_cfg; io=io))
    @test occursin("fixed-protocol te ladder", te_plan_text)

    g_plan_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "10",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "20",
        "--Dmax", "64",
        "--te", "1.0",
        "--g-values", "0.05,0.1",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--outdir", tempdir(),
        "--progress-csv", joinpath(tempdir(), "g_progress.csv"),
        "--print-parallel-plan",
    ])
    g_jobs = parallel_plan_configs(g_plan_cfg)
    @test length(g_jobs) == 2
    @test [job["g"] for job in g_jobs] == [0.05, 0.1]
    @test all(job -> job["g_values"] === nothing, g_jobs)
    @test length(unique(output_path.(g_jobs))) == 2
    @test length(unique(job["progress_csv"] for job in g_jobs)) == 2
    @test occursin("_g0.05_te1_", output_path(g_jobs[1]))
    @test occursin("_g0.1_te1_", output_path(g_jobs[2]))
    g_commands = parallel_plan_commands(g_plan_cfg)
    @test count(command -> occursin("--g 0.05", command), g_commands) == 1
    @test count(command -> occursin("--g 0.1", command), g_commands) == 1
    @test !any(command -> occursin("--g-values", command), g_commands)
    g_plan_text = sprint(io -> print_parallel_plan(g_plan_cfg; io=io))
    @test occursin("fixed-protocol g ladder", g_plan_text)

    trajectory_plan_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "2",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "5",
        "--Dmax", "96",
        "--outdir", tempdir(),
        "--progress-csv", joinpath(tempdir(), "traj_progress.csv"),
        "--trajectory-values", "1,3",
        "--print-parallel-plan",
    ])
    @test campaign_trajectory_values(trajectory_plan_cfg) == [1, 3]
    trajectory_jobs = parallel_plan_configs(trajectory_plan_cfg)
    @test length(trajectory_jobs) == 2
    @test [job["trajectory_index"] for job in trajectory_jobs] == [1, 3]
    @test all(job -> job["trajectory_values"] === nothing, trajectory_jobs)
    @test all(job -> job["M_mcwf"] == 1, trajectory_jobs)
    @test length(unique(output_path.(trajectory_jobs))) == 2
    @test length(unique(job["progress_csv"] for job in trajectory_jobs)) == 2
    @test occursin("_traj1_steps", output_path(trajectory_jobs[1]))
    @test occursin("_traj3_steps", output_path(trajectory_jobs[2]))
    @test occursin("_traj1_steps", basename(trajectory_jobs[1]["progress_csv"]))
    @test occursin("_traj3_steps", basename(trajectory_jobs[2]["progress_csv"]))
    trajectory_commands = parallel_plan_commands(trajectory_plan_cfg)
    @test all(command -> occursin("--M-mcwf 1", command), trajectory_commands)
    @test count(command -> occursin("--trajectory-index 1", command), trajectory_commands) == 1
    @test count(command -> occursin("--trajectory-index 3", command), trajectory_commands) == 1
    @test !any(command -> occursin("--trajectory-values", command), trajectory_commands)
    trajectory_plan_text = sprint(io -> print_parallel_plan(trajectory_plan_cfg; io=io))
    @test occursin("Trajectory jobs use --M-mcwf 1", trajectory_plan_text)

    paired_evolution_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "2,5",
        "--methods", "mcwf",
        "--evolution-method-values", " TROTTER , Continuous ",
        "--steps", "4",
        "--Dmax", "128",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--outdir", tempdir(),
        "--progress-csv", joinpath(tempdir(), "evolution_progress.csv"),
        "--tdvp-outputlevel", "1",
        "--tdvp-sweep-progress",
        "--print-parallel-plan",
    ])
    @test campaign_evolution_method_values(paired_evolution_cfg) == ["trotter", "continuous"]
    paired_jobs = parallel_plan_configs(paired_evolution_cfg)
    @test length(paired_jobs) == 4
    @test count(job -> job["evolution_method"] == "trotter", paired_jobs) == 2
    @test count(job -> job["evolution_method"] == "continuous", paired_jobs) == 2
    @test all(job -> job["evolution_method_values"] === nothing, paired_jobs)
    @test length(unique(output_path.(paired_jobs))) == length(paired_jobs)
    @test length(unique(job["progress_csv"] for job in paired_jobs)) == length(paired_jobs)
    @test all(
        job -> job["tdvp_sweep_progress"] == (job["evolution_method"] == "continuous"),
        paired_jobs,
    )
    expected_tdvp_outputlevel = paired_evolution_cfg["tdvp_outputlevel"]
    @test all(
        job -> job["tdvp_outputlevel"] ==
               (job["evolution_method"] == "continuous" ? expected_tdvp_outputlevel : 0),
        paired_jobs,
    )
    paired_commands = parallel_plan_commands(paired_evolution_cfg)
    @test count(command -> occursin("--evolution-method trotter", command), paired_commands) == 2
    @test count(command -> occursin("--evolution-method continuous", command), paired_commands) == 2
    @test count(command -> occursin("--tdvp-sweep-progress", command), paired_commands) == 2
    @test count(command -> occursin("--tdvp-outputlevel 1", command), paired_commands) == 2
    @test all(
        command -> !occursin("--tdvp-sweep-progress", command),
        filter(command -> occursin("--evolution-method trotter", command), paired_commands),
    )
    @test all(
        command -> !occursin("--tdvp-outputlevel", command),
        filter(command -> occursin("--evolution-method trotter", command), paired_commands),
    )
    @test all(
        command -> occursin("--tdvp-sweep-progress", command),
        filter(command -> occursin("--evolution-method continuous", command), paired_commands),
    )
    @test all(
        command -> occursin("--tdvp-outputlevel 1", command),
        filter(command -> occursin("--evolution-method continuous", command), paired_commands),
    )
    @test !any(command -> occursin("--evolution-method-values", command), paired_commands)
    @test all(command -> occursin("--delta-min 0.5051167496264384", command), paired_commands)
    paired_plan_text = sprint(io -> print_parallel_plan(paired_evolution_cfg; io=io))
    @test occursin("Evolution-method jobs share the requested detuning interval", paired_plan_text)
    @test any(job -> occursin("_mcwf_trotter_steps", output_path(job)), paired_jobs)
    @test any(job -> occursin("_mcwf_continuous_steps", output_path(job)), paired_jobs)
    paired_no_progress_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "2",
        "--methods", "mcwf",
        "--evolution-method-values", "trotter,continuous",
        "--steps", "4",
        "--Dmax", "128",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--outdir", tempdir(),
        "--print-parallel-plan",
    ])
    paired_no_progress_text = sprint(io -> print_parallel_plan(paired_no_progress_cfg; io=io))
    @test occursin("No progress CSV path requested", paired_no_progress_text)
    @test !occursin("generated CSV paths", paired_no_progress_text)
    @test_throws ErrorException parse_args([
        "--evolution-method-values", "trotter,continuous",
    ])
    @test_throws ErrorException parse_args([
        "--evolution-method-values", "trotter",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--tdvp-sweep-progress",
        "--print-parallel-plan",
    ])
    @test_throws ErrorException parse_args([
        "--evolution-method", "trotter",
        "--tdvp-outputlevel", "1",
    ])
    @test_throws ErrorException parse_args([
        "--evolution-method", "continuous",
        "--tdvp-outputlevel", "-1",
    ])
    @test_throws ErrorException parse_args([
        "--evolution-method-values", "trotter",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--tdvp-outputlevel", "1",
        "--print-parallel-plan",
    ])
    @test_throws ErrorException parse_args([
        "--evolution-method-values", "trotter,continuous",
        "--print-parallel-plan",
    ])
    @test_throws ErrorException parse_args([
        "--methods", "mpo",
        "--evolution-method-values", "trotter,continuous",
        "--delta-min", "0.5",
        "--delta-max", "3.0",
        "--print-parallel-plan",
    ])
    @test_throws ErrorException parse_args([
        "--evolution-method-values", "trotter,trotter",
        "--delta-min", "0.5",
        "--delta-max", "3.0",
        "--print-parallel-plan",
    ])
    @test_throws ErrorException parse_args([
        "--evolution-method-values", "trotter,bad",
        "--delta-min", "0.5",
        "--delta-max", "3.0",
        "--print-parallel-plan",
    ])

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
    @test mode_cfg["mode_measurement_stride"] == 1
    mode_ham = campaign_hamiltonian_parameters(64, mode_cfg)
    @test mode_ham.model isa CoolingTNS.IsingModel
    @test mode_ham.bc == :periodic
    @test mode_ham.params.h == -0.75
    @test CoolingTNS.supports_ising_fourier_observables(mode_ham)
    mode_reference = campaign_base_detuning_reference(mode_ham, mode_cfg)
    @test mode_reference.source == LARGE_N_DETUNING_REFERENCE_ISING_MODE_PAIR
    @test mode_reference.delta ≈ CoolingTNS.ising_mode_detuning_reference(mode_ham)
    @test mode_reference.delta > 0
    @test CoolingTNS.ising_mode_detuning_preserves_px("XX")
    @test !CoolingTNS.ising_mode_detuning_preserves_px("XY")
    @test !CoolingTNS.ising_mode_detuning_preserves_px("ZZ")
    @test !CoolingTNS.ising_mode_detuning_has_special_modes(mode_ham)
    antiperiodic_mode_cfg = parse_args([
        "--model", "ising",
        "--bc", "antiperiodic",
        "--Ns", "64",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--measure-modes",
        "--outdir", tempdir(),
    ])
    antiperiodic_mode_ham = campaign_hamiltonian_parameters(64, antiperiodic_mode_cfg)
    @test CoolingTNS.ising_mode_detuning_has_special_modes(antiperiodic_mode_ham)
    @test_throws ErrorException campaign_base_detuning_reference(
        antiperiodic_mode_ham,
        antiperiodic_mode_cfg,
    )
    explicit_antiperiodic_mode_cfg = parse_args([
        "--model", "ising",
        "--bc", "antiperiodic",
        "--Ns", "64",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--delta-min", "0.5",
        "--delta-max", "1.0",
        "--measure-modes",
        "--outdir", tempdir(),
    ])
    explicit_antiperiodic_ref = campaign_base_detuning_reference(
        antiperiodic_mode_ham,
        explicit_antiperiodic_mode_cfg,
    )
    @test explicit_antiperiodic_ref.source == LARGE_N_DETUNING_REFERENCE_SETUP_GAP
    @test isnothing(explicit_antiperiodic_ref.delta)
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
    @test explicit_nonpreserving_ref.source == LARGE_N_DETUNING_REFERENCE_SETUP_GAP
    @test isnothing(explicit_nonpreserving_ref.delta)
    generic_cfg = parse_args(["--outdir", tempdir()])
    generic_reference = campaign_base_detuning_reference(
        campaign_hamiltonian_parameters(64, generic_cfg),
        generic_cfg,
    )
    @test generic_reference.source == LARGE_N_DETUNING_REFERENCE_SETUP_GAP
    @test isnothing(generic_reference.delta)
    @test occursin("ising_bcperiodic", output_path(mode_cfg))
    no_mode_counterpart_cfg = parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--Ns", "64",
        "--R-values", "1,2",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "5",
        "--Dmax", "32",
        "--h", "-0.75",
        "--outdir", tempdir(),
    ])
    @test output_path(mode_cfg) != output_path(no_mode_counterpart_cfg)
    @test occursin("_modes_steps", output_path(mode_cfg))
    @test !occursin("_modes", output_path(no_mode_counterpart_cfg))
    mode_command = join(command_args_for_config(mode_cfg), " ")
    @test occursin("--model ising", mode_command)
    @test occursin("--bc periodic", mode_command)
    @test occursin("--h -0.75", mode_command)
    @test !occursin("--hx", mode_command)
    @test !occursin("--hz", mode_command)
    @test occursin("--measure-modes", mode_command)
    @test !occursin("--mode-measurement-stride", mode_command)

    stride_mode_cfg = parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--Ns", "64",
        "--R-values", "1",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "10",
        "--Dmax", "32",
        "--h", "-0.75",
        "--measure-modes",
        "--mode-measurement-stride", "5",
        "--outdir", tempdir(),
    ])
    @test stride_mode_cfg["mode_measurement_stride"] == 5
    @test occursin("_modestride5", output_path(stride_mode_cfg))
    @test output_path(stride_mode_cfg) != output_path(mode_cfg)
    stride_mode_command = join(command_args_for_config(stride_mode_cfg), " ")
    @test occursin("--mode-measurement-stride 5", stride_mode_command)
    mode_progress_base = joinpath(tempdir(), "tdvp_progress.csv")
    mode_plan_cfg = parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--Ns", "64",
        "--R-values", "1,2",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "10",
        "--Dmax", "32",
        "--h", "-0.75",
        "--measure-modes",
        "--outdir", tempdir(),
        "--progress-csv", mode_progress_base,
        "--print-parallel-plan",
    ])
    mode_no_progress_plan_cfg = parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--Ns", "64",
        "--R-values", "1,2",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "10",
        "--Dmax", "32",
        "--h", "-0.75",
        "--measure-modes",
        "--outdir", tempdir(),
        "--print-parallel-plan",
    ])
    stride_plan_cfg = parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--Ns", "64",
        "--R-values", "1,2",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--steps", "10",
        "--Dmax", "32",
        "--h", "-0.75",
        "--measure-modes",
        "--mode-measurement-stride", "5",
        "--outdir", tempdir(),
        "--progress-csv", mode_progress_base,
        "--print-parallel-plan",
    ])
    @test mode_progress_csv_recommended(mode_no_progress_plan_cfg)
    @test !mode_progress_csv_recommended(mode_plan_cfg)
    @test_logs (:warn, r"--measure-modes with continuous TDVP has no --progress-csv path") begin
        @test warn_if_mode_progress_csv_recommended(mode_no_progress_plan_cfg)
    end
    @test !warn_if_mode_progress_csv_recommended(mode_plan_cfg)
    no_progress_plan_text = sprint() do io
        print_parallel_plan(mode_no_progress_plan_cfg; io=io)
    end
    @test occursin("Mode-resolved continuous TDVP runs", no_progress_plan_text)
    @test occursin("pass --progress-csv", no_progress_plan_text)
    progress_plan_text = sprint() do io
        print_parallel_plan(mode_plan_cfg; io=io)
    end
    @test !occursin("Mode-resolved continuous TDVP runs", progress_plan_text)
    mode_job = first(parallel_plan_configs(mode_plan_cfg))
    stride_job = first(parallel_plan_configs(stride_plan_cfg))
    @test mode_job["progress_csv"] != stride_job["progress_csv"]
    @test occursin("_modes_steps", basename(mode_job["progress_csv"]))
    @test occursin("_modestride5_steps", basename(stride_job["progress_csv"]))
    @test occursin(
        splitext(default_output_filename(mode_job))[1],
        basename(mode_job["progress_csv"]),
    )
    @test occursin(
        splitext(default_output_filename(stride_job))[1],
        basename(stride_job["progress_csv"]),
    )

    protocol_progress_base = joinpath(tempdir(), "tdvp_progress.csv")
    default_protocol_plan_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "2,5",
        "--methods", "mcwf",
        "--steps", "5",
        "--Dmax", "96",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--outdir", tempdir(),
        "--progress-csv", protocol_progress_base,
        "--print-parallel-plan",
    ])
    theta_desc_plan_cfg = parse_args([
        "--Ns", "64",
        "--R-values", "2,5",
        "--methods", "mcwf",
        "--steps", "5",
        "--Dmax", "96",
        "--delta-min", "0.5051167496264384",
        "--delta-max", "3.0307004977586303",
        "--schedule", "descending",
        "--init-state", "theta",
        "--theta", "0.25",
        "--outdir", tempdir(),
        "--progress-csv", protocol_progress_base,
        "--print-parallel-plan",
    ])
    default_protocol_job = first(parallel_plan_configs(default_protocol_plan_cfg))
    theta_desc_job = first(parallel_plan_configs(theta_desc_plan_cfg))
    @test default_protocol_job["progress_csv"] != theta_desc_job["progress_csv"]
    @test occursin("_scheddesc_", basename(theta_desc_job["progress_csv"]))
    @test occursin("_inittheta_theta0.25_", basename(theta_desc_job["progress_csv"]))
    @test occursin(
        splitext(default_output_filename(theta_desc_job))[1],
        basename(theta_desc_job["progress_csv"]),
    )
    @test_throws ErrorException parse_args(["--mode-measurement-stride", "2"])
    @test_throws ErrorException parse_args([
        "--model", "ising",
        "--bc", "periodic",
        "--methods", "mcwf",
        "--evolution-method", "continuous",
        "--measure-modes",
        "--mode-measurement-stride", "0",
    ])

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

    normalized_continuous_cfg = parse_args([
        "--methods", "mcwf",
        "--evolution-method", " Continuous ",
        "--outdir", tempdir(),
    ])
    @test normalized_continuous_cfg["evolution_method"] == "continuous"
    @test sim_params_for("mcwf", normalized_continuous_cfg).evolution_method isa
          CoolingTNS.ContinuousEvolution
    @test occursin(
        "--evolution-method continuous",
        join(command_args_for_config(normalized_continuous_cfg), " "),
    )
    invalid_helper_cfg = copy(normalized_continuous_cfg)
    invalid_helper_cfg["evolution_method"] = "invalid"
    @test_throws ErrorException sim_params_for("mcwf", invalid_helper_cfg)

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
    @test_throws ErrorException parse_args(["--te-values", "0.5,1.0"])
    @test_throws ErrorException parse_args([
        "--te-values", "0.5,1.0",
        "--output", joinpath(tempdir(), "single_te.h5"),
        "--delta-min", "0.5",
        "--delta-max", "1.0",
    ])
    @test_throws ErrorException parse_args(["--te-values", "1.0,1.0"])
    @test_throws ErrorException parse_args(["--te-values", "1.0,-0.5"])
    @test_throws ErrorException parse_args(["--g-values", "0.05,0.1"])
    @test_throws ErrorException parse_args([
        "--g-values", "0.05,0.1",
        "--output", joinpath(tempdir(), "single_g.h5"),
        "--delta-min", "0.5",
        "--delta-max", "1.0",
    ])
    @test_throws ErrorException parse_args([
        "--g-values", "0.1,0.1",
        "--delta-min", "0.5",
        "--delta-max", "1.0",
    ])
    @test_throws ErrorException parse_args([
        "--g-values", "0.1,NaN",
        "--delta-min", "0.5",
        "--delta-max", "1.0",
    ])
    @test_throws ErrorException parse_args([
        "--g-values", "0.12345678901234,0.12345678901235",
        "--delta-min", "0.5",
        "--delta-max", "1.0",
    ])
    @test_throws ErrorException parse_args(["--Ns", "64,64"])
    @test_throws ErrorException parse_args(["--R-values", "2,2"])
    @test_throws ErrorException parse_args(["--methods", "mcwf,mcwf"])
    @test_throws ErrorException parse_args(["--Dmax-values", "160,320"])
    @test_throws ErrorException parse_args(["--plan-julia-threads", "0"])
    @test_throws ErrorException parse_args(["--plan-blas-threads", "0"])
    @test_throws ErrorException parse_args(["--plan-julia-threads", "1"])
    @test_throws ErrorException parse_args(["--plan-blas-threads", "1"])
    @test_throws ErrorException parse_args(["--trajectory-index", "0"])
    @test_throws ErrorException parse_args(["--trajectory-index", "10000"])
    @test_throws ErrorException parse_args(["--trajectory-values", ""])
    @test_throws ErrorException parse_args([
        "--methods", "mpo",
        "--trajectory-index", "2",
    ])
    @test_throws ErrorException parse_args([
        "--methods", "mcwf",
        "--M-mcwf", "2",
        "--trajectory-index", "2",
    ])
    @test_throws ErrorException parse_args(["--trajectory-values", "1,2"])
    @test_throws ErrorException parse_args([
        "--methods", "mpo",
        "--trajectory-values", "1,2",
        "--print-parallel-plan",
    ])
    @test_throws ErrorException parse_args([
        "--methods", "mcwf",
        "--trajectory-values", "1,1",
        "--print-parallel-plan",
    ])
    @test_throws ErrorException parse_args([
        "--methods", "mcwf",
        "--M-mcwf", "2",
        "--trajectory-values", "1,2",
        "--print-parallel-plan",
    ])
    @test_throws ErrorException parse_args([
        "--methods", "mcwf",
        "--trajectory-index", "1",
        "--trajectory-values", "2",
        "--print-parallel-plan",
    ])
    @test_throws ErrorException parse_args([
        "--methods", "mcwf",
        "--trajectory-values", "1,2",
        "--trajectory-index", "3",
        "--print-parallel-plan",
    ])
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
                LARGE_N_ROW_SYSTEM_MAX_BOND_KEY => [1, 4, 8],
                LARGE_N_ROW_SYSTEM_MEAN_BOND_KEY => [1.0, 3.0, 6.0],
                LARGE_N_ROW_EVOLVED_MAX_BOND_KEY => [0, 7, 9],
                LARGE_N_ROW_EVOLVED_MEAN_BOND_KEY => [NaN, 5.0, 7.0],
                LARGE_N_ROW_TDVP_SWEEP_MAX_BOND_KEY => [0, 6, 10],
                CoolingTNS.RESULT_DELTA_LIST => [NaN, 0.5, 3.0],
                CoolingTNS.RESULT_TE_LIST => [NaN, 1.0, 1.25],
                LARGE_N_FINAL_BOND_DIMS_GROUP => [4, 8],
                "elapsed" => 1.25,
                "seed" => largeN_trajectory_seed(20260617, 64, 2, 1),
                "trajectory" => 1,
                CoolingTNS.RESULT_REQUESTED_STEPS => 2,
                CoolingTNS.RESULT_COMPLETED_STEPS => 2,
                "stop_reason" => "",
            ),
        ]

        h5open(path, "w") do f
            write_run_group(f, "R2", traj_rows, -2.0, 8, protocol, [0.5, 3.0])
            traj_rows_noncommon_te = [
                traj_rows[1],
                merge(copy(traj_rows[1]), Dict{String,Any}(
                    CoolingTNS.RESULT_TE_LIST => [NaN, 0.25, 1.75],
                    "elapsed" => 1.5,
                    "seed" => largeN_trajectory_seed(20260617, 64, 2, 2),
                    "trajectory" => 2,
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
            @test read(g[CoolingTNS.RESULT_DELTA_VALUES]) == [0.5, 3.0]
            @test read(g[CoolingTNS.RESULT_ENERGY]) == [-2.0, -1.5, -1.0]
            @test read(g[CoolingTNS.RESULT_RELATIVE_ENERGY]) == [0.0, 0.25, 0.5]
            @test vec(read(g[CoolingTNS.RESULT_ENERGY_TRAJECTORIES])) == [-2.0, -1.5, -1.0]
            @test read(g[CoolingTNS.RESULT_GROUND_STATE_OVERLAP]) == [0.1, 0.2, 0.3]
            @test vec(read(g[CoolingTNS.RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES])) ==
                  [0.1, 0.2, 0.3]
            @test !haskey(g, "E_mean")
            @test !haskey(g, "GS_overlap_mean")
            @test !haskey(g, "GS_overlap_trajectories")
            @test read(g[LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY]) ==
                  LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE
            @test read(g[LARGE_N_DETUNING_REFERENCE_GAP_KEY]) == 0.75
            @test read(g[LARGE_N_DETUNING_DELTA_MIN_KEY]) == 0.5
            @test read(g[LARGE_N_DETUNING_DELTA_MAX_KEY]) == 3.0
            @test isnan(read(g[LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY]))
            @test read(g[LARGE_N_DETUNING_FIXED_ACROSS_DMAX_KEY]) == true
            @test read(g[CoolingTNS.RESULT_REQUESTED_STEPS]) == [2]
            @test read(g[CoolingTNS.RESULT_COMPLETED_STEPS]) == [2]
            @test read(g[LARGE_N_STOP_REASONS_KEY]) == [""]
            @test read(g[LARGE_N_TRAJECTORY_SEEDS_KEY]) ==
                  [largeN_trajectory_seed(20260617, 64, 2, 1)]
            @test read(g[LARGE_N_TRAJECTORY_INDICES_KEY]) == [1]
            @test vec(read(g[LARGE_N_TDVP_SWEEP_MAX_BOND_KEY])) == [0, 6, 10]
            @test read(g[LARGE_N_TDVP_SWEEP_SATURATION_CYCLE_KEY]) == [2]
            @test read(g[CoolingTNS.RESULT_TRUNCATION_ERROR_HISTORY_STATUS]) ==
                  CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED
            @test read(g[LARGE_N_TE_LIST_IS_COMMON_KEY]) == true
            @test isequal(read(g[CoolingTNS.RESULT_TE_LIST]), [NaN, 1.0, 1.25])
            @test isequal(vec(read(g[LARGE_N_TE_LISTS_KEY])), [NaN, 1.0, 1.25])

            g_noncommon_te = f["R2_noncommon_te"]
            @test read(g_noncommon_te[LARGE_N_TE_LIST_IS_COMMON_KEY]) == false
            @test !haskey(g_noncommon_te, CoolingTNS.RESULT_TE_LIST)
            @test read(g_noncommon_te[LARGE_N_TRAJECTORY_SEEDS_KEY]) ==
                  [largeN_trajectory_seed(20260617, 64, 2, 1),
                   largeN_trajectory_seed(20260617, 64, 2, 2)]
            @test read(g_noncommon_te[LARGE_N_TRAJECTORY_INDICES_KEY]) == [1, 2]
            @test isequal(read(g_noncommon_te[LARGE_N_TE_LIST_FIRST_TRAJECTORY_KEY]), [NaN, 1.0, 1.25])
            @test isequal(read(g_noncommon_te[LARGE_N_TE_LISTS_KEY])[:, 2], [NaN, 0.25, 1.75])

            g_gap = f["R3_gap"]
            @test read(g_gap[CoolingTNS.RESULT_DELTA_VALUES]) == [0.75, 1.875, 3.0]
            @test read(g_gap[LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY]) ==
                  LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE
            @test read(g_gap[LARGE_N_DETUNING_REFERENCE_GAP_KEY]) == 0.75
            @test read(g_gap[LARGE_N_DETUNING_DELTA_MIN_KEY]) == 0.75
            @test read(g_gap[LARGE_N_DETUNING_DELTA_MAX_KEY]) == 3.0
            @test read(g_gap[LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY]) == 4.0
            @test read(g_gap[LARGE_N_DETUNING_FIXED_ACROSS_DMAX_KEY]) == false
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
            LARGE_N_ROW_SYSTEM_MAX_BOND_KEY => [1, 4, 8],
            LARGE_N_ROW_SYSTEM_MEAN_BOND_KEY => [1.0, 3.0, 6.0],
            LARGE_N_ROW_EVOLVED_MAX_BOND_KEY => [0, 7, 9],
            LARGE_N_ROW_EVOLVED_MEAN_BOND_KEY => [NaN, 5.0, 7.0],
            LARGE_N_ROW_TDVP_SWEEP_MAX_BOND_KEY => [0, 6, 10],
            CoolingTNS.RESULT_DELTA_LIST => [NaN, 0.5, 1.5],
            CoolingTNS.RESULT_TE_LIST => [NaN, 1.0, 1.0],
            LARGE_N_FINAL_BOND_DIMS_GROUP => [4, 8],
            "elapsed" => 1.25,
            "seed" => 101,
            "trajectory" => 1,
            CoolingTNS.RESULT_REQUESTED_STEPS => 2,
            CoolingTNS.RESULT_COMPLETED_STEPS => 2,
            "stop_reason" => "",
            CoolingTNS.RESULT_MODE_K_INDICES => [1//2, 3//2],
            CoolingTNS.RESULT_MODE_ENERGIES => [0.4, 0.8],
            CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES => [0, 1, 2],
            CoolingTNS.RESULT_MODE_GF => -1,
            CoolingTNS.RESULT_MODE_GF_SOURCE => CoolingTNS.FERMIONIC_GRID_SOURCE_STATE,
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
                "seed" => 102,
                "trajectory" => 2,
            )),
        ]

        h5open(mode_path, "w") do f
            write_run_group(f, "R_modes", traj_rows, -2.0, 8, protocol, [0.5, 1.5])

            sparse_hk_1 = [-1.0 0.0; NaN NaN; 0.0 1.0]
            sparse_hk_2 = [-0.8 0.2; NaN NaN; 0.2 0.8]
            sparse_nk_1 = CoolingTNS.mode_occupation_from_hk(sparse_hk_1)
            sparse_nk_2 = CoolingTNS.mode_occupation_from_hk(sparse_hk_2)
            sparse_base = merge(copy(base_row), Dict{String,Any}(
                CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES => [0, 2],
            ))
            sparse_row_1 = merge(copy(sparse_base), Dict{String,Any}(
                CoolingTNS.RESULT_MODE_HK => sparse_hk_1,
                CoolingTNS.RESULT_MODE_NK => sparse_nk_1,
            ))
            sparse_row_2 = merge(copy(sparse_base), Dict{String,Any}(
                "E" => [-2.0, -1.75, -1.25],
                CoolingTNS.RESULT_MODE_HK => sparse_hk_2,
                CoolingTNS.RESULT_MODE_NK => sparse_nk_2,
                "elapsed" => 1.5,
                "seed" => 102,
                "trajectory" => 2,
            ))
            write_run_group(
                f,
                "R_sparse_single",
                [sparse_row_1],
                -2.0,
                8,
                protocol,
                [0.5, 1.5],
            )
            write_run_group(
                f,
                "R_sparse_ensemble",
                [sparse_row_1, sparse_row_2],
                -2.0,
                8,
                protocol,
                [0.5, 1.5],
            )
        end
        h5open(mode_path, "r") do f
            g = f["R_modes"]
            @test read(g[CoolingTNS.RESULT_MODE_HK]) ≈ (mode_hk_1 .+ mode_hk_2) ./ 2
            @test read(g[CoolingTNS.RESULT_MODE_NK]) ≈ (mode_nk_1 .+ mode_nk_2) ./ 2
            @test read(g[CoolingTNS.RESULT_MODE_K_INDICES]) == Float64.([1//2, 3//2])
            @test read(g[CoolingTNS.RESULT_MODE_ENERGIES]) == [0.4, 0.8]
            @test read(g[CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES]) == [0, 1, 2]
            @test read(g[CoolingTNS.RESULT_MODE_GF]) == -1
            @test read(g[CoolingTNS.RESULT_MODE_GF_SOURCE]) ==
                  CoolingTNS.FERMIONIC_GRID_SOURCE_STATE
            @test read(g[LARGE_N_TRAJECTORY_SEEDS_KEY]) == [101, 102]
            @test read(g[LARGE_N_TRAJECTORY_INDICES_KEY]) == [1, 2]
            @test size(read(g[CoolingTNS.RESULT_MODE_HK_TRAJECTORIES])) == (3, 2, 2)
            @test read(g[CoolingTNS.RESULT_MODE_HK_TRAJECTORIES])[:, :, 1] ≈ mode_hk_1
            @test read(g[CoolingTNS.RESULT_MODE_HK_TRAJECTORIES])[:, :, 2] ≈ mode_hk_2
            @test size(read(g[CoolingTNS.RESULT_MODE_NK_TRAJECTORIES])) == (3, 2, 2)
            @test read(g[CoolingTNS.RESULT_MODE_NK_TRAJECTORIES])[:, :, 1] ≈ mode_nk_1
            @test read(g[CoolingTNS.RESULT_MODE_NK_TRAJECTORIES])[:, :, 2] ≈ mode_nk_2
            @test size(read(g[CoolingTNS.RESULT_MODE_NK_STDERR])) == (3, 2)

            g_single = f["R_sparse_single"]
            single_hk = read(g_single[CoolingTNS.RESULT_MODE_HK])
            single_nk = read(g_single[CoolingTNS.RESULT_MODE_NK])
            single_hk_stderr = read(g_single[CoolingTNS.RESULT_MODE_HK_STDERR])
            single_nk_stderr = read(g_single[CoolingTNS.RESULT_MODE_NK_STDERR])
            @test read(g_single[CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES]) == [0, 2]
            @test read(g_single[LARGE_N_TRAJECTORY_SEEDS_KEY]) == [101]
            @test read(g_single[LARGE_N_TRAJECTORY_INDICES_KEY]) == [1]
            @test all(isnan, single_hk[2, :])
            @test all(isnan, single_nk[2, :])
            @test all(isnan, single_hk_stderr[2, :])
            @test all(isnan, single_nk_stderr[2, :])
            @test single_hk_stderr[[1, 3], :] == zeros(2, 2)
            @test single_nk_stderr[[1, 3], :] == zeros(2, 2)

            g_ensemble = f["R_sparse_ensemble"]
            ensemble_hk = read(g_ensemble[CoolingTNS.RESULT_MODE_HK])
            ensemble_nk = read(g_ensemble[CoolingTNS.RESULT_MODE_NK])
            ensemble_hk_stderr = read(g_ensemble[CoolingTNS.RESULT_MODE_HK_STDERR])
            ensemble_nk_stderr = read(g_ensemble[CoolingTNS.RESULT_MODE_NK_STDERR])
            @test read(g_ensemble[CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES]) == [0, 2]
            @test read(g_ensemble[LARGE_N_TRAJECTORY_SEEDS_KEY]) == [101, 102]
            @test read(g_ensemble[LARGE_N_TRAJECTORY_INDICES_KEY]) == [1, 2]
            @test all(isnan, ensemble_hk[2, :])
            @test all(isnan, ensemble_nk[2, :])
            @test all(isnan, ensemble_hk_stderr[2, :])
            @test all(isnan, ensemble_nk_stderr[2, :])
            @test all(isfinite, ensemble_hk_stderr[[1, 3], :])
            @test all(isfinite, ensemble_nk_stderr[[1, 3], :])
        end
    finally
        rm(mode_path; force=true)
    end
end

@testset "Large-N campaign mode result validation" begin
    function captured_error(f)
        try
            f()
            return nothing
        catch err
            return err
        end
    end

    mode_hk = [-1.0 0.0; 0.0 1.0]
    mode_nk = CoolingTNS.mode_occupation_from_hk(mode_hk)
    energy = [-2.0, -1.5]
    result = Dict{String,Any}(
        CoolingTNS.RESULT_MODE_HK => mode_hk,
        CoolingTNS.RESULT_MODE_NK => mode_nk,
        CoolingTNS.RESULT_MODE_K_INDICES => [1//2, 3//2],
        CoolingTNS.RESULT_MODE_ENERGIES => [0.4, 0.8],
        CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES => [0, 1],
        CoolingTNS.RESULT_MODE_GF => -1,
        CoolingTNS.RESULT_MODE_GF_SOURCE => CoolingTNS.FERMIONIC_GRID_SOURCE_STATE,
    )

    @test validate_mode_measurement_result(result, energy).rows == [1, 2]

    missing_result = copy(result)
    delete!(missing_result, CoolingTNS.RESULT_MODE_NK)
    err = captured_error(() -> validate_mode_measurement_result(missing_result, energy))
    @test err isa ErrorException
    @test occursin("complete Ising Fourier-mode measurement set", sprint(showerror, err))

    nothing_result = copy(result)
    nothing_result[CoolingTNS.RESULT_MODE_HK] = nothing
    err = captured_error(() -> validate_mode_measurement_result(nothing_result, energy))
    @test err isa ErrorException
    @test occursin("complete Ising Fourier-mode measurement set", sprint(showerror, err))

    bad_hk_result = copy(result)
    bad_hk_result[CoolingTNS.RESULT_MODE_HK] = vec(mode_hk)
    err = captured_error(() -> validate_mode_measurement_result(bad_hk_result, energy))
    @test err isa ArgumentError
    @test occursin(CoolingTNS.RESULT_MODE_HK, sprint(showerror, err))
    @test occursin("steps-by-modes matrix", sprint(showerror, err))

    bad_nk_result = copy(result)
    bad_nk_result[CoolingTNS.RESULT_MODE_NK] = vec(mode_nk)
    err = captured_error(() -> validate_mode_measurement_result(bad_nk_result, energy))
    @test err isa DimensionMismatch
    @test occursin(CoolingTNS.RESULT_MODE_NK, sprint(showerror, err))
    @test occursin(CoolingTNS.RESULT_MODE_HK, sprint(showerror, err))
    @test occursin("shape", sprint(showerror, err))

    err = captured_error(() -> validate_mode_measurement_result(result, [-2.0]))
    @test err isa DimensionMismatch
    @test occursin(CoolingTNS.RESULT_ENERGY, sprint(showerror, err))
    @test occursin(CoolingTNS.RESULT_MODE_HK, sprint(showerror, err))
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
        g=0.05,
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
    @test row["stage"] == LARGE_N_PROGRESS_STAGE_UPDATED
    @test row["cycle"] == 1
    @test row["g"] == 0.05
    @test row["energy_per_site"] == -0.25
    @test row["relative_energy"] == relative_energy(-1.0, -4.0)
    @test row[LARGE_N_SYSTEM_MAX_BOND_KEY] == 8
    @test row[LARGE_N_EVOLVED_MAX_BOND_KEY] == 13

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
    @test initial_row["stage"] == LARGE_N_PROGRESS_STAGE_INITIAL
    @test initial_row["energy_per_site"] == -1.0
    @test isnan(initial_row["delta"])
    @test isnan(initial_row[LARGE_N_EVOLVED_MAX_BOND_KEY])

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
    @test prepared_row["stage"] == LARGE_N_PROGRESS_STAGE_PREPARED
    @test isnan(prepared_row["energy_per_site"])
    @test isnan(prepared_row["relative_energy"])
    @test isnan(prepared_row["overlap"])
    @test prepared_row[LARGE_N_SYSTEM_MAX_BOND_KEY] == 8
    @test prepared_row[LARGE_N_EVOLVED_MAX_BOND_KEY] == 11

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
    @test evolved_row["stage"] == LARGE_N_PROGRESS_STAGE_EVOLVED
    @test isnan(evolved_row["energy_per_site"])
    @test isnan(evolved_row["relative_energy"])
    @test isnan(evolved_row["overlap"])
    @test evolved_row[LARGE_N_SYSTEM_MAX_BOND_KEY] == 8
    @test evolved_row[LARGE_N_EVOLVED_MAX_BOND_KEY] == 13
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
    @test sweep_row["stage"] == LARGE_N_PROGRESS_STAGE_TDVP_SWEEP
    @test sweep_row["cycle"] == 1
    @test isnan(sweep_row["energy_per_site"])
    @test sweep_row[LARGE_N_SYSTEM_MAX_BOND_KEY] == 8
    @test sweep_row[LARGE_N_EVOLVED_MAX_BOND_KEY] == 21
    @test sweep_row["tdvp_sweep"] == 3
    @test sweep_row["tdvp_time"] == 0.6

    path = tempname() * ".csv"
    try
        append_progress_csv_row(path, row)
        append_progress_csv_row(path, merge(row, Dict{String,Any}(
            "timestamp" => "contains,comma",
            "stage" => LARGE_N_PROGRESS_STAGE_EVOLVED,
        )))
        lines = readlines(path)
        @test lines[1] == join(LARGE_N_PROGRESS_CSV_COLUMNS, ",")
        @test length(lines) == 3
        @test occursin("\"contains,comma\"", lines[3])
        @test count(==(','), lines[1]) == length(LARGE_N_PROGRESS_CSV_COLUMNS) - 1
    finally
        rm(path; force=true)
    end

    bad_stage_append_path = tempname() * ".csv"
    try
        @test_throws ArgumentError append_progress_csv_row(
            bad_stage_append_path,
            merge(row, Dict{String,Any}("stage" => "renormalized")),
        )
    finally
        rm(bad_stage_append_path; force=true)
    end

    stale_path = tempname() * ".csv"
    try
        write(stale_path, "timestamp,N\nold,4\n")
        @test_throws ArgumentError append_progress_csv_row(stale_path, row)
    finally
        rm(stale_path; force=true)
    end

    bad_info = merge(info, (stage=:renormalized,))
    @test_throws ArgumentError progress_row(
        context,
        bad_info,
        ham_params,
        -4.0,
        (max=8, mean=6.5),
        (max=13, mean=9.25),
        3.5,
    )
    @test_throws ArgumentError progress_base_row(
        context,
        ham_params;
        stage="renormalized",
        step=2,
        cycle=1,
        delta=0.5,
        te=2.0,
        energy_per_site=NaN,
        relative_energy_value=NaN,
        overlap=NaN,
        sys_bs=(max=8, mean=6.5),
        evolved_bs=(max=13, mean=9.25),
        elapsed=3.5,
    )
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
            "--trajectory-index", "3",
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
            @test read(f["mode_measurement_stride"]) == 1
            @test read(f[CoolingTNS.RESULT_INIT_STATE]) == "theta"
            @test read(f[CoolingTNS.RESULT_INIT_THETA]) == 0.0
            @test read(f[CoolingTNS.RESULT_RANDOMIZE_TIMES]) == false
            @test read(f[LARGE_N_TRAJECTORY_SEED_RULE_KEY]) == LARGE_N_TRAJECTORY_SEED_RULE
            @test read(f["h"]) == 0.5
            @test isnan(read(f["hx"]))
            @test isnan(read(f["hz"]))

            gm = f["N2/mcwf"]
            reference = CoolingTNS.ising_mode_detuning_reference(
                CoolingTNS.IsingParameters(2, 1.0, 0.5, :periodic)
            )
            @test read(gm["gap"]) ≈ reference
            @test read(gm[LARGE_N_DETUNING_REFERENCE_GAP_SOURCE_KEY]) ==
                  LARGE_N_DETUNING_REFERENCE_ISING_MODE_PAIR
            @test read(gm[LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY]) ==
                  LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE
            @test read(gm[LARGE_N_DETUNING_REFERENCE_GAP_KEY]) ≈ reference
            @test read(gm[LARGE_N_DETUNING_DELTA_MIN_KEY]) ≈ reference
            @test read(gm[LARGE_N_DETUNING_DELTA_MAX_KEY]) ≈ 6.0 * reference
            @test read(gm[LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY]) == 6.0
            @test read(gm[LARGE_N_DETUNING_FIXED_ACROSS_DMAX_KEY]) == false

            g = f["N2/mcwf/R1"]
            mode_hk = read(g[CoolingTNS.RESULT_MODE_HK])
            mode_nk = read(g[CoolingTNS.RESULT_MODE_NK])
            @test size(mode_hk) == (2, 2)
            @test size(mode_nk) == (2, 2)
            @test mode_nk ≈ CoolingTNS.mode_occupation_from_hk(mode_hk)
            @test all(isfinite, mode_hk)
            @test all(n -> -1e-12 <= n <= 1 + 1e-12, mode_nk)
            @test read(g[CoolingTNS.RESULT_MODE_GF]) == -1
            @test read(g[CoolingTNS.RESULT_MODE_GF_SOURCE]) ==
                  CoolingTNS.FERMIONIC_GRID_SOURCE_STATE
            @test read(g[LARGE_N_TRAJECTORY_SEEDS_KEY]) ==
                  [largeN_trajectory_seed(20260617, 2, 1, 3)]
            @test read(g[LARGE_N_TRAJECTORY_INDICES_KEY]) == [3]
            @test read(g[CoolingTNS.RESULT_MODE_K_INDICES]) == Float64.([-1//2, 1//2])
            @test length(read(g[CoolingTNS.RESULT_MODE_ENERGIES])) == 2
            @test all(isfinite, read(g[CoolingTNS.RESULT_MODE_ENERGIES]))
            @test read(g[CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES]) == [0, 1]
            @test size(read(g[CoolingTNS.RESULT_MODE_HK_TRAJECTORIES])) == (2, 2, 1)
            @test size(read(g[CoolingTNS.RESULT_MODE_NK_TRAJECTORIES])) == (2, 2, 1)
            @test read(g[CoolingTNS.RESULT_MODE_HK_STDERR]) == zeros(2, 2)
            @test read(g[CoolingTNS.RESULT_MODE_NK_STDERR]) == zeros(2, 2)
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
            @test read(f[CoolingTNS.RESULT_RANDOMIZE_TIMES]) == false
            g = f["N2/mcwf/R1"]
            tdvp_sweep_max_bond = read(g[LARGE_N_TDVP_SWEEP_MAX_BOND_KEY])
            @test size(tdvp_sweep_max_bond) == (2, 1)
            @test tdvp_sweep_max_bond[1, 1] == 0
            @test tdvp_sweep_max_bond[2, 1] >= 1
            @test size(read(g[LARGE_N_TDVP_SWEEP_SATURATION_CYCLE_KEY])) == (1,)
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
            @test read(f[CoolingTNS.RESULT_INIT_STATE]) == "product"
            @test read(f[CoolingTNS.RESULT_INIT_THETA]) == 0.0
            @test read(f[CoolingTNS.RESULT_RANDOMIZE_TIMES]) == true
            g = f["N2/mcwf/R1"]
            @test read(g[LARGE_N_TE_LIST_IS_COMMON_KEY]) == true
            @test isequal(read(g[CoolingTNS.RESULT_TE_LIST]), [NaN, 0.0])
            @test isequal(vec(read(g[LARGE_N_TE_LISTS_KEY])), [NaN, 0.0])
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
