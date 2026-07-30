using Test
using CoolingTNS

include(joinpath(@__DIR__, "..", "scripts", "validation",
                 "summarize_tdvp_progress_csv.jl"))

function tdvp_progress_line(; timestamp="2026-06-19T00:00:00",
                            method="mcwf", evolution="continuous",
                            R="2", Dmax="6", g="0.3", stage, step, cycle,
                            delta="0.5", te="2.0", energy_per_site="NaN",
                            relative_energy="NaN", overlap="NaN",
                            system_max_bond="1", system_mean_bond="1.0",
                            evolved_max_bond="NaN", evolved_mean_bond="NaN",
                            tdvp_sweep="NaN", tdvp_time="NaN",
                            stop_on_bond_cap="false",
                            elapsed_seconds,
                            columns=TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS)
    row = Dict(
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TIMESTAMP_KEY => timestamp,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_N_KEY => "4",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_METHOD_KEY => method,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_EVOLUTION_KEY => evolution,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_R_KEY => R,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TRAJECTORY_KEY => "1",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_SEED_KEY => "123",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_DMAX_KEY => Dmax,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_CUTOFF_KEY => "1.0e-6",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_G_KEY => g,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TAU_KEY => "0.2",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_STOP_ON_BOND_CAP_KEY =>
            stop_on_bond_cap,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_KEY => stage,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_STEP_KEY => string(step),
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_CYCLE_KEY => string(cycle),
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_DELTA_KEY => delta,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TE_KEY => te,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_ENERGY_PER_SITE_KEY =>
            energy_per_site,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_RELATIVE_ENERGY_KEY =>
            relative_energy,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_OVERLAP_KEY => overlap,
        TDVPProgressCSVSummary.LARGE_N_SYSTEM_MAX_BOND_KEY => system_max_bond,
        TDVPProgressCSVSummary.LARGE_N_SYSTEM_MEAN_BOND_KEY => system_mean_bond,
        TDVPProgressCSVSummary.LARGE_N_EVOLVED_MAX_BOND_KEY => evolved_max_bond,
        TDVPProgressCSVSummary.LARGE_N_EVOLVED_MEAN_BOND_KEY => evolved_mean_bond,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TDVP_SWEEP_KEY => tdvp_sweep,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TDVP_TIME_KEY => tdvp_time,
        TDVPProgressCSVSummary.LARGE_N_ELAPSED_SECONDS_KEY => string(elapsed_seconds),
    )
    return join(
        (row[col] for col in columns),
        ",",
    )
end

@testset "TDVP progress CSV summary script" begin
    @test TDVPProgressCSVSummary.parse_csv_line(
        "\"contains,comma\",\"escaped \"\"quote\"\"\",plain"
    ) ==
        ["contains,comma", "escaped \"quote\"", "plain"]
    @test TDVPProgressCSVSummary.parse_csv_line("plain,\"with,comma\"") ==
          TDVPProgressCSVSummary.parse_largeN_progress_csv_line(
              "plain,\"with,comma\"",
          )
    help_args = TDVPProgressCSVSummary.parse_args(["--help"])
    @test help_args.help
    @test isempty(help_args.paths)
    @test help_args.cap === nothing
    @test !help_args.stopped_on_cap
    @test !help_args.compact
    short_help_args = TDVPProgressCSVSummary.parse_args(["-h"])
    @test short_help_args.help
    @test TDVPProgressCSVSummary.parse_args(["--help", "--cap"]).help
    @test occursin(
        "[--cap D] [--stopped-on-cap] [--compact] [--help] PROGRESS.csv",
        sprint(TDVPProgressCSVSummary.usage),
    )
    cap_args = TDVPProgressCSVSummary.parse_args(["--cap", "12", "progress.csv"])
    @test cap_args.cap == 12
    @test cap_args.paths == ["progress.csv"]
    @test !cap_args.stopped_on_cap
    @test !cap_args.compact
    stopped_args = TDVPProgressCSVSummary.parse_args([
        "--stopped-on-cap", "progress.csv",
    ])
    @test stopped_args.stopped_on_cap
    @test stopped_args.paths == ["progress.csv"]
    @test !stopped_args.compact
    compact_args = TDVPProgressCSVSummary.parse_args(["--compact", "progress.csv"])
    @test compact_args.compact
    @test compact_args.paths == ["progress.csv"]
    @test !compact_args.stopped_on_cap
    @test_throws ArgumentError TDVPProgressCSVSummary.parse_args(["--cap"])
    @test_throws ArgumentError TDVPProgressCSVSummary.parse_args(
        ["--cap", "-1", "progress.csv"],
    )
    @test_throws ArgumentError TDVPProgressCSVSummary.parse_args(["--cap", "--help"])
    @test_throws ArgumentError TDVPProgressCSVSummary.parse_args(["--cap", "-h"])
    @test_throws ArgumentError TDVPProgressCSVSummary.parse_args(["--unknown"])

    @test TDVPProgressCSVSummary.largeN_method_kind_from_name("mcwf") === :mcwf
    @test TDVPProgressCSVSummary.largeN_method_kind_from_name("MPO") === :mpo
    @test TDVPProgressCSVSummary.largeN_method_maxdim_from_name("mcwf", 7) ==
          tn_method_maxdim(MonteCarloWavefunction(), 7)
    @test TDVPProgressCSVSummary.largeN_method_maxdim_from_name("mpo", 7) ==
          tn_method_maxdim(DensityMatrix(), 7)
    @test TDVPProgressCSVSummary.default_progress_cap("mcwf", 7) ==
          TDVPProgressCSVSummary.largeN_method_maxdim_from_name("mcwf", 7)
    @test TDVPProgressCSVSummary.default_progress_cap("mpo", 7) ==
          TDVPProgressCSVSummary.largeN_method_maxdim_from_name("mpo", 7)
    large_dmax = big(typemax(Int)) + 1
    @test TDVPProgressCSVSummary.largeN_method_maxdim_from_name("mcwf", large_dmax) ==
          large_dmax
    @test TDVPProgressCSVSummary.largeN_method_maxdim_from_name("mpo", large_dmax) ==
          4 * large_dmax
    @test TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGES == (
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_INITIAL,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_PREPARED,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_EVOLVED,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_UPDATED,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_TDVP_SWEEP,
    )
    @test TDVPProgressCSVSummary.largeN_progress_stage(:updated) ==
          TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_UPDATED
    @test TDVPProgressCSVSummary.progress_detuning_coverage_status(0, 5, 0) ==
          TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_NO_COMPLETED_CYCLES
    @test TDVPProgressCSVSummary.progress_detuning_coverage_status(0, 5, 2) ==
          TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_MISSING_DETUNING_VALUES
    @test TDVPProgressCSVSummary.progress_detuning_coverage_status(1, 1, 1) ==
          TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_SINGLE_DETUNING
    @test TDVPProgressCSVSummary.progress_detuning_coverage_status(2, 5, 2) ==
          TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_PARTIAL_GRID_OBSERVED
    @test TDVPProgressCSVSummary.progress_detuning_coverage_status(
        2, 5, 2; stopped=true
    ) == TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_STOPPED_PARTIAL_GRID
    @test TDVPProgressCSVSummary.progress_detuning_coverage_status(5, 5, 5) ==
          TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_FULL_GRID
    @test TDVPProgressCSVSummary.visited_detunings_label_from_counts([2], 3) ==
          "2/3"
    @test TDVPProgressCSVSummary.validate_progress_csv_header(
        "current.csv",
        collect(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS),
    ) === nothing
    @test TDVPProgressCSVSummary.validate_progress_csv_header(
        "future.csv",
        vcat(collect(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS),
             ["future_observable"]),
    ) === nothing
    legacy_progress_columns = String[
        column for column in TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS
        if column != TDVPProgressCSVSummary.LARGE_N_PROGRESS_G_KEY
    ]
    @test TDVPProgressCSVSummary.validate_progress_csv_header(
        "legacy.csv", legacy_progress_columns,
    ) === nothing
    legacy_stop_columns = String[
        column for column in TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS
        if column != TDVPProgressCSVSummary.LARGE_N_PROGRESS_STOP_ON_BOND_CAP_KEY
    ]
    @test TDVPProgressCSVSummary.validate_progress_csv_header(
        "legacy_stop.csv", legacy_stop_columns,
    ) === nothing
    @test TDVPProgressCSVSummary.progress_bool_value(
        "true", TDVPProgressCSVSummary.LARGE_N_PROGRESS_STOP_ON_BOND_CAP_KEY,
    )
    @test !TDVPProgressCSVSummary.progress_bool_value(
        "0", TDVPProgressCSVSummary.LARGE_N_PROGRESS_STOP_ON_BOND_CAP_KEY,
    )
    @test !TDVPProgressCSVSummary.progress_bool_value(
        "", TDVPProgressCSVSummary.LARGE_N_PROGRESS_STOP_ON_BOND_CAP_KEY,
    )
    @test_throws ArgumentError TDVPProgressCSVSummary.progress_bool_value(
        "maybe", TDVPProgressCSVSummary.LARGE_N_PROGRESS_STOP_ON_BOND_CAP_KEY,
    )
    missing_R_columns = String[
        column for column in TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS
        if column != TDVPProgressCSVSummary.LARGE_N_PROGRESS_R_KEY
    ]
    missing_R_err = try
        TDVPProgressCSVSummary.validate_progress_csv_header(
            "missing_R.csv", missing_R_columns,
        )
        nothing
    catch err
        err
    end
    @test missing_R_err isa ArgumentError
    @test occursin("missing required column", sprint(showerror, missing_R_err))
    @test occursin(TDVPProgressCSVSummary.LARGE_N_PROGRESS_R_KEY,
                   sprint(showerror, missing_R_err))
    @test occursin(TDVPProgressCSVSummary.LARGE_N_PROGRESS_G_KEY,
                   sprint(showerror, missing_R_err))
    duplicate_header_err = try
        TDVPProgressCSVSummary.validate_progress_csv_header(
            "duplicate.csv",
            vcat(
                collect(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS),
                [TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_KEY],
            ),
        )
        nothing
    catch err
        err
    end
    @test duplicate_header_err isa ArgumentError
    @test occursin("duplicate column", sprint(showerror, duplicate_header_err))
    @test occursin(TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_KEY,
                   sprint(showerror, duplicate_header_err))
    @test TDVPProgressCSVSummary.LARGE_N_PROGRESS_GROUP_COLUMNS == (
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_N_KEY,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_METHOD_KEY,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_EVOLUTION_KEY,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_R_KEY,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TRAJECTORY_KEY,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_SEED_KEY,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_DMAX_KEY,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_CUTOFF_KEY,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_G_KEY,
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TAU_KEY,
    )
    fixed_identity_a = Dict(
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_N_KEY => "64",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_METHOD_KEY => "mcwf",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_EVOLUTION_KEY => "continuous",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_R_KEY => "5",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TRAJECTORY_KEY => "1",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_SEED_KEY => "84310618",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_DMAX_KEY => "96",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_CUTOFF_KEY => "1.0e-6",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_G_KEY => "0.05",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TAU_KEY => "0.2",
        TDVPProgressCSVSummary.LARGE_N_PROGRESS_TE_KEY => "0.34",
    )
    fixed_identity_b = merge(
        fixed_identity_a,
        Dict(TDVPProgressCSVSummary.LARGE_N_PROGRESS_TE_KEY => "1.86"),
    )
    @test TDVPProgressCSVSummary.group_key(fixed_identity_a) ==
          TDVPProgressCSVSummary.group_key(fixed_identity_b)
    fixed_identity_c = merge(
        fixed_identity_a,
        Dict(TDVPProgressCSVSummary.LARGE_N_PROGRESS_G_KEY => "0.1"),
    )
    @test TDVPProgressCSVSummary.group_key(fixed_identity_a) !=
          TDVPProgressCSVSummary.group_key(fixed_identity_c)
    fixed_label = TDVPProgressCSVSummary.group_label(
        TDVPProgressCSVSummary.group_key(fixed_identity_a)
    )
    @test fixed_label.R == "5"
    @test fixed_label.g == "0.05"
    @test TDVPProgressCSVSummary.unique_progress_file_labels([
        "run_a.csv",
        "run_b.csv",
    ]) == ["run_a.csv", "run_b.csv"]
    @test TDVPProgressCSVSummary.unique_progress_file_labels([
        joinpath("D192", "progress.csv"),
        joinpath("D256", "progress.csv"),
    ]) == [
        joinpath("D192", "progress.csv"),
        joinpath("D256", "progress.csv"),
    ]
    same_progress_path = joinpath("D192", "progress.csv")
    @test TDVPProgressCSVSummary.unique_progress_file_labels([
        same_progress_path,
        same_progress_path,
    ]) == [same_progress_path, "$(same_progress_path)#2"]
    err = try
        TDVPProgressCSVSummary.default_progress_cap("ed", 7)
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    @test occursin("pass --cap D explicitly", sprint(showerror, err))
    @test TDVPProgressCSVSummary.format_protocol_float(2.0) == "2"
    @test TDVPProgressCSVSummary.format_protocol_float(0.5) == "0.5"
    @test TDVPProgressCSVSummary.progress_te_audit_values([
        Dict(TDVPProgressCSVSummary.LARGE_N_PROGRESS_TE_KEY => "NaN"),
    ]) == Float64[]
    @test TDVPProgressCSVSummary.progress_te_audit_values([
        Dict(TDVPProgressCSVSummary.LARGE_N_PROGRESS_TE_KEY => "2.0"),
        Dict(TDVPProgressCSVSummary.LARGE_N_PROGRESS_TE_KEY => "2.00"),
    ]) == [2.0]
    @test TDVPProgressCSVSummary.progress_te_audit_values([
        Dict(TDVPProgressCSVSummary.LARGE_N_PROGRESS_TE_KEY => "1.25"),
        Dict(TDVPProgressCSVSummary.LARGE_N_PROGRESS_TE_KEY => "0.5"),
        Dict(TDVPProgressCSVSummary.LARGE_N_PROGRESS_TE_KEY => "0.75"),
    ]) == [0.5, 1.25]
    @test TDVPProgressCSVSummary.progress_te_label(Float64[]) ==
          TDVPProgressCSVSummary.LARGE_N_LABEL_NA
    @test TDVPProgressCSVSummary.progress_te_label([0.5]) == "0.5"
    @test TDVPProgressCSVSummary.progress_te_label([0.5, 1.25]) == "0.5-1.25"
    @test TDVPProgressCSVSummary.progress_time_protocol_label(Float64[]) ==
          TDVPProgressCSVSummary.LARGE_N_LABEL_NA
    @test TDVPProgressCSVSummary.progress_time_protocol_label([2.0]) ==
          "fixed_observed"
    @test TDVPProgressCSVSummary.progress_time_protocol_label([0.5, 1.25]) ==
          "variable_observed"

    path = tempname() * ".csv"
    try
        open(path, "w") do io
            println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
            println(io, tdvp_progress_line(
                timestamp="\"contains,comma\"",
                stage="initial", step=1, cycle=0, delta="NaN", te="NaN",
                energy_per_site="1.0", relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN", elapsed_seconds=0.5,
            ))
            println(io, tdvp_progress_line(
                stage="prepared", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="1", elapsed_seconds=1.0,
            ))
            println(io, tdvp_progress_line(
                stage="tdvp_sweep", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="4",
                tdvp_sweep="1", tdvp_time="0.2", elapsed_seconds=3.0,
            ))
            println(io, tdvp_progress_line(
                stage="tdvp_sweep", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="6",
                tdvp_sweep="2", tdvp_time="0.4", elapsed_seconds=8.0,
            ))
            println(io, tdvp_progress_line(
                stage="updated", step=2, cycle=1,
                energy_per_site="0.5", relative_energy="1.5", overlap="0.1",
                system_max_bond="5", evolved_max_bond="6", elapsed_seconds=9.0,
            ))
            println(io, tdvp_progress_line(
                stage="prepared", step=3, cycle=2,
                system_max_bond="5", evolved_max_bond="5", elapsed_seconds=10.0,
            ))
            println(io, tdvp_progress_line(
                stage="tdvp_sweep", step=3, cycle=2,
                system_max_bond="5", evolved_max_bond="7",
                tdvp_sweep="1", tdvp_time="0.2", elapsed_seconds=15.0,
            ))
            println(io, tdvp_progress_line(
                stage="updated", step=3, cycle=2,
                delta="0.7",
                energy_per_site="0.25", relative_energy="1.25", overlap="0.2",
                system_max_bond="6", evolved_max_bond="7", elapsed_seconds=16.0,
            ))
            println(io, tdvp_progress_line(
                stage="prepared", step=4, cycle=3,
                system_max_bond="6", evolved_max_bond="6", elapsed_seconds=17.0,
            ))
            println(io, tdvp_progress_line(
                stage="tdvp_sweep", step=4, cycle=3,
                system_max_bond="6", evolved_max_bond="7",
                tdvp_sweep="1", tdvp_time="0.2", elapsed_seconds=18.0,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="initial", step=1, cycle=0, delta="NaN", te="NaN",
                energy_per_site="1.0", relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN", elapsed_seconds=0.5,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="prepared", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="12", elapsed_seconds=17.0,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="evolved", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="23", elapsed_seconds=18.0,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="updated", step=2, cycle=1,
                energy_per_site="0.75", relative_energy="1.75", overlap="0.1",
                system_max_bond="20", evolved_max_bond="23", elapsed_seconds=19.0,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="prepared", step=3, cycle=2,
                system_max_bond="20", evolved_max_bond="20", elapsed_seconds=20.0,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="evolved", step=3, cycle=2,
                system_max_bond="20", evolved_max_bond="24", elapsed_seconds=21.0,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="updated", step=3, cycle=2,
                delta="0.8",
                energy_per_site="0.5", relative_energy="1.5", overlap="0.2",
                system_max_bond="24", evolved_max_bond="24", elapsed_seconds=22.0,
            ))
        end

        rows = TDVPProgressCSVSummary.summarize_progress_file(path)
        @test length(rows) == 2
        row = only(filter(row -> row.method == "mcwf", rows))
        @test row.N == 4
        @test row.R == 2
        @test row.g == "0.3"
        @test row.time_protocol == "fixed_observed"
        @test row.te == "2"
        @test row.threshold == 6
        @test row.completed_cycles == 2
        @test row.visited_detunings == "2/2"
        @test row.detuning_coverage ==
              TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_FULL_GRID
        @test row.final_energy == 0.25
        @test row.system_effective_bond == ">=6"
        @test row.evolved_effective_bond == ">=7"
        @test row.bond_status ==
              TDVPProgressCSVSummary.LARGE_N_BOND_STATUS_SYSTEM_AND_TDVP_SWEEP_CAP
        @test row.system_cap_cycle == 2
        @test row.evolved_cap_cycle == 0
        @test row.tdvp_sweep_cap_cycle == 1
        @test row.tdvp_sweep_cap_sweep == 2
        @test row.transient_cap_cycle == 1
        @test row.transient_cap_sweep == 2
        @test row.max_sweep_increment == 5.0
        @test row.max_sweep_cycle == 1
        @test row.max_sweep == 2
        @test row.last_step == 4
        @test row.last_cycle == 3
        @test row.last_stage == TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_TDVP_SWEEP
        @test length(row.updates) == 2

        mpo_row = only(filter(row -> row.method == "mpo", rows))
        @test mpo_row.evolution == "trotter"
        @test mpo_row.R == 3
        @test mpo_row.g == "0.3"
        @test mpo_row.threshold == 24
        @test mpo_row.completed_cycles == 2
        @test mpo_row.visited_detunings == "2/3"
        @test !mpo_row.stop_on_bond_cap
        @test mpo_row.detuning_coverage ==
              TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_PARTIAL_GRID_OBSERVED
        @test mpo_row.final_energy == 0.5
        @test mpo_row.system_effective_bond == ">=24"
        @test mpo_row.evolved_effective_bond == ">=24"
        @test mpo_row.bond_status ==
              TDVPProgressCSVSummary.LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_CAP
        @test mpo_row.system_cap_cycle == 2
        @test mpo_row.evolved_cap_cycle == 2
        @test mpo_row.tdvp_sweep_cap_cycle == 0
        @test mpo_row.tdvp_sweep_cap_sweep === nothing
        @test mpo_row.transient_cap_cycle == 2
        @test mpo_row.transient_cap_sweep === nothing
        @test mpo_row.max_sweep_cycle == 0
        @test mpo_row.last_step == 3
        @test mpo_row.last_cycle == 2
        @test mpo_row.last_stage == TDVPProgressCSVSummary.LARGE_N_PROGRESS_STAGE_UPDATED

        modern_flag_rows = TDVPProgressCSVSummary.summarize_progress_file(
            path; stopped_on_cap=true,
        )
        modern_flag_mpo_row = only(
            filter(row -> row.method == "mpo", modern_flag_rows)
        )
        @test modern_flag_mpo_row.detuning_coverage ==
              TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_PARTIAL_GRID_OBSERVED
        @test !modern_flag_mpo_row.stop_on_bond_cap

        output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    TDVPProgressCSVSummary.print_markdown(rows)
                end
            end
            read(output_path, String)
        end
        summary_lines = split(chomp(output), "\n")
        @test count(==('|'), summary_lines[1]) == count(==('|'), summary_lines[2])
        @test count(==('|'), summary_lines[1]) - 1 == 27
        @test occursin(
            "| file | N | method | evolution | time protocol | te values | R | g | traj |",
            output,
        )
        @test occursin("| time protocol | te values |", output)
        @test occursin("| continuous | fixed_observed | 2 | 2 | 0.3 |", output)
        @test occursin("| completed cycles | visited detunings | detuning coverage |", output)
        @test occursin("| sys cap | evolved cap | tdvp sweep cap | first transient cap |", output)
        @test occursin("| last step | last cycle | last stage |", output)
        @test occursin(
            TDVPProgressCSVSummary.LARGE_N_BOND_STATUS_SYSTEM_AND_TDVP_SWEEP_CAP,
            output,
        )
        @test occursin(
            TDVPProgressCSVSummary.LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_CAP,
            output,
        )
        @test occursin("| R | g | traj | cycle | delta | E/N |", output)
        @test occursin(
            "| 2 | 0.3 | 1 | 2 | 0.70000000 | 0.25000000 | 6 | 7 | 16.0 |",
            output,
        )
        @test occursin("| 4 | 3 | tdvp_sweep |", output)
        compact_return = Ref{Any}(missing)
        compact_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    compact_return[] =
                        TDVPProgressCSVSummary.print_markdown(rows; compact=true)
                end
            end
            read(output_path, String)
        end
        @test compact_return[] === nothing
        @test occursin(
            "| file | N | method | evolution | time protocol | te values | R | g | traj |",
            compact_output,
        )
        @test !occursin("| R | g | traj | cycle | delta | E/N |", compact_output)
    finally
        rm(path; force=true)
    end

    variable_te_path = tempname() * ".csv"
    try
        open(variable_te_path, "w") do io
            println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
            println(io, tdvp_progress_line(
                stage="initial", step=1, cycle=0, delta="NaN", te="NaN",
                energy_per_site="1.0", relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN", elapsed_seconds=0.5,
            ))
            println(io, tdvp_progress_line(
                stage="updated", step=2, cycle=1, te="0.5",
                energy_per_site="0.5", relative_energy="1.5", overlap="0.1",
                system_max_bond="5", evolved_max_bond="6", elapsed_seconds=3.0,
            ))
            println(io, tdvp_progress_line(
                stage="updated", step=3, cycle=2, te="1.25",
                energy_per_site="0.25", relative_energy="1.25", overlap="0.2",
                system_max_bond="6", evolved_max_bond="7", elapsed_seconds=6.0,
            ))
        end

        variable_te_row = only(TDVPProgressCSVSummary.summarize_progress_file(
            variable_te_path
        ))
        @test variable_te_row.time_protocol == "variable_observed"
        @test variable_te_row.te == "0.5-1.25"
        variable_te_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    TDVPProgressCSVSummary.print_markdown(
                        [variable_te_row]; compact=true
                    )
                end
            end
            read(output_path, String)
        end
        @test occursin(
            "| continuous | variable_observed | 0.5-1.25 | 2 | 0.3 |",
            variable_te_output,
        )
    finally
        rm(variable_te_path; force=true)
    end

    legacy_stop_path = tempname() * ".csv"
    try
        open(legacy_stop_path, "w") do io
            println(io, join(legacy_stop_columns, ","))
            println(io, tdvp_progress_line(
                columns=legacy_stop_columns,
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="updated", step=2, cycle=1, delta="0.5",
                energy_per_site="0.75", relative_energy="1.75", overlap="0.1",
                system_max_bond="10", evolved_max_bond="12", elapsed_seconds=1.0,
            ))
            println(io, tdvp_progress_line(
                columns=legacy_stop_columns,
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="evolved", step=3, cycle=2, delta="0.8",
                system_max_bond="10", evolved_max_bond="24", elapsed_seconds=2.0,
            ))
            println(io, tdvp_progress_line(
                columns=legacy_stop_columns,
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stage="updated", step=3, cycle=2, delta="0.8",
                energy_per_site="0.5", relative_energy="1.5", overlap="0.2",
                system_max_bond="24", evolved_max_bond="24", elapsed_seconds=3.0,
            ))
        end
        legacy_default_row = only(TDVPProgressCSVSummary.summarize_progress_file(
            legacy_stop_path
        ))
        @test !legacy_default_row.stop_on_bond_cap
        @test legacy_default_row.detuning_coverage ==
              TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_PARTIAL_GRID_OBSERVED
        legacy_flag_row = only(TDVPProgressCSVSummary.summarize_progress_file(
            legacy_stop_path; stopped_on_cap=true,
        ))
        @test legacy_flag_row.stop_on_bond_cap
        @test legacy_flag_row.detuning_coverage ==
              TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_STOPPED_PARTIAL_GRID
    finally
        rm(legacy_stop_path; force=true)
    end

    stored_stop_path = tempname() * ".csv"
    try
        open(stored_stop_path, "w") do io
            println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stop_on_bond_cap="true",
                stage="updated", step=2, cycle=1, delta="0.5",
                energy_per_site="0.75", relative_energy="1.75", overlap="0.1",
                system_max_bond="10", evolved_max_bond="12", elapsed_seconds=1.0,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stop_on_bond_cap="true",
                stage="evolved", step=3, cycle=2, delta="0.8",
                system_max_bond="10", evolved_max_bond="24", elapsed_seconds=2.0,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stop_on_bond_cap="true",
                stage="updated", step=3, cycle=2, delta="0.8",
                energy_per_site="0.5", relative_energy="1.5", overlap="0.2",
                system_max_bond="24", evolved_max_bond="24", elapsed_seconds=3.0,
            ))
        end
        stored_stop_row = only(TDVPProgressCSVSummary.summarize_progress_file(
            stored_stop_path
        ))
        @test stored_stop_row.stop_on_bond_cap
        @test stored_stop_row.visited_detunings == "2/3"
        @test stored_stop_row.detuning_coverage ==
              TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_STOPPED_PARTIAL_GRID
    finally
        rm(stored_stop_path; force=true)
    end

    inconsistent_stop_path = tempname() * ".csv"
    try
        open(inconsistent_stop_path, "w") do io
            println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stop_on_bond_cap="true",
                stage="updated", step=2, cycle=1, delta="0.5",
                energy_per_site="0.75", relative_energy="1.75", overlap="0.1",
                system_max_bond="10", evolved_max_bond="12", elapsed_seconds=1.0,
            ))
            println(io, tdvp_progress_line(
                method="mpo", evolution="trotter", R="3", Dmax="6",
                stop_on_bond_cap="false",
                stage="updated", step=3, cycle=2, delta="0.8",
                energy_per_site="0.5", relative_energy="1.5", overlap="0.2",
                system_max_bond="20", evolved_max_bond="22", elapsed_seconds=2.0,
            ))
        end
        @test_throws ArgumentError TDVPProgressCSVSummary.summarize_progress_file(
            inconsistent_stop_path
        )
    finally
        rm(inconsistent_stop_path; force=true)
    end

    duplicate_root = mktempdir()
    try
        duplicate_paths = [
            joinpath(duplicate_root, "D192", "progress.csv"),
            joinpath(duplicate_root, "D256", "progress.csv"),
        ]
        for (csv_path, dmax, energy) in zip(duplicate_paths, ("192", "256"), ("0.8", "0.7"))
            mkpath(dirname(csv_path))
            open(csv_path, "w") do io
                println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
                println(io, tdvp_progress_line(
                    R="10", Dmax=dmax, stage="initial", step=1, cycle=0,
                    delta="NaN", te="NaN", energy_per_site="1.0",
                    relative_energy="2.0", overlap="0.0", system_max_bond="1",
                    evolved_max_bond="NaN", elapsed_seconds=0.5,
                ))
                println(io, tdvp_progress_line(
                    R="10", Dmax=dmax, stage="updated", step=2, cycle=1,
                    energy_per_site=energy, relative_energy="1.5", overlap="0.1",
                    system_max_bond="5", evolved_max_bond="6", elapsed_seconds=3.0,
                ))
            end
        end

        duplicate_rows = TDVPProgressCSVSummary.summarize_progress_files(duplicate_paths)
        @test [row.file for row in duplicate_rows] == [
            joinpath("D192", "progress.csv"),
            joinpath("D256", "progress.csv"),
        ]

        duplicate_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    TDVPProgressCSVSummary.print_markdown(duplicate_rows)
                end
            end
            read(output_path, String)
        end
        @test occursin("| $(joinpath("D192", "progress.csv")) |", duplicate_output)
        @test occursin("| $(joinpath("D256", "progress.csv")) |", duplicate_output)
    finally
        rm(duplicate_root; force=true, recursive=true)
    end

    legacy_g_path = tempname() * ".csv"
    try
        legacy_columns = filter(
            col -> col != TDVPProgressCSVSummary.LARGE_N_PROGRESS_G_KEY,
            TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS,
        )
        open(legacy_g_path, "w") do io
            println(io, join(legacy_columns, ","))
            println(io, tdvp_progress_line(
                columns=legacy_columns,
                stage="initial", step=1, cycle=0, delta="NaN", te="NaN",
                energy_per_site="1.0", relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN", elapsed_seconds=0.5,
            ))
            println(io, tdvp_progress_line(
                columns=legacy_columns,
                stage="updated", step=2, cycle=1,
                energy_per_site="0.5", relative_energy="1.5", overlap="0.1",
                system_max_bond="5", evolved_max_bond="6", elapsed_seconds=3.0,
            ))
        end

        legacy_g_rows = TDVPProgressCSVSummary.summarize_progress_file(legacy_g_path)
        legacy_g_row = only(legacy_g_rows)
        @test legacy_g_row.g == ""
        @test legacy_g_row.has_g_column == false

        legacy_g_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    TDVPProgressCSVSummary.print_markdown(legacy_g_rows)
                end
            end
            read(output_path, String)
        end
        @test occursin("| 2 | legacy_missing | 1 | 123 |", legacy_g_output)
        @test occursin(
            "| 2 | legacy_missing | 1 | 1 | 0.50000000 | 0.50000000 | 5 | 6 | 3.0 |",
            legacy_g_output,
        )
    finally
        rm(legacy_g_path; force=true)
    end

    empty_g_path = tempname() * ".csv"
    try
        open(empty_g_path, "w") do io
            println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
            println(io, tdvp_progress_line(
                g="",
                stage="initial", step=1, cycle=0, delta="NaN", te="NaN",
                energy_per_site="1.0", relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN", elapsed_seconds=0.5,
            ))
            println(io, tdvp_progress_line(
                g="",
                stage="updated", step=2, cycle=1,
                energy_per_site="0.5", relative_energy="1.5", overlap="0.1",
                system_max_bond="5", evolved_max_bond="6", elapsed_seconds=3.0,
            ))
        end

        empty_g_rows = TDVPProgressCSVSummary.summarize_progress_file(empty_g_path)
        empty_g_row = only(empty_g_rows)
        @test empty_g_row.g == ""
        @test empty_g_row.has_g_column == true

        empty_g_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    TDVPProgressCSVSummary.print_markdown(empty_g_rows)
                end
            end
            read(output_path, String)
        end
        @test occursin("| 2 | missing | 1 | 123 |", empty_g_output)
        @test occursin(
            "| 2 | missing | 1 | 1 | 0.50000000 | 0.50000000 | 5 | 6 | 3.0 |",
            empty_g_output,
        )
    finally
        rm(empty_g_path; force=true)
    end

    missing_delta_path = tempname() * ".csv"
    try
        open(missing_delta_path, "w") do io
            println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
            println(io, tdvp_progress_line(
                R="5", stage="initial", step=1, cycle=0, delta="NaN", te="NaN",
                energy_per_site="1.0", relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN", elapsed_seconds=0.5,
            ))
            println(io, tdvp_progress_line(
                R="5", stage="updated", step=2, cycle=1, delta="NaN",
                energy_per_site="0.9", relative_energy="1.9", overlap="0.1",
                system_max_bond="2", evolved_max_bond="3", elapsed_seconds=1.5,
            ))
        end

        missing_delta_row = only(TDVPProgressCSVSummary.summarize_progress_file(
            missing_delta_path
        ))
        @test missing_delta_row.completed_cycles == 1
        @test missing_delta_row.visited_detunings == "0/5"
        @test missing_delta_row.detuning_coverage ==
              TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_MISSING_DETUNING_VALUES
    finally
        rm(missing_delta_path; force=true)
    end

    for bad_stage in ("renormalized", "")
        bad_stage_path = tempname() * ".csv"
        try
            open(bad_stage_path, "w") do io
                println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
                println(io, tdvp_progress_line(
                    stage=bad_stage, step=1, cycle=0, delta="NaN", te="NaN",
                    energy_per_site="1.0", relative_energy="2.0", overlap="0.0",
                    system_max_bond="1", evolved_max_bond="NaN", elapsed_seconds=0.5,
                ))
            end
            @test_throws ArgumentError TDVPProgressCSVSummary.read_progress_csv(bad_stage_path)
        finally
            rm(bad_stage_path; force=true)
        end
    end

    missing_required_column_path = tempname() * ".csv"
    try
        missing_required_columns = String[
            column for column in TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS
            if column != TDVPProgressCSVSummary.LARGE_N_PROGRESS_R_KEY
        ]
        open(missing_required_column_path, "w") do io
            println(io, join(missing_required_columns, ","))
            println(io, tdvp_progress_line(
                columns=missing_required_columns,
                stage="initial", step=1, cycle=0, delta="NaN", te="NaN",
                energy_per_site="1.0", relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN", elapsed_seconds=0.5,
            ))
        end
        err = try
            TDVPProgressCSVSummary.read_progress_csv(missing_required_column_path)
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("missing required column", sprint(showerror, err))
        @test occursin(TDVPProgressCSVSummary.LARGE_N_PROGRESS_R_KEY,
                       sprint(showerror, err))
    finally
        rm(missing_required_column_path; force=true)
    end

    tdvp_only_path = tempname() * ".csv"
    try
        open(tdvp_only_path, "w") do io
            println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
            println(io, tdvp_progress_line(
                stage="initial", step=1, cycle=0, delta="NaN", te="NaN",
                energy_per_site="1.0", relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN", elapsed_seconds=0.5,
            ))
            println(io, tdvp_progress_line(
                stage="prepared", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="1", elapsed_seconds=1.0,
            ))
            println(io, tdvp_progress_line(
                stage="tdvp_sweep", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="6",
                tdvp_sweep="1", tdvp_time="0.2", elapsed_seconds=2.0,
            ))
            println(io, tdvp_progress_line(
                stage="updated", step=2, cycle=1,
                energy_per_site="0.5", relative_energy="1.5", overlap="0.1",
                system_max_bond="5", evolved_max_bond="5", elapsed_seconds=3.0,
            ))
        end

        tdvp_only_row = only(TDVPProgressCSVSummary.summarize_progress_file(tdvp_only_path))
        @test tdvp_only_row.system_effective_bond == "5"
        @test tdvp_only_row.evolved_effective_bond == ">=6"
        @test tdvp_only_row.bond_status ==
              TDVPProgressCSVSummary.LARGE_N_BOND_STATUS_TDVP_SWEEP_CAP
        @test tdvp_only_row.system_cap_cycle == 0
        @test tdvp_only_row.evolved_cap_cycle == 0
        @test tdvp_only_row.tdvp_sweep_cap_cycle == 1
        @test tdvp_only_row.tdvp_sweep_cap_sweep == 1
        @test tdvp_only_row.transient_cap_cycle == 1
        @test tdvp_only_row.transient_cap_sweep == 1
    finally
        rm(tdvp_only_path; force=true)
    end

    both_caps_path = tempname() * ".csv"
    try
        open(both_caps_path, "w") do io
            println(io, join(TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS, ","))
            println(io, tdvp_progress_line(
                R="4", stage="initial", step=1, cycle=0,
                delta="NaN", te="NaN", energy_per_site="1.0",
                relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN",
                elapsed_seconds=0.5,
            ))
            println(io, tdvp_progress_line(
                R="4", stage="prepared", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="1",
                elapsed_seconds=1.0,
            ))
            println(io, tdvp_progress_line(
                R="4", stage="tdvp_sweep", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="6",
                tdvp_sweep="1", tdvp_time="0.2", elapsed_seconds=2.0,
            ))
            println(io, tdvp_progress_line(
                R="4", stage="evolved", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="6",
                elapsed_seconds=2.5,
            ))
            println(io, tdvp_progress_line(
                R="4", stage="updated", step=2, cycle=1,
                energy_per_site="0.5", relative_energy="1.5", overlap="0.1",
                system_max_bond="5", evolved_max_bond="6",
                elapsed_seconds=3.0,
            ))

            println(io, tdvp_progress_line(
                R="5", stage="initial", step=1, cycle=0,
                delta="NaN", te="NaN", energy_per_site="1.0",
                relative_energy="2.0", overlap="0.0",
                system_max_bond="1", evolved_max_bond="NaN",
                elapsed_seconds=0.5,
            ))
            println(io, tdvp_progress_line(
                R="5", stage="prepared", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="1",
                elapsed_seconds=1.0,
            ))
            println(io, tdvp_progress_line(
                R="5", stage="evolved", step=2, cycle=1,
                system_max_bond="1", evolved_max_bond="6",
                elapsed_seconds=2.0,
            ))
            println(io, tdvp_progress_line(
                R="5", stage="updated", step=2, cycle=1,
                energy_per_site="0.5", relative_energy="1.5", overlap="0.1",
                system_max_bond="5", evolved_max_bond="6",
                elapsed_seconds=3.0,
            ))
            println(io, tdvp_progress_line(
                R="5", stage="prepared", step=3, cycle=2,
                system_max_bond="5", evolved_max_bond="5",
                elapsed_seconds=4.0,
            ))
            println(io, tdvp_progress_line(
                R="5", stage="tdvp_sweep", step=3, cycle=2,
                system_max_bond="5", evolved_max_bond="7",
                tdvp_sweep="2", tdvp_time="0.3", elapsed_seconds=5.0,
            ))
            println(io, tdvp_progress_line(
                R="5", stage="updated", step=3, cycle=2,
                energy_per_site="0.25", relative_energy="1.25",
                overlap="0.2", system_max_bond="5",
                evolved_max_bond="7", elapsed_seconds=6.0,
            ))
        end

        both_caps_rows = TDVPProgressCSVSummary.summarize_progress_file(both_caps_path)
        @test length(both_caps_rows) == 2

        tie_row = only(filter(row -> row.R == 4, both_caps_rows))
        @test tie_row.bond_status ==
              TDVPProgressCSVSummary.LARGE_N_BOND_STATUS_EVOLVED_AND_TDVP_SWEEP_CAP
        @test tie_row.system_cap_cycle == 0
        @test tie_row.evolved_cap_cycle == 1
        @test tie_row.tdvp_sweep_cap_cycle == 1
        @test tie_row.tdvp_sweep_cap_sweep == 1
        @test tie_row.transient_cap_cycle == 1
        @test tie_row.transient_cap_sweep == 1

        evolved_first_row = only(filter(row -> row.R == 5, both_caps_rows))
        @test evolved_first_row.bond_status ==
              TDVPProgressCSVSummary.LARGE_N_BOND_STATUS_EVOLVED_AND_TDVP_SWEEP_CAP
        @test evolved_first_row.system_cap_cycle == 0
        @test evolved_first_row.evolved_cap_cycle == 1
        @test evolved_first_row.tdvp_sweep_cap_cycle == 2
        @test evolved_first_row.tdvp_sweep_cap_sweep == 2
        @test evolved_first_row.transient_cap_cycle == 1
        @test evolved_first_row.transient_cap_sweep === nothing
    finally
        rm(both_caps_path; force=true)
    end
end
