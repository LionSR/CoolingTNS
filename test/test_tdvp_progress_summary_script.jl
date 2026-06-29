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
                            elapsed_seconds,
                            columns=TDVPProgressCSVSummary.LARGE_N_PROGRESS_CSV_COLUMNS)
    row = Dict(
        "timestamp" => timestamp,
        "N" => "4",
        "method" => method,
        "evolution" => evolution,
        "R" => R,
        "trajectory" => "1",
        "seed" => "123",
        "Dmax" => Dmax,
        "cutoff" => "1.0e-6",
        "g" => g,
        "tau" => "0.2",
        "stage" => stage,
        "step" => string(step),
        "cycle" => string(cycle),
        "delta" => delta,
        "te" => te,
        "energy_per_site" => energy_per_site,
        "relative_energy" => relative_energy,
        "overlap" => overlap,
        TDVPProgressCSVSummary.LARGE_N_SYSTEM_MAX_BOND_KEY => system_max_bond,
        TDVPProgressCSVSummary.LARGE_N_SYSTEM_MEAN_BOND_KEY => system_mean_bond,
        TDVPProgressCSVSummary.LARGE_N_EVOLVED_MAX_BOND_KEY => evolved_max_bond,
        TDVPProgressCSVSummary.LARGE_N_EVOLVED_MEAN_BOND_KEY => evolved_mean_bond,
        "tdvp_sweep" => tdvp_sweep,
        "tdvp_time" => tdvp_time,
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
    @test TDVPProgressCSVSummary.largeN_method_kind_from_name("mcwf") === :mcwf
    @test TDVPProgressCSVSummary.largeN_method_kind_from_name("MPO") === :mpo
    @test TDVPProgressCSVSummary.default_progress_cap("mcwf", 7) ==
          tn_method_maxdim(MonteCarloWavefunction(), 7)
    @test TDVPProgressCSVSummary.default_progress_cap("mpo", 7) ==
          tn_method_maxdim(DensityMatrix(), 7)
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
    @test TDVPProgressCSVSummary.progress_detuning_coverage_status(5, 5, 5) ==
          TDVPProgressCSVSummary.LARGE_N_DETUNING_COVERAGE_FULL_GRID
    @test TDVPProgressCSVSummary.LARGE_N_PROGRESS_GROUP_COLUMNS == (
        "N",
        "method",
        "evolution",
        "R",
        "trajectory",
        "seed",
        "Dmax",
        "cutoff",
        "g",
        "tau",
    )
    fixed_identity_a = Dict(
        "N" => "64",
        "method" => "mcwf",
        "evolution" => "continuous",
        "R" => "5",
        "trajectory" => "1",
        "seed" => "84310618",
        "Dmax" => "96",
        "cutoff" => "1.0e-6",
        "g" => "0.05",
        "tau" => "0.2",
        "te" => "0.34",
    )
    fixed_identity_b = merge(fixed_identity_a, Dict("te" => "1.86"))
    @test TDVPProgressCSVSummary.group_key(fixed_identity_a) ==
          TDVPProgressCSVSummary.group_key(fixed_identity_b)
    fixed_identity_c = merge(fixed_identity_a, Dict("g" => "0.1"))
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
    err = try
        TDVPProgressCSVSummary.default_progress_cap("ed", 7)
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    @test occursin("pass --cap D explicitly", sprint(showerror, err))

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
        @test count(==('|'), summary_lines[1]) - 1 == 25
        @test occursin("| file | N | method | evolution | R | g | traj |", output)
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
    finally
        rm(path; force=true)
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
            col -> col != "g",
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
