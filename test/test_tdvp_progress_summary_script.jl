using Test

include(joinpath(@__DIR__, "..", "scripts", "validation",
                 "summarize_tdvp_progress_csv.jl"))

const TDVP_PROGRESS_COLUMNS = [
    "timestamp",
    "N",
    "method",
    "evolution",
    "R",
    "trajectory",
    "seed",
    "Dmax",
    "cutoff",
    "tau",
    "stage",
    "step",
    "cycle",
    "delta",
    "te",
    "energy_per_site",
    "relative_energy",
    "overlap",
    "system_max_bond",
    "system_mean_bond",
    "evolved_max_bond",
    "evolved_mean_bond",
    "tdvp_sweep",
    "tdvp_time",
    "elapsed_seconds",
]

function tdvp_progress_line(; timestamp="2026-06-19T00:00:00",
                            method="mcwf", evolution="continuous",
                            R="2", Dmax="6", stage, step, cycle,
                            delta="0.5", te="2.0", energy_per_site="NaN",
                            relative_energy="NaN", overlap="NaN",
                            system_max_bond="1", system_mean_bond="1.0",
                            evolved_max_bond="NaN", evolved_mean_bond="NaN",
                            tdvp_sweep="NaN", tdvp_time="NaN",
                            elapsed_seconds)
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
        "tau" => "0.2",
        "stage" => stage,
        "step" => string(step),
        "cycle" => string(cycle),
        "delta" => delta,
        "te" => te,
        "energy_per_site" => energy_per_site,
        "relative_energy" => relative_energy,
        "overlap" => overlap,
        "system_max_bond" => system_max_bond,
        "system_mean_bond" => system_mean_bond,
        "evolved_max_bond" => evolved_max_bond,
        "evolved_mean_bond" => evolved_mean_bond,
        "tdvp_sweep" => tdvp_sweep,
        "tdvp_time" => tdvp_time,
        "elapsed_seconds" => string(elapsed_seconds),
    )
    return join((row[col] for col in TDVP_PROGRESS_COLUMNS), ",")
end

@testset "TDVP progress CSV summary script" begin
    @test TDVPProgressCSVSummary.parse_csv_line(
        "\"contains,comma\",\"escaped \"\"quote\"\"\",plain"
    ) ==
        ["contains,comma", "escaped \"quote\"", "plain"]
    @test TDVPProgressCSVSummary.default_progress_cap("mcwf", 7) == 7
    @test TDVPProgressCSVSummary.default_progress_cap("mpo", 7) == 28
    @test_throws ArgumentError TDVPProgressCSVSummary.default_progress_cap("ed", 7)

    path = tempname() * ".csv"
    try
        open(path, "w") do io
            println(io, join(TDVP_PROGRESS_COLUMNS, ","))
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
                energy_per_site="0.5", relative_energy="1.5", overlap="0.2",
                system_max_bond="24", evolved_max_bond="24", elapsed_seconds=22.0,
            ))
        end

        rows = TDVPProgressCSVSummary.summarize_progress_file(path)
        @test length(rows) == 2
        row = only(filter(row -> row.method == "mcwf", rows))
        @test row.N == 4
        @test row.R == 2
        @test row.threshold == 6
        @test row.completed_cycles == 2
        @test row.final_energy == 0.25
        @test row.system_effective_bond == ">=6"
        @test row.evolved_effective_bond == ">=7"
        @test row.bond_status == "not_converged_system_and_evolved_cap"
        @test row.system_cap_cycle == 2
        @test row.transient_cap_cycle == 1
        @test row.transient_cap_sweep == 2
        @test row.max_sweep_increment == 5.0
        @test row.max_sweep_cycle == 1
        @test row.max_sweep == 2
        @test row.last_step == 4
        @test row.last_cycle == 3
        @test row.last_stage == "tdvp_sweep"
        @test length(row.updates) == 2

        mpo_row = only(filter(row -> row.method == "mpo", rows))
        @test mpo_row.evolution == "trotter"
        @test mpo_row.R == 3
        @test mpo_row.threshold == 24
        @test mpo_row.completed_cycles == 2
        @test mpo_row.final_energy == 0.5
        @test mpo_row.system_effective_bond == ">=24"
        @test mpo_row.evolved_effective_bond == ">=24"
        @test mpo_row.bond_status == "not_converged_system_and_evolved_cap"
        @test mpo_row.system_cap_cycle == 2
        @test mpo_row.transient_cap_cycle == 2
        @test mpo_row.transient_cap_sweep === nothing
        @test mpo_row.max_sweep_cycle == 0
        @test mpo_row.last_step == 3
        @test mpo_row.last_cycle == 2
        @test mpo_row.last_stage == "updated"

        output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    TDVPProgressCSVSummary.print_markdown(rows)
                end
            end
            read(output_path, String)
        end
        @test occursin("| file | N | method | evolution | R | traj |", output)
        @test occursin("| last step | last cycle | last stage |", output)
        @test occursin("not_converged_system_and_evolved_cap", output)
        @test occursin("| 2 | 1 | 2 | 0.50000000 | 0.25000000 | 6 | 7 | 16.0 |", output)
        @test occursin("| 4 | 3 | tdvp_sweep |", output)
    finally
        rm(path; force=true)
    end
end
