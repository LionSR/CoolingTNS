using Test
using HDF5
using CoolingTNS

include(joinpath(@__DIR__, "..", "scripts", "validation",
                 "summarize_largeN_bond_dimensions.jl"))

markdown_column_counts(text::AbstractString) = [
    count(==('|'), line) - 1
    for line in split(text, '\n')
    if startswith(line, "|")
]

@testset "Large-N schedule-period labels" begin
    @test is_deterministic_schedule("round_robin")
    @test is_deterministic_schedule("descending")
    @test !is_deterministic_schedule("random")
    @test completed_requested_periods_label([5], [12], 10, "descending") == "0.50/1.20"
    @test completed_requested_periods_label([3, 5], [8, 8], 5, "round_robin") == "0.60-1.00/1.60"
    @test completed_requested_periods_label([5], [12], 10, "random") == "n/a"
    @test completed_requested_periods_label([5], [12], 10, "unknown") == "n/a"
    @test completed_requested_periods_label([5], [12], 0, "descending") == "n/a"
    @test detuning_coverage_status([10], [12], 10, "descending") ==
          "full_grid_observed"
    @test detuning_coverage_status([5], [12], 10, "descending") ==
          "stopped_partial_grid"
    @test detuning_coverage_status([2], [2], 5, "round_robin") ==
          "requested_partial_grid"
    @test detuning_coverage_status([5], [12], 10, "random") == "n/a"
    @test detuning_coverage_status(Int[], [12], 10, "descending") == "n/a"
    @test detuning_coverage_status([5], Int[], 10, "descending") == "n/a"
    @test detuning_coverage_status([5], [12], 0, "descending") == "n/a"
    @test detuning_coverage_status([3], [8], 1, "descending") == "single_detuning"

    delta_history = [
        NaN NaN
        1.0 2.0
        2.0 2.0
        1.0 3.0
        3.0 3.0
    ]
    @test delta_history_matrix_from_values(2.0) == reshape([2.0], 1, 1)
    @test distinct_completed_delta_counts(delta_history, [3, 4]) == [2, 2]
    @test_throws ErrorException distinct_completed_delta_counts(delta_history, [3, 4, 2])

    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, CoolingTNS.RESULT_DELTA_LIST, [NaN, 1.0, 2.0, 1.0, 3.0])
            history = delta_history_matrix(f)
            @test size(history) == (5, 1)
            @test distinct_completed_delta_counts(history, [3]) == [2]
            @test visited_detunings_label(f, [3], 0) == "n/a"
        end
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary flags partial deterministic detuning coverage" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 8)
            write(f, CoolingTNS.RESULT_SCHEDULE, "descending")
            gn = create_group(f, "N4")
            write(gn, "N", 4)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R5")

            write(gr, "M", 1)
            write(gr, CoolingTNS.RESULT_ENERGY, [-1.0, 0.0, 1.0])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 1.0, 2.0])
            write(gr, "system_max_bond", [1, 2, 3])
            write(gr, "system_mean_bond", [1.0, 2.0, 3.0])
            write(gr, "evolved_max_bond", [0, 4, 8])
            write(gr, "evolved_mean_bond", [NaN, 4.0, 8.0])
            write(gr, CoolingTNS.RESULT_REQUESTED_STEPS, [8])
            write(gr, CoolingTNS.RESULT_COMPLETED_STEPS, [2])
            write(gr, CoolingTNS.RESULT_DELTA_LIST, [NaN, 5.0, 4.0])
        end

        row = only(summarize_file(path))
        @test row.R == 5
        @test row.completed_requested == "2/8"
        @test row.completed_requested_periods == "0.40/1.60"
        @test row.visited_detunings == "2/5"
        @test row.detuning_coverage == "stopped_partial_grid"
    finally
        rm(path; force=true)
    end
end

@testset "Large-N bond-dimension summary script" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 12)
            write(f, "evolution_method", "continuous")
            write(f, CoolingTNS.RESULT_SCHEDULE, "descending")
            gn = create_group(f, "N4")
            write(gn, "N", 4)
            gm = create_group(gn, "mcwf")
            write(gm, "detuning_protocol_source", "fixed_range")
            write(gm, "detuning_delta_min", 0.5)
            write(gm, "detuning_delta_max", 3.0)
            write(gm, "detuning_delta_max_factor", NaN)
            gr = create_group(gm, "R2")

            write(gr, "M", 2)
            write(gr, CoolingTNS.RESULT_ENERGY, [-1.0, 2.0, 4.0])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 1.0, 2.0])
            write(gr, "system_max_bond", [1 1; 6 9; 12 10])
            write(gr, "system_mean_bond", [1.0 1.0; 4.0 5.0; 8.0 5.0])
            write(gr, "evolved_max_bond", [0 0; 12 6; 8 14])
            write(gr, "evolved_mean_bond", [NaN NaN; 8.0 7.0; 9.0 11.0])
            write(gr, "tdvp_sweep_max_bond", [0 0; 6 13; 8 14])
            write(gr, "tdvp_sweep_saturation_cycle", [0, 1])
            write(
                gr,
                CoolingTNS.RESULT_TRUNCATION_ERROR_HISTORY_STATUS,
                CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED,
            )
            write(gr, "elapsed_seconds", [10.0, 15.5])
            write(gr, CoolingTNS.RESULT_REQUESTED_STEPS, [3, 3])
            write(gr, CoolingTNS.RESULT_COMPLETED_STEPS, [2, 2])
            write(gr, "stop_reasons", ["", "bond_cap"])
            write(gr, CoolingTNS.RESULT_DELTA_VALUES, [0.5, 3.0])
            write(gr, "delta_lists", [NaN NaN; 3.0 3.0; 0.5 3.0])

            bd = create_group(gr, "final_bond_dims")
            write(bd, "trajectory_1", [4, 8, 12])
            write(bd, "trajectory_2", [12, 12, 12])
        end

        rows = summarize_file(path)
        @test length(rows) == 1
        row = only(rows)
        @test row.N == 4
        @test row.method == "mcwf"
        @test row.evolution == "continuous"
        @test row.R == 2
        @test row.M == 2
        @test row.schedule == "descending"
        @test row.completed_requested == "2/3"
        @test row.completed_requested_periods == "1.00/1.50"
        @test row.visited_detunings == "1-2/2"
        @test row.detuning_coverage == "full_grid_observed"
        @test row.elapsed_total_seconds == 25.5
        @test row.traj_cycles_per_hour ≈ 3600 * 4 / 25.5
        @test row.stop_reason == "bond_capx1/2"
        @test row.delta_protocol == "fixed_range"
        @test row.delta_range == "[0.50000000,3.00000000]"
        @test row.delta_factor == "n/a"
        @test row.threshold == 12
        @test row.final_e_over_n == 1.0
        @test row.relative_energy == 2.0
        @test row.best_e_over_n == -0.25
        @test row.best_relative_energy == 0.0
        @test row.tail_e_over_n ≈ 5 / 12
        @test row.tail_relative_energy == 1.0
        @test row.tail_count == 3
        @test row.system_effective_bond == ">=12"
        @test row.evolved_effective_bond == ">=14"
        @test row.tdvp_sweep_effective_bond == ">=14"
        @test row.truncation_error_history_status ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED
        @test row.bond_status ==
              "not_converged_system_and_evolved_and_tdvp_sweep_cap"
        @test row.final_system_max == 12
        @test row.final_system_mean == 6.5
        @test row.peak_evolved_max == 14
        @test row.peak_evolved_mean == 10.0
        @test row.peak_tdvp_sweep_max == 14
        @test row.system_saturation_cycle == 2
        @test row.evolved_saturation_cycle == 1
        @test row.tdvp_sweep_saturation_cycle == 1
        @test row.q50 == 10.0
        @test row.q75 == 11.0
        @test row.q90 ≈ 11.6
        @test row.frac50 ≈ 5 / 6
        @test row.frac75 ≈ 2 / 3
        @test row.frac90 ≈ 2 / 3

        output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_markdown(rows)
                end
            end
            read(output_path, String)
        end
        @test occursin(
            "| file | N | method | evolution | R | M | schedule | completed/requested | completed/requested periods | visited detunings | detuning coverage | elapsed_total | traj cycles/hour | stop_reason | delta_protocol | delta_range | delta_factor | Dcap |",
            output,
        )
        @test length(unique(markdown_column_counts(output))) == 1
        @test occursin("| bond_status | truncation errors | final E/N |", output)
        @test occursin("| final E/N | relE | best E/N | best relE | tail E/N |", output)
        @test occursin(
            "| $(basename(path)) | 4 | mcwf | continuous | 2 | 2 | " *
            "descending | 2/3 | 1.00/1.50 | 1-2/2 | full_grid_observed | 25.5 | 564.71 | bond_capx1/2 | fixed_range | " *
            "[0.50000000,3.00000000] | n/a | 12 | >=12 | >=14 | >=14 | " *
            "not_converged_system_and_evolved_and_tdvp_sweep_cap | " *
            "not_recorded | " *
            "1.00000000 | 2.00000 | " *
            "-0.25000000 | 0.00000 | 0.41666667 | 1.00000 | 3 |",
            output,
        )
        @test occursin("| 2 | 2 | descending | 2/3 | 1.00/1.50 | 1-2/2 | full_grid_observed | 25.5 | 564.71 | bond_capx1/2 | fixed_range |", output)

        compact_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_compact_markdown(rows)
                end
            end
            read(output_path, String)
        end
        @test occursin(
            "| file | N | method | evolution | R | M | schedule | completed/requested | completed/requested periods | visited detunings | detuning coverage | final E/N | best E/N | mode max abs dE/N | Dcap |",
            compact_output,
        )
        @test length(unique(markdown_column_counts(compact_output))) == 1
        @test occursin("| bond_status | truncation errors | elapsed_total |", compact_output)
        @test occursin("| elapsed_total | traj cycles/hour | stop_reason |", compact_output)
        @test occursin(
            "| $(basename(path)) | 4 | mcwf | continuous | 2 | 2 | " *
            "descending | 2/3 | 1.00/1.50 | 1-2/2 | full_grid_observed | 1.00000000 | -0.25000000 | n/a | 12 | >=12 | >=14 | >=14 | " *
            "not_converged_system_and_evolved_and_tdvp_sweep_cap | not_recorded | 25.5 | 564.71 | bond_capx1/2 |",
            compact_output,
        )
        @test parse_args(["--compact", path]).compact
        @test parse_args(["--compact", path]).paths == [path]
        @test_throws ArgumentError parse_args(["--unknown", path])
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary handles missing detuning metadata" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 4)
            gn = create_group(f, "N2")
            write(gn, "N", 2)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [-1.0, 0.0])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 1.0])
            write(gr, "system_max_bond", [1, 2])
            write(gr, "system_mean_bond", [1.0, 2.0])
            write(gr, "evolved_max_bond", [0, 4])
            write(gr, "evolved_mean_bond", [NaN, 4.0])
        end

        row = only(summarize_file(path))
        @test row.evolution == "unknown"
        @test row.schedule == "unknown"
        @test row.completed_requested == "1/1"
        @test row.completed_requested_periods == "n/a"
        @test row.visited_detunings == "n/a"
        @test row.detuning_coverage == "n/a"
        @test isnan(row.elapsed_total_seconds)
        @test isnan(row.traj_cycles_per_hour)
        @test row.stop_reason == "none"
        @test row.delta_protocol == "unknown"
        @test row.delta_range == "unknown"
        @test row.delta_factor == "unknown"
        @test row.tdvp_sweep_effective_bond == "n/a"
        @test ismissing(row.peak_tdvp_sweep_max)
        @test ismissing(row.tdvp_sweep_saturation_cycle)
        @test row.truncation_error_history_status ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_LEGACY_MISSING
        @test row.bond_status == "not_converged_evolved_cap"

        output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_markdown([row])
                end
            end
            read(output_path, String)
        end
        @test occursin(
            "| $(basename(path)) | 2 | mcwf | unknown | 1 | 1 | " *
            "unknown | 1/1 | n/a | n/a | n/a | NaN | NaN | none | unknown | " *
            "unknown | unknown | 4 | 2 | >=4 | n/a | not_converged_evolved_cap | legacy_missing |",
            output,
        )
        @test occursin("| 1 | 1 | unknown | 1/1 | n/a | n/a | n/a | NaN | NaN | none | unknown |", output)
        @test occursin("| 4.00 | n/a | none | 1 | n/a |", output)
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary detects measured truncation-error histories" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 8)
            gn = create_group(f, "N2")
            write(gn, "N", 2)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, CoolingTNS.RESULT_ENERGY, [-1.0, -0.5])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 0.5])
            write(gr, "system_max_bond", [1, 2])
            write(gr, "system_mean_bond", [1.0, 2.0])
            write(gr, "evolved_max_bond", [0, 3])
            write(gr, "evolved_mean_bond", [NaN, 3.0])
            write(gr, CoolingTNS.RESULT_TRUNCATION_ERRORS, [0.0, 1e-8])
        end

        row = only(summarize_file(path))
        @test row.truncation_error_history_status ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_MEASURED

        compact_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_compact_markdown([row])
                end
            end
            read(output_path, String)
        end
        @test occursin("| no_cap_hit | measured |", compact_output)
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary distinguishes empty truncation-error histories" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 8)
            gn = create_group(f, "N2")
            write(gn, "N", 2)
            gm = create_group(gn, "mcwf")

            gr_empty = create_group(gm, "R1")
            write(gr_empty, "M", 1)
            write(gr_empty, CoolingTNS.RESULT_ENERGY, [-1.0, -0.5])
            write(gr_empty, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 0.5])
            write(gr_empty, "system_max_bond", [1, 2])
            write(gr_empty, "system_mean_bond", [1.0, 2.0])
            write(gr_empty, "evolved_max_bond", [0, 3])
            write(gr_empty, "evolved_mean_bond", [NaN, 3.0])
            write(gr_empty, CoolingTNS.RESULT_TRUNCATION_ERRORS, Float64[])

            gr_explicit = create_group(gm, "R2")
            write(gr_explicit, "M", 1)
            write(gr_explicit, CoolingTNS.RESULT_ENERGY, [-1.0, -0.25])
            write(gr_explicit, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 0.75])
            write(gr_explicit, "system_max_bond", [1, 2])
            write(gr_explicit, "system_mean_bond", [1.0, 2.0])
            write(gr_explicit, "evolved_max_bond", [0, 3])
            write(gr_explicit, "evolved_mean_bond", [NaN, 3.0])
            write(gr_explicit, CoolingTNS.RESULT_TRUNCATION_ERRORS, Float64[])
            write(
                gr_explicit,
                CoolingTNS.RESULT_TRUNCATION_ERROR_HISTORY_STATUS,
                CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED,
            )
        end

        rows = sort(summarize_file(path); by=row -> row.R)
        @test length(rows) == 2
        @test rows[1].truncation_error_history_status ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_EMPTY
        @test rows[2].truncation_error_history_status ==
              CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED

        compact_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_compact_markdown(rows)
                end
            end
            read(output_path, String)
        end
        @test occursin("| no_cap_hit | empty |", compact_output)
        @test occursin("| no_cap_hit | not_recorded |", compact_output)
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary reports mode energy reconstruction" begin
    path = tempname() * ".h5"
    try
        N = 4
        J, h = 1.0, 0.5
        ham_params = CoolingTNS.IsingParameters(N, J, h, :periodic)
        k_indices = CoolingTNS.allowed_k_indices(N, -1)
        mode_hk = Matrix{Float64}(undef, 3, length(k_indices))
        mode_hk[1, :] .= -1.0
        mode_hk[2, :] .= NaN
        mode_hk[3, :] .= -0.2
        mode_energy = CoolingTNS.ising_energy_from_mode_hk(
            k_indices, mode_hk[[1, 3], :], ham_params
        )
        final_energy_offset = 0.04

        h5open(path, "w") do f
            write(f, "Dmax", 8)
            write(f, "model", "ising")
            write(f, "bc", "periodic")
            write(f, "J", J)
            write(f, "h", h)
            write(f, "evolution_method", "continuous")
            gn = create_group(f, "N4")
            write(gn, "N", N)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [mode_energy[1], 100.0, mode_energy[2] + final_energy_offset])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 1.0, 2.0])
            write(gr, "system_max_bond", [1, 2, 3])
            write(gr, "system_mean_bond", [1.0, 2.0, 3.0])
            write(gr, "evolved_max_bond", [0, 3, 4])
            write(gr, "evolved_mean_bond", [NaN, 3.0, 4.0])
            write(gr, CoolingTNS.RESULT_MODE_HK, mode_hk)
            write(gr, CoolingTNS.RESULT_MODE_NK, CoolingTNS.mode_occupation_from_hk(mode_hk))
            write(gr, CoolingTNS.RESULT_MODE_K_INDICES, Float64.(k_indices))
            write(gr, CoolingTNS.RESULT_MODE_ENERGIES,
                  CoolingTNS.mode_energies_Jh(k_indices, J, h, N))
            write(gr, CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES, [0, 2])
            write(gr, CoolingTNS.RESULT_MODE_GF, -1)
            write(gr, CoolingTNS.RESULT_MODE_GF_SOURCE, "state")
        end

        row = only(summarize_file(path))
        @test row.mode_gF == -1
        @test row.mode_gF_source == "state"
        @test row.mode_measured_rows == "2/3"
        @test row.mode_last_measured_e_over_n ≈ mode_energy[2] / N
        @test row.mode_last_measured_abs_err_over_n ≈ final_energy_offset / N
        @test row.mode_max_abs_err_over_n ≈ final_energy_offset / N

        output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_markdown([row])
                end
            end
            read(output_path, String)
        end
        @test occursin(
            "| mode gF | mode source | mode rows | mode last-measured E/N | mode last-measured abs dE/N | mode max abs dE/N |",
            output,
        )
        @test occursin("| -1 | state | 2/3 |", output)

        compact_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_compact_markdown([row])
                end
            end
            read(output_path, String)
        end
        @test occursin("| mode max abs dE/N |", compact_output)
        @test occursin("| $(format_float(final_energy_offset / N, 3)) | 8 |", compact_output)
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary rejects partial mode metadata" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 8)
            write(f, "model", "ising")
            write(f, "bc", "periodic")
            write(f, "J", 1.0)
            write(f, "h", 0.5)
            gn = create_group(f, "N4")
            write(gn, "N", 4)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [0.0])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0])
            write(gr, "system_max_bond", [1])
            write(gr, "system_mean_bond", [1.0])
            write(gr, "evolved_max_bond", [0])
            write(gr, "evolved_mean_bond", [NaN])
            write(gr, CoolingTNS.RESULT_MODE_HK, reshape(fill(-1.0, 4), 1, 4))
        end

        err = try
            summarize_file(path)
            nothing
        catch err
            err
        end
        @test err isa ErrorException
        message = sprint(showerror, err)
        @test occursin("incomplete mode-observable metadata", message)
        @test occursin(CoolingTNS.RESULT_MODE_HK, message)
        @test occursin(CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES, message)
        @test occursin(CoolingTNS.RESULT_MODE_GF_SOURCE, message)
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary rejects nonintegrable mode metadata" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 8)
            write(f, "model", "niising")
            write(f, "bc", "open")
            write(f, "J", 1.0)
            write(f, "h", NaN)
            gn = create_group(f, "N4")
            write(gn, "N", 4)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [0.0])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0])
            write(gr, "system_max_bond", [1])
            write(gr, "system_mean_bond", [1.0])
            write(gr, "evolved_max_bond", [0])
            write(gr, "evolved_mean_bond", [NaN])
            mode_hk = reshape(fill(-1.0, 4), 1, 4)
            k_indices = Float64.([-1.5, -0.5, 0.5, 1.5])
            write(gr, CoolingTNS.RESULT_MODE_HK, mode_hk)
            write(gr, CoolingTNS.RESULT_MODE_NK, CoolingTNS.mode_occupation_from_hk(mode_hk))
            write(gr, CoolingTNS.RESULT_MODE_K_INDICES, k_indices)
            write(gr, CoolingTNS.RESULT_MODE_ENERGIES, ones(length(k_indices)))
            write(gr, CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES, [0])
            write(gr, CoolingTNS.RESULT_MODE_GF, -1)
            write(gr, CoolingTNS.RESULT_MODE_GF_SOURCE, "state")
        end

        err = try
            summarize_file(path)
            nothing
        catch err
            err
        end
        @test err isa ErrorException
        @test occursin(
            "mode-energy reconstruction is defined here only for the integrable Ising chain",
            sprint(showerror, err),
        )
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary treats all-zero TDVP sweep placeholders as missing" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 4)
            write(f, "evolution_method", "continuous")
            gn = create_group(f, "N2")
            write(gn, "N", 2)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [-1.0, 0.0])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 1.0])
            write(gr, "system_max_bond", [1, 2])
            write(gr, "system_mean_bond", [1.0, 2.0])
            write(gr, "evolved_max_bond", [0, 4])
            write(gr, "evolved_mean_bond", [NaN, 4.0])
            write(gr, "tdvp_sweep_max_bond", [0, 0])
            write(gr, "tdvp_sweep_saturation_cycle", [0])
        end

        row = only(summarize_file(path))
        @test row.tdvp_sweep_effective_bond == "n/a"
        @test ismissing(row.peak_tdvp_sweep_max)
        @test ismissing(row.tdvp_sweep_saturation_cycle)
        @test row.bond_status == "not_converged_evolved_cap"
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary reports gap-scaled detuning metadata" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 4)
            gn = create_group(f, "N2")
            write(gn, "N", 2)
            gm = create_group(gn, "mcwf")
            write(gm, "detuning_protocol_source", "gap_scaled_range")
            write(gm, "detuning_delta_min", 0.75)
            write(gm, "detuning_delta_max", 3.0)
            write(gm, "detuning_delta_max_factor", 4.0)
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [-1.0, 0.0])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 1.0])
            write(gr, "system_max_bond", [1, 2])
            write(gr, "system_mean_bond", [1.0, 2.0])
            write(gr, "evolved_max_bond", [0, 4])
            write(gr, "evolved_mean_bond", [NaN, 4.0])
        end

        row = only(summarize_file(path))
        @test row.delta_protocol == "gap_scaled_range"
        @test row.delta_range == "[0.75000000,3.00000000]"
        @test row.delta_factor == "4.000"
    finally
        rm(path; force=true)
    end
end
