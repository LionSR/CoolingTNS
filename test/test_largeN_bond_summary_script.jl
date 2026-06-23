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

function write_minimal_mode_summary_file(path::AbstractString, mode_hk, mode_nk;
                                         mode_ek_values=nothing,
                                         mode_measurement_cycles=[0],
                                         energy_values=nothing,
                                         energy_dataset_name="E_mean")
    N = 4
    J, h = 1.0, 0.5
    k_indices = CoolingTNS.allowed_k_indices(N, -1)
    n_rows = mode_hk isa AbstractMatrix ? size(mode_hk, 1) : 1
    energy = energy_values === nothing ? zeros(Float64, n_rows) : Float64.(energy_values)
    length(energy) == n_rows || error("test helper energy_values length must match mode_hk rows")
    h5open(path, "w") do f
        write(f, "Dmax", 8)
        write(f, "model", "ising")
        write(f, "bc", "periodic")
        write(f, "J", J)
        write(f, "h", h)
        gn = create_group(f, "N4")
        write(gn, "N", N)
        gm = create_group(gn, "mcwf")
        gr = create_group(gm, "R1")

        write(gr, "M", 1)
        write(gr, energy_dataset_name, energy)
        write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, zeros(Float64, n_rows))
        write(gr, "system_max_bond", ones(Int, n_rows))
        write(gr, "system_mean_bond", ones(Float64, n_rows))
        write(gr, "evolved_max_bond", n_rows == 1 ? [0] : vcat(0, ones(Int, n_rows - 1)))
        write(gr, "evolved_mean_bond", n_rows == 1 ? [NaN] : vcat(NaN, ones(Float64, n_rows - 1)))
        write(gr, CoolingTNS.RESULT_MODE_HK, mode_hk)
        write(gr, CoolingTNS.RESULT_MODE_NK, mode_nk)
        write(gr, CoolingTNS.RESULT_MODE_K_INDICES, Float64.(k_indices))
        gaps = mode_ek_values === nothing ?
            CoolingTNS.mode_energies_Jh(k_indices, J, h, N) :
            mode_ek_values
        write(gr, CoolingTNS.RESULT_MODE_ENERGIES, gaps)
        write(gr, CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES, mode_measurement_cycles)
        write(gr, CoolingTNS.RESULT_MODE_GF, -1)
        write(gr, CoolingTNS.RESULT_MODE_GF_SOURCE, "state")
    end
    return nothing
end

function write_split_trajectory_summary_file(
    path::AbstractString;
    trajectory::Integer,
    energy_values::AbstractVector{<:Real},
    system_max::AbstractVector{<:Integer},
    evolved_max::AbstractVector{<:Integer},
    completed_steps::Integer,
    stop_reason::AbstractString,
    elapsed_seconds::Real,
    overlap_values=nothing,
    write_e0::Bool=true,
    write_stop_reason::Bool=true,
    write_delta_history::Bool=true,
    te::Real=1.0,
    randomize_times::Bool=false,
    init_state::AbstractString="product",
    theta::Real=0.0,
)
    N = 4
    E0 = -4.0
    length(energy_values) == length(system_max) == length(evolved_max) ||
        error("split trajectory test data must have matching history lengths")
    h5open(path, "w") do f
        write(f, "Dmax", 8)
        write(f, "steps", 4)
        write(f, "te", Float64(te))
        write(f, CoolingTNS.RESULT_RANDOMIZE_TIMES, randomize_times)
        write(f, "init_state", String(init_state))
        write(f, "theta", Float64(theta))
        write(f, LARGE_N_EVOLUTION_METHOD_KEY, "continuous")
        write(f, CoolingTNS.RESULT_SCHEDULE, "descending")
        gn = create_group(f, "N4")
        write(gn, "N", N)
        gm = create_group(gn, "mcwf")
        write_e0 && write(gm, "E0", E0)
        write(gm, LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY,
              LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE)
        write(gm, LARGE_N_DETUNING_DELTA_MIN_KEY, 0.5)
        write(gm, LARGE_N_DETUNING_DELTA_MAX_KEY, 3.0)
        write(gm, LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY, NaN)
        gr = create_group(gm, "R2")

        energy = Float64.(energy_values)
        overlap = overlap_values === nothing ?
            fill(0.5, length(energy)) :
            Float64.(overlap_values)
        length(overlap) == length(energy) ||
            error("split trajectory test data must have matching overlap and energy lengths")
        write(gr, "M", 1)
        write(gr, CoolingTNS.RESULT_ENERGY, energy)
        write(gr, CoolingTNS.RESULT_ENERGY_TRAJECTORIES, reshape(energy, :, 1))
        write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY,
              relative_energy.(energy, Ref(E0)))
        write(gr, CoolingTNS.RESULT_GROUND_STATE_OVERLAP, overlap)
        write(gr, CoolingTNS.RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES,
              reshape(overlap, :, 1))
        write(gr, "system_max_bond", Int.(system_max))
        write(gr, "system_mean_bond", Float64.(system_max))
        write(gr, "evolved_max_bond", Int.(evolved_max))
        write(gr, "evolved_mean_bond", Float64.(evolved_max))
        write(gr, LARGE_N_BOND_SATURATION_THRESHOLD_KEY, 8)
        write(gr, LARGE_N_SYSTEM_SATURATION_CYCLE_KEY,
              [first_bond_saturation_cycle(system_max, 8)])
        write(gr, LARGE_N_EVOLVED_SATURATION_CYCLE_KEY,
              [first_bond_saturation_cycle(evolved_max, 8)])
        write(gr, CoolingTNS.RESULT_TRUNCATION_ERROR_HISTORY_STATUS,
              CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED)
        write(gr, LARGE_N_ELAPSED_SECONDS_KEY, [Float64(elapsed_seconds)])
        write(gr, "trajectory_indices", [Int(trajectory)])
        write(gr, "trajectory_seeds", [largeN_trajectory_seed(20260617, N, 2, trajectory)])
        write(gr, CoolingTNS.RESULT_REQUESTED_STEPS, [4])
        write(gr, CoolingTNS.RESULT_COMPLETED_STEPS, [Int(completed_steps)])
        write_stop_reason && write(gr, LARGE_N_STOP_REASONS_KEY, [String(stop_reason)])
        if write_delta_history
            write(gr, "delta_lists", reshape([NaN; 3.0; 0.5; 3.0][1:length(energy)], :, 1))
            write(gr, CoolingTNS.RESULT_DELTA_VALUES, [0.5, 3.0])
        end
    end
    return nothing
end

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
    @test init_protocol_label("product", 0.0) == "product"
    @test init_protocol_label("ground", 0.0) == "ground"
    @test init_protocol_label("theta", 0.25) == "theta=0.250"

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
            counts = distinct_completed_delta_counts(history, [3])
            @test counts == [2]
            @test visited_detunings_label_from_counts(counts, 0) == "n/a"
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
            write(f, "te", 1.25)
            write(f, CoolingTNS.RESULT_RANDOMIZE_TIMES, false)
            write(f, "init_state", "product")
            write(f, "theta", 0.0)
            write(f, LARGE_N_EVOLUTION_METHOD_KEY, "continuous")
            write(f, CoolingTNS.RESULT_SCHEDULE, "descending")
            gn = create_group(f, "N4")
            write(gn, "N", 4)
            gm = create_group(gn, "mcwf")
            write(gm, LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY,
                  LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE)
            write(gm, LARGE_N_DETUNING_DELTA_MIN_KEY, 0.5)
            write(gm, LARGE_N_DETUNING_DELTA_MAX_KEY, 3.0)
            write(gm, LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY, NaN)
            gr = create_group(gm, "R2")

            write(gr, "M", 2)
            write(gr, CoolingTNS.RESULT_ENERGY, [-1.0, 2.0, 4.0])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 1.0, 2.0])
            write(gr, CoolingTNS.RESULT_GROUND_STATE_OVERLAP, [0.75, 0.50, 0.15])
            write(gr, "system_max_bond", [1 1; 6 9; 12 10])
            write(gr, "system_mean_bond", [1.0 1.0; 4.0 5.0; 8.0 5.0])
            write(gr, "evolved_max_bond", [0 0; 12 6; 8 14])
            write(gr, "evolved_mean_bond", [NaN NaN; 8.0 7.0; 9.0 11.0])
            write(gr, LARGE_N_TDVP_SWEEP_MAX_BOND_KEY, [0 0; 6 13; 8 14])
            write(gr, LARGE_N_TDVP_SWEEP_SATURATION_CYCLE_KEY, [0, 1])
            write(
                gr,
                CoolingTNS.RESULT_TRUNCATION_ERROR_HISTORY_STATUS,
                CoolingTNS.TRUNCATION_ERROR_HISTORY_NOT_RECORDED,
            )
            write(gr, LARGE_N_ELAPSED_SECONDS_KEY, [10.0, 15.5])
            write(gr, CoolingTNS.RESULT_REQUESTED_STEPS, [3, 3])
            write(gr, CoolingTNS.RESULT_COMPLETED_STEPS, [2, 2])
            write(gr, LARGE_N_STOP_REASONS_KEY, ["", "bond_cap"])
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
        @test row.te == 1.25
        @test row.randomize_times == false
        @test row.time_protocol == "fixed"
        @test row.init_state == "product"
        @test row.init_protocol == "product"
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
        @test row.delta_protocol == LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE
        @test row.delta_range == "[0.50000000,3.00000000]"
        @test row.delta_factor == "n/a"
        @test row.threshold == 12
        @test row.initial_e_over_n == -0.25
        @test row.initial_relative_energy == 0.0
        @test row.initial_overlap == 0.75
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
            "| file | N | method | evolution | te | time protocol | init | R | M | schedule | completed/requested | completed/requested periods | visited detunings | detuning coverage | elapsed_total | traj cycles/hour | stop_reason | delta_protocol | delta_range | delta_factor | Dcap |",
            output,
        )
        @test length(unique(markdown_column_counts(output))) == 1
        @test occursin("| bond_status | truncation errors | initial E/N |", output)
        @test occursin("| initial E/N | initial relE | initial overlap | final E/N |", output)
        @test occursin("| final E/N | relE | best E/N | best relE | tail E/N |", output)
        @test occursin(
            "| $(basename(path)) | 4 | mcwf | continuous | 1.250 | fixed | product | 2 | 2 | " *
            "descending | 2/3 | 1.00/1.50 | 1-2/2 | full_grid_observed | 25.5 | 564.71 | bond_capx1/2 | fixed_range | " *
            "[0.50000000,3.00000000] | n/a | 12 | >=12 | >=14 | >=14 | " *
            "not_converged_system_and_evolved_and_tdvp_sweep_cap | " *
            "not_recorded | " *
            "-0.25000000 | 0.00000 | 0.75000 | " *
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
            "| file | N | method | evolution | te | time protocol | init | R | M | schedule | completed/requested | completed/requested periods | visited detunings | detuning coverage | initial E/N | initial overlap | final E/N | best E/N | mode max abs dE/N | Dcap |",
            compact_output,
        )
        @test length(unique(markdown_column_counts(compact_output))) == 1
        @test occursin("| bond_status | truncation errors | elapsed_total |", compact_output)
        @test occursin("| elapsed_total | traj cycles/hour | stop_reason |", compact_output)
        @test occursin(
            "| $(basename(path)) | 4 | mcwf | continuous | 1.250 | fixed | product | 2 | 2 | " *
            "descending | 2/3 | 1.00/1.50 | 1-2/2 | full_grid_observed | -0.25000000 | 0.75000 | 1.00000000 | -0.25000000 | n/a | 12 | >=12 | >=14 | >=14 | " *
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

@testset "Large-N summary combines split trajectory-axis files" begin
    path1 = tempname() * ".h5"
    path3 = tempname() * ".h5"
    missing_delta_path = tempname() * ".h5"
    missing_stop_path = tempname() * ".h5"
    legacy_e0_path1 = tempname() * ".h5"
    legacy_e0_path2 = tempname() * ".h5"
    duplicate_path = tempname() * ".h5"
    te_mismatch_path = tempname() * ".h5"
    randomized_time_path = tempname() * ".h5"
    ground_init_path = tempname() * ".h5"
    try
        write_split_trajectory_summary_file(
            path1;
            trajectory=1,
            energy_values=[-1.0, -2.0, -1.0],
            system_max=[1, 4, 8],
            evolved_max=[0, 4, 8],
            completed_steps=2,
            stop_reason="bond_cap",
            elapsed_seconds=10.0,
        )
        write_split_trajectory_summary_file(
            path3;
            trajectory=3,
            energy_values=[-1.0, -1.5, -2.0, -2.5],
            system_max=[1, 3, 5, 6],
            evolved_max=[0, 4, 6, 7],
            completed_steps=3,
            stop_reason="",
            elapsed_seconds=20.0,
        )
        write_split_trajectory_summary_file(
            missing_delta_path;
            trajectory=5,
            energy_values=[-1.0, -1.5, -2.0],
            system_max=[1, 3, 5],
            evolved_max=[0, 4, 6],
            completed_steps=2,
            stop_reason="",
            elapsed_seconds=7.0,
            write_delta_history=false,
        )
        write_split_trajectory_summary_file(
            missing_stop_path;
            trajectory=6,
            energy_values=[-1.0, -1.5, -2.0],
            system_max=[1, 3, 5],
            evolved_max=[0, 4, 6],
            completed_steps=2,
            stop_reason="",
            elapsed_seconds=7.0,
            write_stop_reason=false,
        )
        write_split_trajectory_summary_file(
            legacy_e0_path1;
            trajectory=7,
            energy_values=[-1.0, -1.5, -2.0],
            system_max=[1, 3, 5],
            evolved_max=[0, 4, 6],
            completed_steps=2,
            stop_reason="",
            elapsed_seconds=7.0,
            write_e0=false,
        )
        write_split_trajectory_summary_file(
            legacy_e0_path2;
            trajectory=9,
            energy_values=[-1.0, -1.25, -1.5],
            system_max=[1, 3, 5],
            evolved_max=[0, 4, 6],
            completed_steps=2,
            stop_reason="",
            elapsed_seconds=9.0,
            write_e0=false,
        )
        write_split_trajectory_summary_file(
            duplicate_path;
            trajectory=1,
            energy_values=[-1.0, -1.5],
            system_max=[1, 4],
            evolved_max=[0, 4],
            completed_steps=1,
            stop_reason="",
            elapsed_seconds=5.0,
        )
        write_split_trajectory_summary_file(
            te_mismatch_path;
            trajectory=1,
            energy_values=[-1.0, -1.5],
            system_max=[1, 4],
            evolved_max=[0, 4],
            completed_steps=1,
            stop_reason="",
            elapsed_seconds=5.0,
            te=0.5,
        )
        write_split_trajectory_summary_file(
            randomized_time_path;
            trajectory=1,
            energy_values=[-1.0, -1.5],
            system_max=[1, 4],
            evolved_max=[0, 4],
            completed_steps=1,
            stop_reason="",
            elapsed_seconds=5.0,
            randomize_times=true,
        )
        write_split_trajectory_summary_file(
            ground_init_path;
            trajectory=1,
            energy_values=[-4.0, -3.5],
            system_max=[1, 4],
            evolved_max=[0, 4],
            completed_steps=1,
            stop_reason="",
            elapsed_seconds=5.0,
            overlap_values=[1.0, 0.9],
            init_state="ground",
        )

        rows = vcat(summarize_file(path1), summarize_file(path3))
        @test length(rows) == 2
        combined = combine_trajectory_rows(rows)
        @test length(combined) == 1
        row = only(combined)
        @test row.file == "trajectory_ensemble(traj=1,3)"
        @test row.te == 1.0
        @test row.time_protocol == "fixed"
        @test row.init_protocol == "product"
        @test row.source_files == (basename(path1), basename(path3))
        @test row.trajectory_indices == [1, 3]
        @test row.M == 2
        @test row.completed_requested == "2-3/4"
        @test row.completed_requested_periods == "1.00-1.50/2.00"
        @test row.visited_detunings == "2/2"
        @test row.detuning_coverage == "full_grid_observed"
        @test row.elapsed_total_seconds == 30.0
        @test row.traj_cycles_per_hour ≈ 3600 * 5 / 30
        @test row.stop_reason == "bond_capx1/2"
        @test row.initial_e_over_n ≈ -1.0 / 4
        @test row.initial_relative_energy ≈ relative_energy(-1.0, -4.0)
        @test row.initial_overlap ≈ 0.5
        @test row.final_e_over_n ≈ mean([-1.0 / 4, -2.5 / 4])
        @test row.relative_energy ≈ mean([
            relative_energy(-1.0, -4.0),
            relative_energy(-2.5, -4.0),
        ])
        @test row.best_e_over_n ≈ mean([-2.0 / 4, -2.5 / 4])
        @test row.tail_count == "3-4"
        @test row.system_effective_bond == ">=8"
        @test row.evolved_effective_bond == ">=8"
        @test row.bond_status == "not_converged_system_and_evolved_cap"
        @test row.final_system_max == 8
        @test row.peak_evolved_max == 8
        @test row.system_saturation_cycle == 2
        @test row.evolved_saturation_cycle == 2
        @test ismissing(row.mode_max_abs_err_over_n)

        compact_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    summarize_largeN_bond_dimensions_main([
                        "--compact",
                        "--combine-trajectories",
                        path1,
                        path3,
                    ])
                end
            end
            read(output_path, String)
        end
        @test occursin("trajectory_ensemble(traj=1,3)", compact_output)
        @test occursin("| 4 | mcwf | continuous | 1.000 | fixed | product | 2 | 2 | descending | 2-3/4 |", compact_output)
        @test occursin("| full_grid_observed | -0.25000000 | 0.50000 | -0.43750000 | -0.56250000 | n/a | 8 | >=8 | >=8 | n/a | not_converged_system_and_evolved_cap | not_recorded | 30.0 | 600.00 | bond_capx1/2 |", compact_output)

        @test_throws ErrorException combine_trajectory_rows(
            vcat(rows, summarize_file(duplicate_path))
        )
        split_by_te_rows = combine_trajectory_rows(vcat(
            summarize_file(path1),
            summarize_file(te_mismatch_path),
        ))
        @test length(split_by_te_rows) == 2
        @test sort([row.te for row in split_by_te_rows]) == [0.5, 1.0]
        split_by_time_protocol_rows = combine_trajectory_rows(vcat(
            summarize_file(path1),
            summarize_file(randomized_time_path),
        ))
        @test length(split_by_time_protocol_rows) == 2
        @test sort([row.time_protocol for row in split_by_time_protocol_rows]) ==
              ["fixed", "randomized"]
        split_by_init_rows = combine_trajectory_rows(vcat(
            summarize_file(path1),
            summarize_file(ground_init_path),
        ))
        @test length(split_by_init_rows) == 2
        @test sort([row.init_protocol for row in split_by_init_rows]) ==
              ["ground", "product"]
        mixed_delta_history_row = only(combine_trajectory_rows(vcat(
            summarize_file(path1),
            summarize_file(missing_delta_path),
        )))
        @test mixed_delta_history_row.M == 2
        @test mixed_delta_history_row.visited_delta_counts == [2]
        @test mixed_delta_history_row.missing_delta_history_count == 1
        @test mixed_delta_history_row.visited_detunings == "2/2+unknownx1/2"
        mixed_stop_reason_row = only(combine_trajectory_rows(vcat(
            summarize_file(path1),
            summarize_file(missing_stop_path),
        )))
        @test mixed_stop_reason_row.M == 2
        @test mixed_stop_reason_row.stop_reason_values == ["bond_cap", ""]
        @test mixed_stop_reason_row.stop_reason == "bond_capx1/2"
        legacy_e0_row = only(combine_trajectory_rows(vcat(
            summarize_file(legacy_e0_path1),
            summarize_file(legacy_e0_path2),
        )))
        @test legacy_e0_row.file == "trajectory_ensemble(traj=7,9)"
        @test isnan(legacy_e0_row.E0)
        @test legacy_e0_row.M == 2
        @test parse_args(["--combine-trajectories", path1]).combine_trajectories
    finally
        rm(path1; force=true)
        rm(path3; force=true)
        rm(missing_delta_path; force=true)
        rm(missing_stop_path; force=true)
        rm(legacy_e0_path1; force=true)
        rm(legacy_e0_path2; force=true)
        rm(duplicate_path; force=true)
        rm(te_mismatch_path; force=true)
        rm(randomized_time_path; force=true)
        rm(ground_init_path; force=true)
    end
end

@testset "Large-N summary validates trajectory metadata fallbacks" begin
    function write_legacy_split_metadata_file(path; trajectory_indices=nothing,
                                             energy_trajectories=nothing,
                                             stop_reasons=nothing)
        h5open(path, "w") do f
            write(f, "Dmax", 4)
            write(f, "steps", 1)
            gn = create_group(f, "N2")
            write(gn, "N", 2)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 2)
            write(gr, CoolingTNS.RESULT_ENERGY, [-2.0, -1.0])
            write(gr, CoolingTNS.RESULT_RELATIVE_ENERGY, [0.0, 0.5])
            write(gr, "system_max_bond", [1 1; 2 3])
            write(gr, "system_mean_bond", [1.0 1.0; 2.0 3.0])
            write(gr, "evolved_max_bond", [0 0; 3 4])
            write(gr, "evolved_mean_bond", [NaN NaN; 3.0 4.0])
            trajectory_indices === nothing ||
                write(gr, "trajectory_indices", Int.(trajectory_indices))
            energy_trajectories === nothing ||
                write(gr, CoolingTNS.RESULT_ENERGY_TRAJECTORIES,
                      Float64.(energy_trajectories))
            stop_reasons === nothing ||
                write(gr, LARGE_N_STOP_REASONS_KEY, String.(stop_reasons))
        end
    end

    fallback_path = tempname() * ".h5"
    bad_indices_path = tempname() * ".h5"
    bad_energy_columns_path = tempname() * ".h5"
    bad_energy_rows_path = tempname() * ".h5"
    bad_stop_reasons_path = tempname() * ".h5"
    try
        write_legacy_split_metadata_file(fallback_path)
        fallback_row = only(summarize_file(fallback_path))
        @test fallback_row.trajectory_indices == [1, 2]
        @test fallback_row.stop_reason_values == ["", ""]
        @test fallback_row.stop_reason == "none"
        @test fallback_row.final_e_over_n_values == [-0.5, -0.5]
        @test fallback_row.best_e_over_n_values == [-1.0, -1.0]
        @test fallback_row.tail_e_over_n_values == [-0.75, -0.75]
        @test isnan(fallback_row.initial_overlap)
        @test all(isnan, fallback_row.initial_overlap_values)

        write_legacy_split_metadata_file(bad_indices_path; trajectory_indices=[1])
        @test_throws ErrorException summarize_file(bad_indices_path)

        write_legacy_split_metadata_file(
            bad_energy_columns_path;
            energy_trajectories=reshape([-2.0, -1.0], :, 1),
        )
        @test_throws ErrorException summarize_file(bad_energy_columns_path)

        write_legacy_split_metadata_file(
            bad_energy_rows_path;
            energy_trajectories=reshape([-2.0, -1.0], 1, 2),
        )
        @test_throws ErrorException summarize_file(bad_energy_rows_path)

        write_legacy_split_metadata_file(
            bad_stop_reasons_path; stop_reasons=["bond_cap"]
        )
        @test_throws ErrorException summarize_file(bad_stop_reasons_path)
    finally
        rm(fallback_path; force=true)
        rm(bad_indices_path; force=true)
        rm(bad_energy_columns_path; force=true)
        rm(bad_energy_rows_path; force=true)
        rm(bad_stop_reasons_path; force=true)
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
        @test isnan(row.te)
        @test row.time_protocol == "fixed"
        @test row.init_protocol == "unknown"
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
            "| $(basename(path)) | 2 | mcwf | unknown | NaN | fixed | unknown | 1 | 1 | " *
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
            write(f, LARGE_N_EVOLUTION_METHOD_KEY, "continuous")
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

@testset "Large-N summary validates stored mode occupations" begin
    path = tempname() * ".h5"
    try
        mode_hk = reshape([-1.0, -0.5, 0.0, 1.0], 1, 4)
        bad_mode_nk = CoolingTNS.mode_occupation_from_hk(mode_hk)
        bad_mode_nk[1, 2] += 0.1
        write_minimal_mode_summary_file(path, mode_hk, bad_mode_nk)

        err = try
            summarize_file(path)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin(CoolingTNS.RESULT_MODE_NK, message)
        @test occursin(CoolingTNS.RESULT_MODE_HK, message)
        @test occursin("derived occupation", message)
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary rejects mode occupation shape mismatches" begin
    path = tempname() * ".h5"
    try
        mode_hk = reshape([-1.0, -0.5, 0.0, 1.0], 1, 4)
        bad_shape_nk = vec(CoolingTNS.mode_occupation_from_hk(mode_hk))
        write_minimal_mode_summary_file(path, mode_hk, bad_shape_nk)

        err = try
            summarize_file(path)
            nothing
        catch err
            err
        end
        @test err isa DimensionMismatch
        message = sprint(showerror, err)
        @test occursin(CoolingTNS.RESULT_MODE_NK, message)
        @test occursin(CoolingTNS.RESULT_MODE_HK, message)
        @test occursin("shape", message)
    finally
        rm(path; force=true)
    end

    hk_path = tempname() * ".h5"
    try
        mode_hk = [-1.0, -0.5, 0.0, 1.0]
        write_minimal_mode_summary_file(
            hk_path,
            mode_hk,
            CoolingTNS.mode_occupation_from_hk(mode_hk),
        )

        err = try
            summarize_file(hk_path)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin(CoolingTNS.RESULT_MODE_HK, message)
        @test occursin("steps-by-modes matrix", message)
    finally
        rm(hk_path; force=true)
    end
end

@testset "Large-N summary validates mode measurement cycles" begin
    scalar_path = tempname() * ".h5"
    try
        mode_hk = reshape([-1.0, -0.5, 0.0, 1.0], 1, 4)
        write_minimal_mode_summary_file(
            scalar_path,
            mode_hk,
            CoolingTNS.mode_occupation_from_hk(mode_hk);
            mode_measurement_cycles=0,
        )
        row = only(summarize_file(scalar_path))
        @test row.mode_measured_rows == "1/1"
    finally
        rm(scalar_path; force=true)
    end

    for (cycles, expected) in [
        ([1, 0], "sorted"),
        ([0, 0], "unique"),
        ([0, 2], "0:1"),
    ]
        path = tempname() * ".h5"
        try
            mode_hk = [
                -1.0 -0.5 0.0 1.0
                 0.0  0.5 0.5 0.0
            ]
            write_minimal_mode_summary_file(
                path,
                mode_hk,
                CoolingTNS.mode_occupation_from_hk(mode_hk);
                mode_measurement_cycles=cycles,
            )

            err = try
                summarize_file(path)
                nothing
            catch err
                err
            end
            @test err isa ArgumentError
            message = sprint(showerror, err)
            @test occursin(CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES, message)
            @test occursin(expected, message)
        finally
            rm(path; force=true)
        end
    end

    path = tempname() * ".h5"
    try
        mode_hk = [
            -1.0 -0.5 0.0 1.0
             NaN  NaN NaN NaN
        ]
        write_minimal_mode_summary_file(
            path,
            mode_hk,
            CoolingTNS.mode_occupation_from_hk(mode_hk);
            mode_measurement_cycles=[0, 1],
        )

        err = try
            summarize_file(path)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin(CoolingTNS.RESULT_MODE_HK, message)
        @test occursin("non-finite", message)
        @test occursin("measured cycle 1", message)
    finally
        rm(path; force=true)
    end

    energy_path = tempname() * ".h5"
    try
        mode_hk = reshape([-1.0, -0.5, 0.0, 1.0], 1, 4)
        write_minimal_mode_summary_file(
            energy_path,
            mode_hk,
            CoolingTNS.mode_occupation_from_hk(mode_hk);
            energy_values=[NaN],
        )

        err = try
            summarize_file(energy_path)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin("E_mean", message)
        @test occursin("non-finite", message)
    finally
        rm(energy_path; force=true)
    end

    canonical_energy_path = tempname() * ".h5"
    try
        mode_hk = reshape([-1.0, -0.5, 0.0, 1.0], 1, 4)
        write_minimal_mode_summary_file(
            canonical_energy_path,
            mode_hk,
            CoolingTNS.mode_occupation_from_hk(mode_hk);
            energy_values=[NaN],
            energy_dataset_name=CoolingTNS.RESULT_ENERGY,
        )

        err = try
            summarize_file(canonical_energy_path)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin(CoolingTNS.RESULT_ENERGY, message)
        @test !occursin("E_mean", message)
        @test occursin("non-finite", message)
    finally
        rm(canonical_energy_path; force=true)
    end
end

@testset "Large-N summary validates stored positive mode gaps" begin
    path = tempname() * ".h5"
    try
        N = 4
        J, h = 1.0, 0.5
        k_indices = CoolingTNS.allowed_k_indices(N, -1)
        mode_hk = reshape([-1.0, -0.5, 0.0, 1.0], 1, 4)
        bad_gaps = CoolingTNS.mode_energies_Jh(k_indices, J, h, N)
        bad_gaps[2] += 0.1
        write_minimal_mode_summary_file(
            path,
            mode_hk,
            CoolingTNS.mode_occupation_from_hk(mode_hk);
            mode_ek_values=bad_gaps,
        )

        err = try
            summarize_file(path)
            nothing
        catch err
            err
        end
        @test err isa ArgumentError
        message = sprint(showerror, err)
        @test occursin(CoolingTNS.RESULT_MODE_ENERGIES, message)
        @test occursin(CoolingTNS.RESULT_MODE_K_INDICES, message)
        @test occursin("mode_energies_Jh", message)
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary rejects positive mode gap shape mismatches" begin
    path = tempname() * ".h5"
    try
        N = 4
        J, h = 1.0, 0.5
        k_indices = CoolingTNS.allowed_k_indices(N, -1)
        mode_hk = reshape([-1.0, -0.5, 0.0, 1.0], 1, 4)
        bad_shape_gaps = CoolingTNS.mode_energies_Jh(k_indices, J, h, N)[1:end-1]
        write_minimal_mode_summary_file(
            path,
            mode_hk,
            CoolingTNS.mode_occupation_from_hk(mode_hk);
            mode_ek_values=bad_shape_gaps,
        )

        err = try
            summarize_file(path)
            nothing
        catch err
            err
        end
        @test err isa DimensionMismatch
        message = sprint(showerror, err)
        @test occursin(CoolingTNS.RESULT_MODE_ENERGIES, message)
        @test occursin(CoolingTNS.RESULT_MODE_K_INDICES, message)
        @test occursin("length", message)
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
            write(f, LARGE_N_EVOLUTION_METHOD_KEY, "continuous")
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
            write(gr, LARGE_N_TDVP_SWEEP_MAX_BOND_KEY, [0, 0])
            write(gr, LARGE_N_TDVP_SWEEP_SATURATION_CYCLE_KEY, [0])
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
            write(gm, LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY,
                  LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE)
            write(gm, LARGE_N_DETUNING_DELTA_MIN_KEY, 0.75)
            write(gm, LARGE_N_DETUNING_DELTA_MAX_KEY, 3.0)
            write(gm, LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY, 4.0)
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
        @test row.delta_protocol == LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE
        @test row.delta_range == "[0.75000000,3.00000000]"
        @test row.delta_factor == "4.000"
    finally
        rm(path; force=true)
    end
end
