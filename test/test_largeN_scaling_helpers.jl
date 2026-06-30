using Test
using CoolingTNS

include(joinpath(@__DIR__, "..", "scripts", "validation", "largeN_scaling_helpers.jl"))

@testset "Large-N scaling helper functions" begin
    @test LARGE_N_TRAJECTORY_SEED_RULE ==
          "trajectory_seed = base_seed + 1_000_000*N + 10_000*R + trajectory; " *
          "valid for 1 <= R < 100 and 1 <= trajectory < 10000"
    @test LARGE_N_DETUNING_REFERENCE_GAP_SOURCE_KEY == "detuning_reference_gap_source"
    @test LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY == "detuning_protocol_source"
    @test LARGE_N_DETUNING_REFERENCE_GAP_KEY == "detuning_reference_gap"
    @test LARGE_N_DETUNING_DELTA_MIN_KEY == "detuning_delta_min"
    @test LARGE_N_DETUNING_DELTA_MAX_KEY == "detuning_delta_max"
    @test LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY == "detuning_delta_max_factor"
    @test LARGE_N_DETUNING_FIXED_ACROSS_DMAX_KEY == "detuning_fixed_across_dmax"
    @test LARGE_N_DETUNING_REFERENCE_SETUP_GAP == "setup_gap"
    @test LARGE_N_DETUNING_REFERENCE_ISING_MODE_PAIR == "ising_mode_pair_reference"
    @test LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE == "gap_scaled_range"
    @test LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE == "fixed_range"
    @test LARGE_N_LABEL_NA == "n/a"
    @test LARGE_N_LABEL_UNKNOWN == "unknown"
    @test LARGE_N_LABEL_NONE == "none"
    @test LARGE_N_LABEL_MISSING == "missing"
    @test LARGE_N_LABEL_LEGACY_MISSING == "legacy_missing"
    @test LARGE_N_N_GROUP_PREFIX == "N"
    @test LARGE_N_R_GROUP_PREFIX == "R"
    @test LARGE_N_SYSTEM_SIZE_KEY == "N"
    @test LARGE_N_TRAJECTORY_COUNT_KEY == "M"
    @test LARGE_N_GROUND_ENERGY_KEY == "E0"
    @test LARGE_N_REFERENCE_GAP_KEY == "gap"
    @test LARGE_N_ENERGY_STDERR_KEY == "E_stderr"
    @test LARGE_N_PURITY_TRAJECTORIES_KEY == "purity_trajectories"
    @test LARGE_N_LEGACY_ENERGY_MEAN_KEY == "E_mean"
    @test LARGE_N_LEGACY_GROUND_STATE_OVERLAP_KEY == "GS_overlap_mean"
    @test LARGE_N_LEGACY_GROUND_STATE_OVERLAP_TRAJECTORIES_KEY ==
          "GS_overlap_trajectories"
    @test largeN_n_group_name(64) == "N64"
    @test largeN_r_group_name(10) == "R10"
    @test is_largeN_n_group_name("N64")
    @test !is_largeN_n_group_name("R10")
    @test is_largeN_r_group_name("R10")
    @test !is_largeN_r_group_name("N64")
    @test largeN_r_from_group_name("R10") == 10
    @test LARGE_N_EVOLUTION_METHOD_KEY == "evolution_method"
    @test LARGE_N_SYSTEM_SOLVE_REUSED_ACROSS_R_KEY == "system_solve_reused_across_R"
    @test LARGE_N_BOND_SATURATION_THRESHOLD_KEY == "bond_saturation_threshold"
    @test LARGE_N_SYSTEM_SATURATION_CYCLE_KEY == "system_saturation_cycle"
    @test LARGE_N_EVOLVED_SATURATION_CYCLE_KEY == "evolved_saturation_cycle"
    @test LARGE_N_SYSTEM_MAX_BOND_KEY == "system_max_bond"
    @test LARGE_N_SYSTEM_MEAN_BOND_KEY == "system_mean_bond"
    @test LARGE_N_EVOLVED_MAX_BOND_KEY == "evolved_max_bond"
    @test LARGE_N_EVOLVED_MEAN_BOND_KEY == "evolved_mean_bond"
    @test LARGE_N_TDVP_SWEEP_MAX_BOND_KEY == "tdvp_sweep_max_bond"
    @test LARGE_N_TDVP_SWEEP_SATURATION_CYCLE_KEY ==
          "tdvp_sweep_saturation_cycle"
    @test LARGE_N_ELAPSED_SECONDS_KEY == "elapsed_seconds"
    @test LARGE_N_STOP_REASONS_KEY == "stop_reasons"
    @test LARGE_N_TRAJECTORY_SEED_RULE_KEY == "trajectory_seed_rule"
    @test LARGE_N_TRAJECTORY_SEEDS_KEY == "trajectory_seeds"
    @test LARGE_N_TRAJECTORY_INDICES_KEY == "trajectory_indices"
    @test LARGE_N_DELTA_LISTS_KEY == "delta_lists"
    @test LARGE_N_DELTA_LIST_FIRST_TRAJECTORY_KEY == "delta_list_first_trajectory"
    @test LARGE_N_DELTA_LIST_IS_COMMON_KEY == "delta_list_is_common"
    @test LARGE_N_TE_LISTS_KEY == "te_lists"
    @test LARGE_N_TE_LIST_FIRST_TRAJECTORY_KEY == "te_list_first_trajectory"
    @test LARGE_N_TE_LIST_IS_COMMON_KEY == "te_list_is_common"
    @test LARGE_N_FINAL_BOND_DIMS_GROUP == "final_bond_dims"
    @test LARGE_N_FINAL_BOND_DIMS_TRAJECTORY_PREFIX == "trajectory_"
    @test LARGE_N_DETUNING_COVERAGE_NA == LARGE_N_LABEL_NA
    @test LARGE_N_DETUNING_COVERAGE_SINGLE_DETUNING == "single_detuning"
    @test LARGE_N_DETUNING_COVERAGE_FULL_GRID == "full_grid_observed"
    @test LARGE_N_DETUNING_COVERAGE_REQUESTED_PARTIAL_GRID ==
          "requested_partial_grid"
    @test LARGE_N_DETUNING_COVERAGE_STOPPED_PARTIAL_GRID ==
          "stopped_partial_grid"
    @test LARGE_N_DETUNING_COVERAGE_PARTIAL_GRID_OBSERVED ==
          "partial_grid_observed"
    @test LARGE_N_DETUNING_COVERAGE_NO_COMPLETED_CYCLES ==
          "no_completed_cycles"
    @test LARGE_N_DETUNING_COVERAGE_MISSING_DETUNING_VALUES ==
          "missing_detuning_values"
    @test LARGE_N_ROW_SYSTEM_MAX_BOND_KEY == "sys_maxbond"
    @test LARGE_N_ROW_SYSTEM_MEAN_BOND_KEY == "sys_meanbond"
    @test LARGE_N_ROW_EVOLVED_MAX_BOND_KEY == "evolved_maxbond"
    @test LARGE_N_ROW_EVOLVED_MEAN_BOND_KEY == "evolved_meanbond"
    @test LARGE_N_ROW_TDVP_SWEEP_MAX_BOND_KEY == "tdvp_sweep_maxbond"
    @test LARGE_N_BOND_CAP_SOURCE_SYSTEM == "system"
    @test LARGE_N_BOND_CAP_SOURCE_EVOLVED == "evolved"
    @test LARGE_N_BOND_CAP_SOURCE_TDVP_SWEEP == "tdvp_sweep"
    @test LARGE_N_BOND_CAP_SOURCES == ("system", "evolved", "tdvp_sweep")
    @test LARGE_N_BOND_STATUS_NO_CAP_HIT == "no_cap_hit"
    @test LARGE_N_BOND_STATUS_SYSTEM_CAP == "not_converged_system_cap"
    @test LARGE_N_BOND_STATUS_EVOLVED_CAP == "not_converged_evolved_cap"
    @test LARGE_N_BOND_STATUS_TDVP_SWEEP_CAP == "not_converged_tdvp_sweep_cap"
    @test LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_CAP ==
          "not_converged_system_and_evolved_cap"
    @test LARGE_N_BOND_STATUS_SYSTEM_AND_TDVP_SWEEP_CAP ==
          "not_converged_system_and_tdvp_sweep_cap"
    @test LARGE_N_BOND_STATUS_EVOLVED_AND_TDVP_SWEEP_CAP ==
          "not_converged_evolved_and_tdvp_sweep_cap"
    @test LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_AND_TDVP_SWEEP_CAP ==
          "not_converged_system_and_evolved_and_tdvp_sweep_cap"
    @test LARGE_N_BOND_STATUS_PREFIX == "not_converged"
    @test LARGE_N_BOND_STATUS_SUFFIX == "cap"
    @test LARGE_N_BOND_STATUSES == (
        LARGE_N_BOND_STATUS_NO_CAP_HIT,
        LARGE_N_BOND_STATUS_SYSTEM_CAP,
        LARGE_N_BOND_STATUS_EVOLVED_CAP,
        LARGE_N_BOND_STATUS_TDVP_SWEEP_CAP,
        LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_CAP,
        LARGE_N_BOND_STATUS_SYSTEM_AND_TDVP_SWEEP_CAP,
        LARGE_N_BOND_STATUS_EVOLVED_AND_TDVP_SWEEP_CAP,
        LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_AND_TDVP_SWEEP_CAP,
    )
    @test LARGE_N_PROGRESS_STAGE_INITIAL == "initial"
    @test LARGE_N_PROGRESS_STAGE_PREPARED == "prepared"
    @test LARGE_N_PROGRESS_STAGE_EVOLVED == "evolved"
    @test LARGE_N_PROGRESS_STAGE_UPDATED == "updated"
    @test LARGE_N_PROGRESS_STAGE_TDVP_SWEEP == "tdvp_sweep"
    @test LARGE_N_PROGRESS_STAGES == (
        "initial",
        "prepared",
        "evolved",
        "updated",
        "tdvp_sweep",
    )
    @test LARGE_N_PROGRESS_TDVP_SWEEP_KEY == "tdvp_sweep"
    @test LARGE_N_PROGRESS_TDVP_TIME_KEY == "tdvp_time"
    @test LARGE_N_PROGRESS_STAGE_KEY == "stage"
    @test LARGE_N_PROGRESS_STEP_KEY == "step"
    @test LARGE_N_PROGRESS_CYCLE_KEY == "cycle"
    @test LARGE_N_PROGRESS_DELTA_KEY == "delta"
    @test LARGE_N_PROGRESS_TE_KEY == "te"
    @test LARGE_N_PROGRESS_TIMESTAMP_KEY == "timestamp"
    @test LARGE_N_PROGRESS_N_KEY == "N"
    @test LARGE_N_PROGRESS_METHOD_KEY == "method"
    @test LARGE_N_PROGRESS_EVOLUTION_KEY == "evolution"
    @test LARGE_N_PROGRESS_R_KEY == "R"
    @test LARGE_N_PROGRESS_TRAJECTORY_KEY == "trajectory"
    @test LARGE_N_PROGRESS_SEED_KEY == "seed"
    @test LARGE_N_PROGRESS_DMAX_KEY == "Dmax"
    @test LARGE_N_PROGRESS_CUTOFF_KEY == "cutoff"
    @test LARGE_N_PROGRESS_G_KEY == "g"
    @test LARGE_N_PROGRESS_TAU_KEY == "tau"
    @test LARGE_N_PROGRESS_ENERGY_PER_SITE_KEY == "energy_per_site"
    @test LARGE_N_PROGRESS_RELATIVE_ENERGY_KEY == "relative_energy"
    @test LARGE_N_PROGRESS_OVERLAP_KEY == "overlap"
    @test LARGE_N_PROGRESS_CSV_COLUMNS == (
        LARGE_N_PROGRESS_TIMESTAMP_KEY,
        LARGE_N_PROGRESS_N_KEY,
        LARGE_N_PROGRESS_METHOD_KEY,
        LARGE_N_PROGRESS_EVOLUTION_KEY,
        LARGE_N_PROGRESS_R_KEY,
        LARGE_N_PROGRESS_TRAJECTORY_KEY,
        LARGE_N_PROGRESS_SEED_KEY,
        LARGE_N_PROGRESS_DMAX_KEY,
        LARGE_N_PROGRESS_CUTOFF_KEY,
        LARGE_N_PROGRESS_G_KEY,
        LARGE_N_PROGRESS_TAU_KEY,
        LARGE_N_PROGRESS_STAGE_KEY,
        LARGE_N_PROGRESS_STEP_KEY,
        LARGE_N_PROGRESS_CYCLE_KEY,
        LARGE_N_PROGRESS_DELTA_KEY,
        LARGE_N_PROGRESS_TE_KEY,
        LARGE_N_PROGRESS_ENERGY_PER_SITE_KEY,
        LARGE_N_PROGRESS_RELATIVE_ENERGY_KEY,
        LARGE_N_PROGRESS_OVERLAP_KEY,
        LARGE_N_SYSTEM_MAX_BOND_KEY,
        LARGE_N_SYSTEM_MEAN_BOND_KEY,
        LARGE_N_EVOLVED_MAX_BOND_KEY,
        LARGE_N_EVOLVED_MEAN_BOND_KEY,
        LARGE_N_PROGRESS_TDVP_SWEEP_KEY,
        LARGE_N_PROGRESS_TDVP_TIME_KEY,
        LARGE_N_ELAPSED_SECONDS_KEY,
    )
    @test LARGE_N_PROGRESS_GROUP_COLUMNS == (
        LARGE_N_PROGRESS_N_KEY,
        LARGE_N_PROGRESS_METHOD_KEY,
        LARGE_N_PROGRESS_EVOLUTION_KEY,
        LARGE_N_PROGRESS_R_KEY,
        LARGE_N_PROGRESS_TRAJECTORY_KEY,
        LARGE_N_PROGRESS_SEED_KEY,
        LARGE_N_PROGRESS_DMAX_KEY,
        LARGE_N_PROGRESS_CUTOFF_KEY,
        LARGE_N_PROGRESS_G_KEY,
        LARGE_N_PROGRESS_TAU_KEY,
    )
    @test largeN_progress_stage(:initial) == LARGE_N_PROGRESS_STAGE_INITIAL
    @test largeN_progress_stage(:prepared) == LARGE_N_PROGRESS_STAGE_PREPARED
    @test largeN_progress_stage(:evolved) == LARGE_N_PROGRESS_STAGE_EVOLVED
    @test largeN_progress_stage(:updated) == LARGE_N_PROGRESS_STAGE_UPDATED
    @test largeN_progress_stage(:tdvp_sweep) == LARGE_N_PROGRESS_STAGE_TDVP_SWEEP
    @test_throws ArgumentError largeN_progress_stage(:renormalized)
    @test require_largeN_progress_stage_label("initial") == LARGE_N_PROGRESS_STAGE_INITIAL
    @test require_largeN_progress_stage_label(LARGE_N_PROGRESS_STAGE_TDVP_SWEEP) ==
          LARGE_N_PROGRESS_STAGE_TDVP_SWEEP
    @test_throws ArgumentError require_largeN_progress_stage_label("renormalized")
    @test progress_detuning_coverage_status(0, 5, 0) ==
          LARGE_N_DETUNING_COVERAGE_NO_COMPLETED_CYCLES
    @test progress_detuning_coverage_status(0, 5, 2) ==
          LARGE_N_DETUNING_COVERAGE_MISSING_DETUNING_VALUES
    @test progress_detuning_coverage_status(1, 1, 1) ==
          LARGE_N_DETUNING_COVERAGE_SINGLE_DETUNING
    @test progress_detuning_coverage_status(2, 5, 2) ==
          LARGE_N_DETUNING_COVERAGE_PARTIAL_GRID_OBSERVED
    @test progress_detuning_coverage_status(5, 5, 5) ==
          LARGE_N_DETUNING_COVERAGE_FULL_GRID
    @test progress_detuning_coverage_status(5, 0, 5) == LARGE_N_DETUNING_COVERAGE_NA
    @test largeN_trajectory_seed(20260617, 64, 10, 1) == 84360618
    @test largeN_trajectory_seed(7, 2, 1, 3) == 2010010
    @test_throws ArgumentError largeN_trajectory_seed(7, 2, 0, 1)
    @test_throws ArgumentError largeN_trajectory_seed(7, 2, 100, 1)
    @test_throws ArgumentError largeN_trajectory_seed(7, 2, 1, 0)
    @test_throws ArgumentError largeN_trajectory_seed(7, 2, 1, 10_000)

    gap_protocol = largeN_detuning_protocol(0.5; delta_max_factor=6.0)
    @test gap_protocol.source == LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE
    @test gap_protocol.reference_gap == 0.5
    @test gap_protocol.delta_min == 0.5
    @test gap_protocol.delta_max == 3.0
    @test gap_protocol.fixed_across_dmax == false
    @test largeN_delta_values(gap_protocol, 1) == [0.5]
    @test largeN_delta_values(gap_protocol, 3) == [0.5, 1.75, 3.0]

    fixed_protocol = largeN_detuning_protocol(
        0.75; delta_min=0.25, delta_max=1.25, delta_max_factor=6.0
    )
    @test fixed_protocol.source == LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE
    @test fixed_protocol.reference_gap == 0.75
    @test fixed_protocol.delta_min == 0.25
    @test fixed_protocol.delta_max == 1.25
    @test isnan(fixed_protocol.delta_max_factor)
    @test fixed_protocol.fixed_across_dmax == true
    @test largeN_delta_values(fixed_protocol, 2) == [0.25, 1.25]

    @test largeN_method_kind_from_name("mcwf") === :mcwf
    @test largeN_method_kind_from_name("MPO") === :mpo
    @test_throws ArgumentError largeN_method_kind_from_name("ed")

    fixed_protocol_negative_reference = largeN_detuning_protocol(
        -0.035; delta_min=0.5, delta_max=0.5, delta_max_factor=6.0
    )
    @test fixed_protocol_negative_reference.source == LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE
    @test fixed_protocol_negative_reference.reference_gap == -0.035
    @test fixed_protocol_negative_reference.delta_min == 0.5
    @test fixed_protocol_negative_reference.delta_max == 0.5

    @test_throws ArgumentError largeN_detuning_protocol(0.0; delta_max_factor=6.0)
    @test_throws ArgumentError largeN_detuning_protocol(0.5; delta_max_factor=0.5)
    @test_throws ArgumentError largeN_detuning_protocol(0.5; delta_min=0.5)
    @test_throws ArgumentError largeN_detuning_protocol(0.5; delta_min=1.0, delta_max=0.5)
    @test_throws ArgumentError largeN_detuning_protocol(NaN; delta_min=0.5, delta_max=0.5)
    @test_throws ArgumentError largeN_delta_values(fixed_protocol, 0)

    @test tn_method_maxdim(MonteCarloWavefunction(), 12) == 12
    @test tn_method_maxdim(DensityMatrix(), 12) == 48
    @test tn_trotter_maxdim(MonteCarloWavefunction(), 12) == 12
    @test tn_trotter_maxdim(DensityMatrix(), 12) == 48

    @test first_bond_saturation_cycle([1, 11, 50, 159], 320) == 0
    @test first_bond_saturation_cycle([1, 11, 50, 320], 320) == 3
    @test first_bond_saturation_cycle([320, 11, 50], 320) == 0
    @test first_bond_saturation_cycle([1, 320, 320], 320) == 1

    @test first_recorded_saturation_cycle([0, 0, 0]) == 0
    @test first_recorded_saturation_cycle([0, 4, 2]) == 2
    @test first_recorded_saturation_cycle([3, 1, 2]) == 1

    @test saturation_cycle_label(0) == "none"
    @test saturation_cycle_label(4) == "4"
    @test saturation_cycle_label(missing) == "n/a"
    @test bond_cap_status(0, 0) == LARGE_N_BOND_STATUS_NO_CAP_HIT
    @test bond_cap_status(3, 0) == LARGE_N_BOND_STATUS_SYSTEM_CAP
    @test bond_cap_status(0, 2) == LARGE_N_BOND_STATUS_EVOLVED_CAP
    @test bond_cap_status(3, 2) == LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_CAP
    @test bond_cap_status(0, 0, 4) == LARGE_N_BOND_STATUS_TDVP_SWEEP_CAP
    @test bond_cap_status(3, 0, 4) ==
          LARGE_N_BOND_STATUS_SYSTEM_AND_TDVP_SWEEP_CAP
    @test bond_cap_status(0, 2, 4) ==
          LARGE_N_BOND_STATUS_EVOLVED_AND_TDVP_SWEEP_CAP
    @test bond_cap_status(3, 2, 4) ==
          LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_AND_TDVP_SWEEP_CAP
    @test require_largeN_bond_status_label(LARGE_N_BOND_STATUS_NO_CAP_HIT) ==
          LARGE_N_BOND_STATUS_NO_CAP_HIT
    @test require_largeN_bond_status_label(
        LARGE_N_BOND_STATUS_EVOLVED_AND_TDVP_SWEEP_CAP
    ) == LARGE_N_BOND_STATUS_EVOLVED_AND_TDVP_SWEEP_CAP
    @test _largeN_bond_status_label(("tdvp_sweep", "system")) ==
          LARGE_N_BOND_STATUS_SYSTEM_AND_TDVP_SWEEP_CAP
    @test_throws ArgumentError require_largeN_bond_status_label("")
    @test_throws ArgumentError require_largeN_bond_status_label("not_converged_bath_cap")
    @test_throws ArgumentError _largeN_bond_status_label(("system", "system"))
    @test_throws ArgumentError _largeN_bond_status_label(("bath",))
    @test effective_bond_dimension_label(48, 0, 64) == "48"
    @test effective_bond_dimension_label(64, 2, 64) == ">=64"
    @test effective_bond_dimension_label(72, 2, 64) == ">=72"

    @test bond_dimension_quantiles([10, 20, 30, 40], [0.5, 0.75]) ≈ [25.0, 32.5]
    @test all(isnan, bond_dimension_quantiles(Int[], [0.5, 0.9]))

    @test bond_dimension_fraction_at_least([10, 20, 30, 40], 25) == 0.5
    @test isnan(bond_dimension_fraction_at_least(Int[], 25))
    @test bond_dimension_threshold_fractions([10, 20, 30, 40], 40, [0.5, 0.75]) ==
          [0.75, 0.5]

    system_max = [1 1; 6 9; 12 10]
    system_mean = [1.0 1.0; 4.0 5.0; 8.0 5.0]
    evolved_max = [99 99; 12 6; 8 14]
    evolved_mean = [NaN NaN; 8.0 7.0; 9.0 11.0]
    @test final_system_max_bond(system_max) == 12
    @test final_system_mean_bond(system_mean) == 6.5
    @test peak_evolved_max_bond(evolved_max) == 14
    @test peak_evolved_mean_bond(evolved_mean) == 10.0
end
