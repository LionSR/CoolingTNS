#!/usr/bin/env julia
"""
Summarize effective bond dimensions from large-N tensor-network campaign HDF5 files.

Example:

    julia --project=. scripts/validation/summarize_largeN_bond_dimensions.jl \
        /tmp/coolingtns_largeN_mcwf_N64_R1_steps4_Dmax320.h5 \
        /tmp/coolingtns_largeN_mcwf_N64_R2-5-10_steps4_Dmax320.h5

    julia --project=. scripts/validation/summarize_largeN_bond_dimensions.jl \
        --compact /tmp/coolingtns_largeN_mcwf_N64_R1_steps4_Dmax320.h5

    julia --project=. scripts/validation/summarize_largeN_bond_dimensions.jl \
        --compact --combine-trajectories /tmp/largeN_*_traj*.h5

The output is a Markdown table.  When present, the stored detuning protocol,
coupling strength `g`, initial-state protocol, bath-evolution time `te`, and
time protocol are shown next to the bond-dimension diagnostics, so
initial-state controls, fixed-detuning cutoff, coupling-strength scans,
time-ladder, and randomized-time sweeps can be audited from the summary alone.
The compact and full tables also show the stored MCWF trajectory labels and
seeds, making single-trajectory diagnostics distinguishable from ensemble
summaries without inspecting the HDF5 file by hand.  Legacy MCWF files without
stored seeds are displayed as `legacy_missing`; deterministic non-MCWF rows
without seeds are displayed as `n/a`.
The `delta_range` column is the stored protocol
interval; for `R=1`, the campaign samples only the lower endpoint.  For
multi-trajectory data, final-link quantiles and threshold fractions are computed
per trajectory and then averaged over trajectories.  Stop-on-cap provenance is
read directly from the HDF5 fields `requested_steps`, `completed_steps`,
`stop_reasons`, and `elapsed_seconds` when available.  The elapsed column is a
sum over trajectory elapsed times, matching the sequential campaign driver.
With `--combine-trajectories`, compatible split trajectory-axis files are grouped
by their physical protocol, including `g`, the initial state, `te`, and whether
cycle times are fixed or randomized, and summarized as one row after verifying
that their stored trajectory labels are non-overlapping.  For stopped prefixes
with unequal completed cycle counts, the energy columns are statistics of the
individual trajectory summaries rather than a reconstructed cycle-aligned
ensemble history.  Protocol buckets containing only one file are left unchanged,
since there is no independent trajectory file to combine with it.
The `traj cycles/hour` column is the corresponding completed trajectory-cycle
throughput, `3600 * sum(completed_steps) / elapsed_total`.  For deterministic
multi-frequency schedules, `completed/requested periods` converts the same cycle
counts into full detuning-grid passes by dividing by `R`.  Random schedules and
legacy files without schedule metadata are reported as `n/a`.  The
`visited detunings` column counts distinct stored detuning values used during
completed cycles, again excluding the initial `NaN` measurement row.
The `detuning coverage` column then records whether a deterministic schedule
has actually completed at least one full detuning-grid period; a prefix shorter
than one period is marked explicitly rather than left for the reader to infer.
Single-detuning runs report `single_detuning`, and random schedules report
`n/a`.  The `truncation errors` column reports whether measured truncation-error
histories are available: current campaign files write `not_recorded`, legacy
files without the provenance field report `legacy_missing`, files with a
nonempty `truncation_errors` dataset report `measured`, and files with an empty
`truncation_errors` dataset report `empty`.
"""

using CoolingTNS
using HDF5
using Printf
using Statistics

include(joinpath(@__DIR__, "largeN_scaling_helpers.jl"))

const LINK_QUANTILE_PROBABILITIES = [0.50, 0.75, 0.90, 0.95]
const LINK_THRESHOLD_FRACTIONS = [0.50, 0.75, 0.90]
const ENERGY_TAIL_WINDOW = 10

function usage()
    println(
        "usage: julia --project=. scripts/validation/summarize_largeN_bond_dimensions.jl " *
        "[--compact] [--combine-trajectories] [--skip-invalid] FILE.h5 [FILE2.h5 ...]"
    )
end

function format_float(value::Real, digits::Int=2)
    !isfinite(value) && return "NaN"
    digits == 1 && return @sprintf("%.1f", value)
    digits == 2 && return @sprintf("%.2f", value)
    digits == 3 && return @sprintf("%.3f", value)
    digits == 5 && return @sprintf("%.5f", value)
    digits == 8 && return @sprintf("%.8f", value)
    return string(round(Float64(value); digits=digits))
end

format_integer_or_na(value::Integer) = string(value)
format_integer_or_na(::Missing) = LARGE_N_LABEL_NA
format_float_or_na(value::Real, digits::Int=2) = format_float(value, digits)
format_float_or_na(::Missing, digits::Int=2) = LARGE_N_LABEL_NA
function format_error_float(value::Real, digits::Int=3)
    !isfinite(value) && return "NaN"
    value == 0 && return format_float(value, digits)
    abs(value) < 10.0^(-digits) && return @sprintf("%.*e", max(digits - 1, 0), value)
    return format_float(value, digits)
end
format_error_float_or_na(value::Real, digits::Int=3) = format_error_float(value, digits)
format_error_float_or_na(::Missing, digits::Int=3) = LARGE_N_LABEL_NA
format_string_or_na(value) = string(value)
format_string_or_na(::Missing) = LARGE_N_LABEL_NA

function scalar_or_vector(value)
    value isa AbstractArray && return vec(value)
    return [value]
end

function read_integer_vector(run_group, key::AbstractString, default::AbstractVector{<:Integer})
    haskey(run_group, key) || return Int.(default)
    return Int.(scalar_or_vector(read(run_group[key])))
end

function read_float_vector(run_group, key::AbstractString)
    haskey(run_group, key) || return Float64[]
    return Float64.(scalar_or_vector(read(run_group[key])))
end

function read_string_vector(run_group, key::AbstractString)
    haskey(run_group, key) || return String[]
    return String.(scalar_or_vector(read(run_group[key])))
end

function read_trajectory_indices(run_group, M::Integer)
    default_indices = collect(1:M)
    return read_integer_vector(run_group, LARGE_N_TRAJECTORY_INDICES_KEY, default_indices)
end

function read_trajectory_seeds(run_group, M::Integer)
    if haskey(run_group, LARGE_N_TRAJECTORY_SEEDS_KEY)
        seeds = Int.(scalar_or_vector(read(run_group[LARGE_N_TRAJECTORY_SEEDS_KEY])))
        length(seeds) == M ||
            error(
                "trajectory_seeds length $(length(seeds)) does not match M=$M"
            )
        return Union{Missing,Int}[seed for seed in seeds]
    end
    return Union{Missing,Int}[missing for _ in 1:M]
end

function read_energy_trajectory_matrix(run_group, energy_mean::AbstractVector, M::Integer)
    if haskey(run_group, RESULT_ENERGY_TRAJECTORIES)
        values = read(run_group[RESULT_ENERGY_TRAJECTORIES])
        values isa AbstractVector && return reshape(Float64.(values), :, 1)
        return Matrix{Float64}(values)
    end
    return repeat(reshape(Float64.(energy_mean), :, 1), 1, M)
end

function read_group_value(primary_group, fallback_group, key::AbstractString, default)
    return read_first_group_value(key, default, primary_group, fallback_group)
end

function read_first_group_value(key::AbstractString, default, groups...)
    for group in groups
        haskey(group, key) && return read(group[key])
    end
    return default
end

function detuning_interval_label(delta_min::Real, delta_max::Real)
    isfinite(delta_min) && isfinite(delta_max) ||
        return LARGE_N_LABEL_UNKNOWN
    return "[$(format_float(delta_min, 8)),$(format_float(delta_max, 8))]"
end

function delta_values_interval_label(run_group)
    haskey(run_group, RESULT_DELTA_VALUES) || return LARGE_N_LABEL_UNKNOWN
    delta_values = Float64.(read(run_group[RESULT_DELTA_VALUES]))
    isempty(delta_values) && return LARGE_N_LABEL_UNKNOWN
    return detuning_interval_label(minimum(delta_values), maximum(delta_values))
end

function detuning_factor_label(source::AbstractString, factor::Real)
    source == LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE && isfinite(factor) &&
        return format_float(factor, 3)
    source == LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE && return LARGE_N_LABEL_NA
    return LARGE_N_LABEL_UNKNOWN
end

function detuning_protocol_summary(method_group, run_group)
    source = String(
        read_group_value(
            run_group,
            method_group,
            LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY,
            LARGE_N_LABEL_UNKNOWN,
        ),
    )
    delta_min = Float64(
        read_group_value(run_group, method_group, LARGE_N_DETUNING_DELTA_MIN_KEY, NaN),
    )
    delta_max = Float64(
        read_group_value(run_group, method_group, LARGE_N_DETUNING_DELTA_MAX_KEY, NaN),
    )
    factor = Float64(
        read_group_value(
            run_group,
            method_group,
            LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY,
            NaN,
        ),
    )

    interval = source == LARGE_N_LABEL_UNKNOWN ?
        delta_values_interval_label(run_group) :
        detuning_interval_label(delta_min, delta_max)
    return (
        delta_protocol=source,
        delta_range=interval,
        delta_factor=detuning_factor_label(source, factor),
    )
end

function missing_mode_reconstruction_summary()
    return (
        mode_gF=missing,
        mode_gF_source=missing,
        mode_measured_rows=missing,
        mode_last_measured_e_over_n=missing,
        mode_last_measured_abs_err_over_n=missing,
        mode_max_abs_err_over_n=missing,
    )
end

function mode_measurement_row_label(n_measured::Integer, n_rows::Integer)
    n_rows <= 0 && return LARGE_N_LABEL_NA
    return "$(n_measured)/$(n_rows)"
end

"""Return false for absent mode data, true for a complete payload, and reject partial payloads."""
function validate_mode_observable_payload(run_group)
    present_keys = String[
        key for key in RESULT_MODE_OBSERVABLE_PAYLOAD_KEYS if haskey(run_group, key)
    ]
    isempty(present_keys) && return false

    missing_keys = String[
        key for key in RESULT_MODE_OBSERVABLE_PAYLOAD_KEYS if !haskey(run_group, key)
    ]
    isempty(missing_keys) && return true

    error(
        "incomplete mode-observable metadata: found " *
        join(present_keys, ", ") *
        " but missing " *
        join(missing_keys, ", ") *
        ". Remove the partial mode datasets or write the complete mode-observable payload."
    )
end

function mode_reconstruction_summary(root, run_group, N::Integer, energy_mean;
                                     energy_name::AbstractString=RESULT_ENERGY)
    validate_mode_observable_payload(run_group) ||
        return missing_mode_reconstruction_summary()

    mode_gF = Int(read(run_group[RESULT_MODE_GF]))
    mode_gF_source = String(read(run_group[RESULT_MODE_GF_SOURCE]))

    model = haskey(root, LARGE_N_ROOT_MODEL_KEY) ?
        String(read(root[LARGE_N_ROOT_MODEL_KEY])) :
        LARGE_N_LABEL_UNKNOWN
    bc = haskey(root, LARGE_N_ROOT_BC_KEY) ?
        String(read(root[LARGE_N_ROOT_BC_KEY])) :
        LARGE_N_LABEL_UNKNOWN
    if model != "ising" || !(bc in ("periodic", "antiperiodic"))
        error(
            "mode measurements are present, but the file describes model='$model' " *
            "with bc='$bc'; mode-energy reconstruction is defined here only for " *
            "the integrable Ising chain with periodic or antiperiodic spin BC"
        )
    end
    haskey(root, LARGE_N_ROOT_J_KEY) && haskey(root, LARGE_N_ROOT_H_KEY) ||
        error("mode measurements are present, but root datasets J and h are missing")

    # The stored k-grid determines the fermionic sector.  Here `ham_params`
    # supplies the common Ising parameters needed by the shared reconstruction
    # routine.
    ham_params = IsingParameters(
        N,
        Float64(read(root[LARGE_N_ROOT_J_KEY])),
        Float64(read(root[LARGE_N_ROOT_H_KEY])),
        Symbol(bc),
    )
    mode_hk = Float64.(read(run_group[RESULT_MODE_HK]))
    mode_nk = Float64.(read(run_group[RESULT_MODE_NK]))
    k_indices = Float64.(vec(read(run_group[RESULT_MODE_K_INDICES])))
    mode_ek_values = Float64.(vec(read(run_group[RESULT_MODE_ENERGIES])))
    validate_mode_ek_values_match_grid(
        mode_ek_values, k_indices, N, ham_params.params.J, ham_params.params.h)
    measured = validate_mode_measurement_rows(
        mode_hk,
        mode_nk,
        read(run_group[RESULT_MODE_MEASUREMENT_CYCLES]);
        energy=energy_mean,
        energy_name=energy_name,
    )
    valid_rows = measured.rows

    measured_label = mode_measurement_row_label(length(valid_rows), length(energy_mean))
    mode_energy = ising_energy_from_mode_hk(k_indices, mode_hk[valid_rows, :], ham_params)
    direct_energy = Float64.(energy_mean[valid_rows])
    abs_error_over_n = abs.(mode_energy .- direct_energy) ./ N
    return (
        mode_gF=mode_gF,
        mode_gF_source=mode_gF_source,
        mode_measured_rows=measured_label,
        mode_last_measured_e_over_n=mode_energy[end] / N,
        mode_last_measured_abs_err_over_n=abs_error_over_n[end],
        mode_max_abs_err_over_n=maximum(abs_error_over_n),
    )
end

energy_tail_start(nsteps::Integer) = max(1, nsteps - ENERGY_TAIL_WINDOW + 1)

function range_label(values::AbstractVector{<:Integer})
    isempty(values) && return LARGE_N_LABEL_NA
    lo, hi = extrema(values)
    lo == hi && return string(lo)
    return "$lo-$hi"
end

function range_float_label(values::AbstractVector{<:Real}; digits::Int=2)
    isempty(values) && return LARGE_N_LABEL_NA
    finite_values = Float64[value for value in values if isfinite(value)]
    isempty(finite_values) && return LARGE_N_LABEL_NA
    lo, hi = extrema(finite_values)
    lo == hi && return format_float(lo, digits)
    return "$(format_float(lo, digits))-$(format_float(hi, digits))"
end

function schedule_label(root, method_group, run_group)
    schedule = read_first_group_value(
        RESULT_SCHEDULE,
        nothing,
        run_group,
        method_group,
        root,
    )
    schedule === nothing && return LARGE_N_LABEL_UNKNOWN
    return string(parse_multi_frequency_schedule(schedule))
end

function randomize_times_flag(root, method_group, run_group)
    return Bool(
        read_first_group_value(RESULT_RANDOMIZE_TIMES, false, run_group, method_group, root)
    )
end

time_protocol_label(randomize_times::Bool) = randomize_times ? "randomized" : "fixed"

function initial_state_metadata(root, method_group, run_group)
    init_state = String(
        read_first_group_value(
            RESULT_INIT_STATE,
            LARGE_N_LABEL_UNKNOWN,
            run_group,
            method_group,
            root,
        )
    )
    theta = Float64(
        read_first_group_value(RESULT_INIT_THETA, NaN, run_group, method_group, root)
    )
    return init_state, theta
end

function init_protocol_label(init_state::AbstractString, theta::Real)
    init_state == "theta" && return "theta=$(format_float(theta, 3))"
    return String(init_state)
end

function initial_state_group_key(row)
    init_state = String(row.init_state)
    theta_key = init_state == "theta" && isfinite(row.theta) ? row.theta : missing
    return (init_state, theta_key)
end

function initial_state_sort_key(row)
    init_state = String(row.init_state)
    theta_key = init_state == "theta" && isfinite(row.theta) ? row.theta : Inf
    return (init_state, theta_key, String(row.init_protocol))
end

is_deterministic_schedule(schedule::AbstractString) =
    schedule in ("round_robin", "descending")

function completed_requested_periods_label(
    completed_steps::AbstractVector{<:Integer},
    requested_steps::AbstractVector{<:Integer},
    R::Integer,
    schedule::AbstractString,
)
    is_deterministic_schedule(schedule) || return LARGE_N_DETUNING_COVERAGE_NA
    R > 0 || return LARGE_N_DETUNING_COVERAGE_NA
    completed_periods = Float64.(completed_steps) ./ R
    requested_periods = Float64.(requested_steps) ./ R
    return "$(range_float_label(completed_periods))/$(range_float_label(requested_periods))"
end

function detuning_coverage_status(
    completed_steps::AbstractVector{<:Integer},
    requested_steps::AbstractVector{<:Integer},
    R::Integer,
    schedule::AbstractString,
)
    is_deterministic_schedule(schedule) || return LARGE_N_DETUNING_COVERAGE_NA
    R > 0 || return LARGE_N_DETUNING_COVERAGE_NA
    (isempty(completed_steps) || isempty(requested_steps)) &&
        return LARGE_N_DETUNING_COVERAGE_NA
    R == 1 && return LARGE_N_DETUNING_COVERAGE_SINGLE_DETUNING
    minimum(completed_steps) >= R && return LARGE_N_DETUNING_COVERAGE_FULL_GRID
    # This is a run-level status: if any trajectory was requested for a full
    # period, an under-one-period completed prefix is treated as stopped short.
    maximum(requested_steps) < R &&
        return LARGE_N_DETUNING_COVERAGE_REQUESTED_PARTIAL_GRID
    return LARGE_N_DETUNING_COVERAGE_STOPPED_PARTIAL_GRID
end

function delta_history_matrix(run_group)
    if haskey(run_group, LARGE_N_DELTA_LISTS_KEY)
        return delta_history_matrix_from_values(read(run_group[LARGE_N_DELTA_LISTS_KEY]))
    elseif haskey(run_group, RESULT_DELTA_LIST)
        return delta_history_matrix_from_values(read(run_group[RESULT_DELTA_LIST]))
    end
    return nothing
end

function delta_history_matrix_from_values(values)
    values isa Number && return reshape([Float64(values)], 1, 1)
    array = Float64.(values)
    array isa AbstractVector && return reshape(array, :, 1)
    array isa AbstractMatrix && return Matrix(array)
    error("delta history must be stored as a scalar, vector, or matrix")
end

function distinct_completed_delta_counts(delta_history, completed_steps::AbstractVector{<:Integer})
    delta_history === nothing && return Int[]
    ncols = size(delta_history, 2)
    counts = Int[]
    for (trajectory, completed) in enumerate(completed_steps)
        column = ncols == 1 ? 1 : trajectory
        column <= ncols || error(
            "$(LARGE_N_DELTA_LISTS_KEY) has $ncols trajectory columns, but completed_steps " *
            "contains at least $trajectory trajectories"
        )
        last_row = min(Int(completed) + 1, size(delta_history, 1))
        if last_row < 2
            push!(counts, 0)
            continue
        end
        used_values = [
            value for value in view(delta_history, 2:last_row, column)
            if isfinite(value)
        ]
        push!(counts, length(unique(used_values)))
    end
    return counts
end

function visited_detunings_label_from_counts(
    counts::AbstractVector{<:Integer},
    R::Integer;
    unknown_count::Integer=0,
    total_count::Integer=length(counts) + unknown_count,
)
    R > 0 || return LARGE_N_LABEL_NA
    unknown_count >= 0 ||
        error("unknown detuning-history count must be non-negative")
    total_count >= length(counts) + unknown_count ||
        error("total detuning-history count is smaller than known plus unknown counts")
    isempty(counts) && unknown_count == 0 && return LARGE_N_LABEL_NA
    known_label = isempty(counts) ? "" : "$(range_label(counts))/$(R)"
    unknown_count == 0 && return known_label
    unknown_label = "unknownx$(unknown_count)/$(total_count)"
    return isempty(known_label) ? unknown_label : "$(known_label)+$(unknown_label)"
end

function stop_reason_label(reasons::AbstractVector{<:AbstractString})
    values = String.(reasons)
    nonempty = filter(!isempty, values)
    isempty(nonempty) && return LARGE_N_LABEL_NONE
    unique_reasons = sort(unique(nonempty))
    length(unique_reasons) == 1 && length(nonempty) == length(values) &&
        return only(unique_reasons)
    return join(
        ["$(reason)x$(count(==(reason), nonempty))/$(length(values))" for reason in unique_reasons],
        "+",
    )
end

function trajectory_cycles_per_hour(completed_steps::AbstractVector{<:Integer},
                                    elapsed_seconds::Real)
    isfinite(elapsed_seconds) && elapsed_seconds > 0 || return NaN
    return 3600.0 * sum(completed_steps) / elapsed_seconds
end

function method_from_name(method_name::AbstractString)
    # HDF5 stores method names as strings; the cap itself is still determined
    # by the library dispatch rule in `tn_method_maxdim`.
    kind = largeN_method_kind_from_name(method_name)
    kind === :mcwf && return MonteCarloWavefunction()
    kind === :mpo && return DensityMatrix()
    error("unreachable large-N method kind '$kind'")
end

function saturation_threshold_for(root, method_group, run_group, method_name::AbstractString)
    haskey(run_group, LARGE_N_BOND_SATURATION_THRESHOLD_KEY) &&
        return Int(read(run_group[LARGE_N_BOND_SATURATION_THRESHOLD_KEY]))
    haskey(method_group, LARGE_N_BOND_SATURATION_THRESHOLD_KEY) &&
        return Int(read(method_group[LARGE_N_BOND_SATURATION_THRESHOLD_KEY]))
    return tn_method_maxdim(method_from_name(method_name), Int(read(root[LARGE_N_ROOT_DMAX_KEY])))
end

function final_link_dimensions(run_group)
    !haskey(run_group, LARGE_N_FINAL_BOND_DIMS_GROUP) && return Vector{Vector{Int}}()
    bond_group = run_group[LARGE_N_FINAL_BOND_DIMS_GROUP]
    names = sort(
        String.(keys(bond_group));
        by=name -> parse(Int, replace(name, LARGE_N_FINAL_BOND_DIMS_TRAJECTORY_PREFIX => "")),
    )
    return [Int.(read(bond_group[name])) for name in names]
end

function mean_link_quantiles(link_dims_by_trajectory)
    isempty(link_dims_by_trajectory) &&
        return fill(NaN, length(LINK_QUANTILE_PROBABILITIES))
    values = [
        bond_dimension_quantiles(dims, LINK_QUANTILE_PROBABILITIES)
        for dims in link_dims_by_trajectory
    ]
    return vec(mean(reduce(hcat, values); dims=2))
end

function mean_link_threshold_fractions(link_dims_by_trajectory, threshold::Integer)
    isempty(link_dims_by_trajectory) &&
        return fill(NaN, length(LINK_THRESHOLD_FRACTIONS))
    values = [
        bond_dimension_threshold_fractions(dims, threshold, LINK_THRESHOLD_FRACTIONS)
        for dims in link_dims_by_trajectory
    ]
    return vec(mean(reduce(hcat, values); dims=2))
end

function first_saturation_from_dataset(run_group, dataset_name::AbstractString)
    haskey(run_group, dataset_name) || return 0
    return first_recorded_saturation_cycle(Int.(read(run_group[dataset_name])))
end

function first_saturation_from_history(history, threshold::Integer)
    cycles = [
        first_bond_saturation_cycle(history[:, trajectory], threshold)
        for trajectory in axes(history, 2)
    ]
    return first_recorded_saturation_cycle(cycles)
end

"""
    read_largeN_energy_mean_with_name(run_group)

Read the large-N aggregate energy time series.  New files use the canonical
`RESULT_ENERGY` key, while archived files may still use the legacy `E_mean`
dataset.  Return both the values and the HDF5 dataset name used, so downstream
validation errors identify the actual source dataset.
"""
function read_largeN_energy_mean_with_name(run_group)
    haskey(run_group, RESULT_ENERGY) &&
        return (values=read(run_group[RESULT_ENERGY]), name=RESULT_ENERGY)
    haskey(run_group, LARGE_N_LEGACY_ENERGY_MEAN_KEY) &&
        return (
            values=read(run_group[LARGE_N_LEGACY_ENERGY_MEAN_KEY]),
            name=LARGE_N_LEGACY_ENERGY_MEAN_KEY,
        )
    error(
        "large-N run group is missing both $RESULT_ENERGY and legacy " *
        "$LARGE_N_LEGACY_ENERGY_MEAN_KEY " *
        "energy-mean datasets"
    )
end

"""
    read_overlap_trajectory_matrix(run_group, nsteps, M)

Return a `(nsteps, M)` matrix of ground-state overlaps.  Current campaign files
store trajectory-resolved overlaps; older files may store only the aggregate
overlap history.  Files without overlap data are summarized with `NaN` entries
so legacy bond-dimension tables remain readable.
"""
function read_overlap_trajectory_matrix(run_group, nsteps::Integer, M::Integer)
    trajectory_key = if haskey(run_group, RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES)
        RESULT_GROUND_STATE_OVERLAP_TRAJECTORIES
    elseif haskey(run_group, LARGE_N_LEGACY_GROUND_STATE_OVERLAP_TRAJECTORIES_KEY)
        LARGE_N_LEGACY_GROUND_STATE_OVERLAP_TRAJECTORIES_KEY
    else
        nothing
    end
    if trajectory_key !== nothing
        values = read(run_group[trajectory_key])
        values isa AbstractVector && return reshape(Float64.(values), :, 1)
        return Matrix{Float64}(values)
    end

    overlap_mean = if haskey(run_group, RESULT_GROUND_STATE_OVERLAP)
        Float64.(vec(read(run_group[RESULT_GROUND_STATE_OVERLAP])))
    elseif haskey(run_group, LARGE_N_LEGACY_GROUND_STATE_OVERLAP_KEY)
        Float64.(vec(read(run_group[LARGE_N_LEGACY_GROUND_STATE_OVERLAP_KEY])))
    else
        fill(NaN, nsteps)
    end
    return repeat(reshape(overlap_mean, :, 1), 1, M)
end

function truncation_error_history_status(run_group)
    if haskey(run_group, RESULT_TRUNCATION_ERROR_HISTORY_STATUS)
        return require_truncation_error_history_status_label(
            read(run_group[RESULT_TRUNCATION_ERROR_HISTORY_STATUS])
        )
    end
    if haskey(run_group, RESULT_TRUNCATION_ERRORS)
        values = read(run_group[RESULT_TRUNCATION_ERRORS])
        # The in-repository writer skips empty result arrays; this branch handles
        # externally authored or hand-edited legacy files defensively.
        values isa AbstractArray && isempty(values) &&
            return TRUNCATION_ERROR_HISTORY_EMPTY
        return TRUNCATION_ERROR_HISTORY_MEASURED
    end
    return TRUNCATION_ERROR_HISTORY_LEGACY_MISSING
end

"""
    read_largeN_system_size(file_name, n_group_name, n_group)

Read the stored large-N system size and require it to match the canonical HDF5
group name, e.g. an `N64` group must store `N = 64`.
"""
function read_largeN_system_size(file_name::AbstractString, n_group_name::AbstractString,
                                 n_group::HDF5.Group)
    stored_N = Int(read(n_group[LARGE_N_SYSTEM_SIZE_KEY]))
    group_N = largeN_n_from_group_name(n_group_name)
    stored_N == group_N || throw(ArgumentError(
        "large-N HDF5 group $(basename(file_name))/$n_group_name stores " *
        "$(LARGE_N_SYSTEM_SIZE_KEY)=$stored_N but the group name encodes " *
        "$(LARGE_N_SYSTEM_SIZE_KEY)=$group_N"
    ))
    return stored_N
end

function summarize_run(file_name::AbstractString, root, n_group_name::AbstractString,
                       n_group::HDF5.Group, method_name::AbstractString,
                       method_group, r_group_name::AbstractString, run_group)
    N = read_largeN_system_size(file_name, n_group_name, n_group)
    R = largeN_r_from_group_name(r_group_name)
    M = Int(read(run_group[LARGE_N_TRAJECTORY_COUNT_KEY]))
    evolution = String(
        read_group_value(method_group, root, LARGE_N_EVOLUTION_METHOD_KEY, LARGE_N_LABEL_UNKNOWN)
    )
    te = Float64(read_first_group_value(RESULT_TE, NaN, run_group, method_group, root))
    g = Float64(read_first_group_value(LARGE_N_ROOT_G_KEY, NaN, run_group, method_group, root))
    randomize_times = randomize_times_flag(root, method_group, run_group)
    time_protocol = time_protocol_label(randomize_times)
    init_state, theta = initial_state_metadata(root, method_group, run_group)
    init_protocol = init_protocol_label(init_state, theta)
    threshold = saturation_threshold_for(root, method_group, run_group, method_name)
    trajectory_indices = read_trajectory_indices(run_group, M)
    length(trajectory_indices) == M ||
        error(
            "trajectory_indices length $(length(trajectory_indices)) does not match M=$M " *
            "in $(basename(file_name))/$n_group_name/$method_name/$r_group_name"
        )
    trajectory_seeds = read_trajectory_seeds(run_group, M)
    E0 = Float64(read_group_value(method_group, root, LARGE_N_GROUND_ENERGY_KEY, NaN))

    energy_dataset = read_largeN_energy_mean_with_name(run_group)
    energy_mean = vec(Float64.(energy_dataset.values))
    energy_trajectories = read_energy_trajectory_matrix(run_group, energy_mean, M)
    size(energy_trajectories, 2) == M ||
        error(
            "energy trajectory column count $(size(energy_trajectories, 2)) does not " *
            "match M=$M in $(basename(file_name))/$n_group_name/$method_name/$r_group_name"
        )
    size(energy_trajectories, 1) == length(energy_mean) ||
        error(
            "energy trajectory row count $(size(energy_trajectories, 1)) does not " *
            "match energy length $(length(energy_mean)) in " *
            "$(basename(file_name))/$n_group_name/$method_name/$r_group_name"
        )
    overlap_trajectories = read_overlap_trajectory_matrix(run_group, length(energy_mean), M)
    size(overlap_trajectories, 2) == M ||
        error(
            "overlap trajectory column count $(size(overlap_trajectories, 2)) does not " *
            "match M=$M in $(basename(file_name))/$n_group_name/$method_name/$r_group_name"
        )
    size(overlap_trajectories, 1) == length(energy_mean) ||
        error(
            "overlap trajectory row count $(size(overlap_trajectories, 1)) does not " *
            "match energy length $(length(energy_mean)) in " *
            "$(basename(file_name))/$n_group_name/$method_name/$r_group_name"
        )
    relative_energy_mean = vec(Float64.(read(run_group[RESULT_RELATIVE_ENERGY])))
    inferred_completed_steps = max(length(energy_mean) - 1, 0)
    default_requested_steps = haskey(root, LARGE_N_ROOT_STEPS_KEY) ?
        Int(read(root[LARGE_N_ROOT_STEPS_KEY])) :
        inferred_completed_steps
    requested_steps_values = read_integer_vector(
        run_group,
        RESULT_REQUESTED_STEPS,
        fill(default_requested_steps, M),
    )
    completed_steps_values = read_integer_vector(
        run_group,
        RESULT_COMPLETED_STEPS,
        fill(inferred_completed_steps, M),
    )
    schedule = schedule_label(root, method_group, run_group)
    completed_requested_periods = completed_requested_periods_label(
        completed_steps_values, requested_steps_values, R, schedule
    )
    visited_delta_counts = distinct_completed_delta_counts(
        delta_history_matrix(run_group), completed_steps_values
    )
    visited_detunings = visited_detunings_label_from_counts(visited_delta_counts, R)
    missing_delta_history_count = M - length(visited_delta_counts)
    missing_delta_history_count >= 0 ||
        error(
            "visited detuning count length $(length(visited_delta_counts)) exceeds M=$M " *
            "in $(basename(file_name))/$n_group_name/$method_name/$r_group_name"
        )
    detuning_coverage = detuning_coverage_status(
        completed_steps_values, requested_steps_values, R, schedule
    )
    elapsed_values = read_float_vector(run_group, LARGE_N_ELAPSED_SECONDS_KEY)
    elapsed_seconds = isempty(elapsed_values) ? NaN : sum(elapsed_values)
    traj_cycles_per_hour = trajectory_cycles_per_hour(
        completed_steps_values, elapsed_seconds
    )
    stop_reasons = read_string_vector(run_group, LARGE_N_STOP_REASONS_KEY)
    if isempty(stop_reasons)
        stop_reasons = fill("", M)
    elseif length(stop_reasons) != M
        error(
            "stop_reasons length $(length(stop_reasons)) does not match M=$M " *
            "in $(basename(file_name))/$n_group_name/$method_name/$r_group_name"
        )
    end
    system_max_bond = bond_history_matrix(read(run_group[LARGE_N_SYSTEM_MAX_BOND_KEY]))
    system_mean_bond = bond_history_matrix(read(run_group[LARGE_N_SYSTEM_MEAN_BOND_KEY]))
    evolved_max_bond = bond_history_matrix(read(run_group[LARGE_N_EVOLVED_MAX_BOND_KEY]))
    evolved_mean_bond = bond_history_matrix(read(run_group[LARGE_N_EVOLVED_MEAN_BOND_KEY]))
    tdvp_sweep_dataset_present = haskey(run_group, LARGE_N_TDVP_SWEEP_MAX_BOND_KEY)
    tdvp_sweep_max_bond = tdvp_sweep_dataset_present ?
        bond_history_matrix(read(run_group[LARGE_N_TDVP_SWEEP_MAX_BOND_KEY])) :
        Matrix{Int}(undef, 0, 0)
    tdvp_sweep_saturation_dataset_cycle = first_saturation_from_dataset(
        run_group, LARGE_N_TDVP_SWEEP_SATURATION_CYCLE_KEY
    )
    has_tdvp_sweep_history =
        !isempty(tdvp_sweep_max_bond) &&
        (any(value -> value != 0, tdvp_sweep_max_bond) ||
         tdvp_sweep_saturation_dataset_cycle > 0)

    initial_e_over_n = energy_mean[1] / N
    initial_relative_energy = relative_energy_mean[1]
    initial_overlap = mean(vec(overlap_trajectories[1, :]))
    initial_e_over_n_values = vec(energy_trajectories[1, :]) ./ N
    initial_relative_energy_values = isfinite(E0) ?
        relative_energy.(vec(energy_trajectories[1, :]), Ref(E0)) :
        fill(initial_relative_energy, M)
    initial_overlap_values = vec(overlap_trajectories[1, :])
    final_e_over_n = energy_mean[end] / N
    final_relative_energy = relative_energy_mean[end]
    best_e_over_n = minimum(energy_mean) / N
    best_relative_energy = minimum(relative_energy_mean)
    tail_start = energy_tail_start(length(energy_mean))
    tail_count = length(energy_mean) - tail_start + 1
    tail_e_over_n = mean(energy_mean[tail_start:end]) / N
    tail_relative_energy = mean(relative_energy_mean[tail_start:end])
    final_e_over_n_values = vec(energy_trajectories[end, :]) ./ N
    final_relative_energy_values = isfinite(E0) ?
        relative_energy.(vec(energy_trajectories[end, :]), Ref(E0)) :
        fill(final_relative_energy, M)
    best_energy_values = vec(minimum(energy_trajectories; dims=1))
    best_e_over_n_values = best_energy_values ./ N
    best_relative_energy_values = isfinite(E0) ?
        relative_energy.(best_energy_values, Ref(E0)) :
        fill(best_relative_energy, M)
    tail_mean_energy_values = Float64[
        mean(view(energy_trajectories, tail_start:size(energy_trajectories, 1), trajectory))
        for trajectory in axes(energy_trajectories, 2)
    ]
    tail_e_over_n_values = tail_mean_energy_values ./ N
    tail_relative_energy_values = isfinite(E0) ?
        Float64[
            mean(relative_energy.(
                view(energy_trajectories, tail_start:size(energy_trajectories, 1), trajectory),
                Ref(E0),
            ))
            for trajectory in axes(energy_trajectories, 2)
        ] :
        fill(tail_relative_energy, M)
    final_system_max = final_system_max_bond(system_max_bond)
    final_system_mean = final_system_mean_bond(system_mean_bond)
    peak_evolved_max = peak_evolved_max_bond(evolved_max_bond)
    peak_evolved_mean = peak_evolved_mean_bond(evolved_mean_bond)
    peak_tdvp_sweep_max = has_tdvp_sweep_history ?
        maximum(tdvp_sweep_max_bond) :
        missing

    system_saturation_cycle = first_saturation_from_dataset(
        run_group, LARGE_N_SYSTEM_SATURATION_CYCLE_KEY
    )
    system_saturation_cycle == 0 &&
        (system_saturation_cycle = first_saturation_from_history(system_max_bond, threshold))
    evolved_saturation_cycle = first_saturation_from_dataset(
        run_group, LARGE_N_EVOLVED_SATURATION_CYCLE_KEY
    )
    evolved_saturation_cycle == 0 &&
        (evolved_saturation_cycle = first_saturation_from_history(evolved_max_bond, threshold))
    tdvp_sweep_saturation_cycle_for_status = 0
    tdvp_sweep_saturation_cycle = if has_tdvp_sweep_history
        cycle = tdvp_sweep_saturation_dataset_cycle
        cycle == 0 && (cycle = first_saturation_from_history(tdvp_sweep_max_bond, threshold))
        tdvp_sweep_saturation_cycle_for_status = cycle
        cycle
    else
        missing
    end
    system_effective_bond = effective_bond_dimension_label(
        final_system_max, system_saturation_cycle, threshold
    )
    evolved_effective_bond = effective_bond_dimension_label(
        peak_evolved_max, evolved_saturation_cycle, threshold
    )
    tdvp_sweep_effective_bond = has_tdvp_sweep_history ?
        effective_bond_dimension_label(
            peak_tdvp_sweep_max, tdvp_sweep_saturation_cycle, threshold
        ) :
        LARGE_N_LABEL_NA
    truncation_error_status = truncation_error_history_status(run_group)
    bond_status = require_largeN_bond_status_label(
        bond_cap_status(
            system_saturation_cycle, evolved_saturation_cycle,
            tdvp_sweep_saturation_cycle_for_status,
        )
    )

    link_dims = final_link_dimensions(run_group)
    quantiles = mean_link_quantiles(link_dims)
    fractions = mean_link_threshold_fractions(link_dims, threshold)
    detuning = detuning_protocol_summary(method_group, run_group)
    mode_summary = mode_reconstruction_summary(
        root, run_group, N, energy_mean; energy_name=energy_dataset.name
    )

    return (
        source_files=(basename(file_name),),
        E0=E0,
        trajectory_indices=trajectory_indices,
        trajectory_seeds=trajectory_seeds,
        completed_steps_values=completed_steps_values,
        requested_steps_values=requested_steps_values,
        visited_delta_counts=visited_delta_counts,
        missing_delta_history_count=missing_delta_history_count,
        elapsed_values=elapsed_values,
        stop_reason_values=stop_reasons,
        initial_e_over_n_values=initial_e_over_n_values,
        initial_relative_energy_values=initial_relative_energy_values,
        initial_overlap_values=initial_overlap_values,
        final_e_over_n_values=final_e_over_n_values,
        final_relative_energy_values=final_relative_energy_values,
        best_e_over_n_values=best_e_over_n_values,
        best_relative_energy_values=best_relative_energy_values,
        tail_e_over_n_values=tail_e_over_n_values,
        tail_relative_energy_values=tail_relative_energy_values,
        tail_count_values=fill(tail_count, M),
        file=basename(file_name),
        N=N,
        method=method_name,
        evolution=evolution,
        te=te,
        g=g,
        randomize_times=randomize_times,
        time_protocol=time_protocol,
        init_state=init_state,
        theta=theta,
        init_protocol=init_protocol,
        R=R,
        M=M,
        schedule=schedule,
        completed_requested="$(range_label(completed_steps_values))/$(range_label(requested_steps_values))",
        completed_requested_periods=completed_requested_periods,
        visited_detunings=visited_detunings,
        detuning_coverage=detuning_coverage,
        elapsed_total_seconds=elapsed_seconds,
        traj_cycles_per_hour=traj_cycles_per_hour,
        stop_reason=stop_reason_label(stop_reasons),
        delta_protocol=detuning.delta_protocol,
        delta_range=detuning.delta_range,
        delta_factor=detuning.delta_factor,
        threshold=threshold,
        initial_e_over_n=initial_e_over_n,
        initial_relative_energy=initial_relative_energy,
        initial_overlap=initial_overlap,
        final_e_over_n=final_e_over_n,
        relative_energy=final_relative_energy,
        best_e_over_n=best_e_over_n,
        best_relative_energy=best_relative_energy,
        tail_e_over_n=tail_e_over_n,
        tail_relative_energy=tail_relative_energy,
        tail_count=tail_count,
        mode_gF=mode_summary.mode_gF,
        mode_gF_source=mode_summary.mode_gF_source,
        mode_measured_rows=mode_summary.mode_measured_rows,
        mode_last_measured_e_over_n=mode_summary.mode_last_measured_e_over_n,
        mode_last_measured_abs_err_over_n=mode_summary.mode_last_measured_abs_err_over_n,
        mode_max_abs_err_over_n=mode_summary.mode_max_abs_err_over_n,
        system_effective_bond=system_effective_bond,
        evolved_effective_bond=evolved_effective_bond,
        tdvp_sweep_effective_bond=tdvp_sweep_effective_bond,
        truncation_error_history_status=truncation_error_status,
        bond_status=bond_status,
        final_system_max=final_system_max,
        final_system_mean=final_system_mean,
        peak_evolved_max=peak_evolved_max,
        peak_evolved_mean=peak_evolved_mean,
        peak_tdvp_sweep_max=peak_tdvp_sweep_max,
        system_saturation_cycle=system_saturation_cycle,
        evolved_saturation_cycle=evolved_saturation_cycle,
        tdvp_sweep_saturation_cycle=tdvp_sweep_saturation_cycle,
        q50=quantiles[1],
        q75=quantiles[2],
        q90=quantiles[3],
        q95=quantiles[4],
        frac50=fractions[1],
        frac75=fractions[2],
        frac90=fractions[3],
    )
end

function summarize_file(path::AbstractString)
    rows = NamedTuple[]
    h5open(path, "r") do root
        for n_group_name in sort(String.(keys(root)))
            is_largeN_n_group_name(n_group_name) || continue
            root[n_group_name] isa HDF5.Group || continue
            haskey(root[n_group_name], LARGE_N_SYSTEM_SIZE_KEY) || continue
            n_group = root[n_group_name]
            for method_name in sort(String.(keys(n_group)))
                n_group[method_name] isa HDF5.Group || continue
                method_group = n_group[method_name]
                for r_group_name in sort(
                    String.(keys(method_group));
                    by=name -> (
                        is_largeN_r_group_name(name) ?
                            largeN_r_from_group_name(name) :
                            typemax(Int)
                    ),
                )
                    is_largeN_r_group_name(r_group_name) || continue
                    method_group[r_group_name] isa HDF5.Group || continue
                    push!(
                        rows,
                        summarize_run(path, root, n_group_name, n_group, method_name,
                                      method_group, r_group_name, method_group[r_group_name]),
                    )
                end
            end
        end
    end
    return rows
end

function trajectory_ensemble_key(row)
    return (
        row.N,
        row.method,
        row.evolution,
        isfinite(row.te) ? row.te : missing,
        isfinite(row.g) ? row.g : missing,
        row.randomize_times,
        initial_state_group_key(row),
        row.R,
        row.schedule,
        row.delta_protocol,
        row.delta_range,
        row.delta_factor,
        row.threshold,
        isnan(row.E0) ? missing : row.E0,
    )
end

function concatenate_field(rows, field::Symbol)
    values = Any[]
    for row in rows
        append!(values, collect(getfield(row, field)))
    end
    return values
end

function concatenate_int_field(rows, field::Symbol)
    return Int[value for value in concatenate_field(rows, field)]
end

function concatenate_optional_int_field(rows, field::Symbol)
    values = Union{Missing,Int}[]
    for value in concatenate_field(rows, field)
        push!(values, ismissing(value) ? missing : Int(value))
    end
    return values
end

function concatenate_float_field(rows, field::Symbol)
    return Float64[value for value in concatenate_field(rows, field)]
end

function concatenate_string_field(rows, field::Symbol)
    return String[value for value in concatenate_field(rows, field)]
end

function weighted_row_mean(rows, field::Symbol)
    weights = Int[row.M for row in rows]
    values = Float64[getfield(row, field) for row in rows]
    return sum(values .* weights) / sum(weights)
end

function combined_string_status(values::AbstractVector{<:AbstractString})
    isempty(values) && return LARGE_N_LABEL_NA
    unique_values = sort(unique(values))
    length(unique_values) == 1 && return only(unique_values)
    return "mixed:" * join(unique_values, "+")
end

function first_saturation_from_rows(rows, field::Symbol)
    cycles = Int[
        getfield(row, field)
        for row in rows
        if !(getfield(row, field) isa Missing) && getfield(row, field) > 0
    ]
    return isempty(cycles) ? 0 : minimum(cycles)
end

function maximum_or_missing(values)
    present = [value for value in values if !(value isa Missing)]
    isempty(present) && return missing
    return maximum(Int.(present))
end

function trajectory_indices_label(indices::AbstractVector{<:Integer})
    isempty(indices) && return LARGE_N_LABEL_NONE
    return join(sort(Int.(indices)), ",")
end

function sorted_trajectory_seed_pairs(indices::AbstractVector{<:Integer},
                                      seeds::AbstractVector)
    # HDF5 stores trajectory indices and seeds as positionally paired vectors.
    # Sort pairs together so the printed `traj` and `seed` columns stay aligned.
    length(indices) == length(seeds) ||
        error(
            "trajectory seed count $(length(seeds)) does not match " *
            "trajectory index count $(length(indices))"
        )
    pairs = collect(zip(Int.(indices), seeds))
    return sort(pairs; by=first)
end

function trajectory_seeds_label(indices::AbstractVector{<:Integer}, seeds::AbstractVector)
    isempty(seeds) && return LARGE_N_LABEL_NONE
    pairs = sorted_trajectory_seed_pairs(indices, seeds)
    return join(
        [
            ismissing(seed) ? LARGE_N_LABEL_LEGACY_MISSING : string(Int(seed))
            for (_, seed) in pairs
        ],
        ",",
    )
end

function trajectory_seeds_label(method_name::AbstractString,
                                indices::AbstractVector{<:Integer},
                                seeds::AbstractVector)
    isempty(seeds) && return trajectory_seeds_label(indices, seeds)
    if largeN_method_kind_from_name(method_name) != :mcwf && all(ismissing, seeds)
        return LARGE_N_LABEL_NA
    end
    return trajectory_seeds_label(indices, seeds)
end

"""
    combine_trajectory_bucket(rows)

Combine file-level rows that describe the same physical protocol but disjoint
stored MCWF trajectory labels.  A singleton bucket is returned unchanged.  For a
multi-file bucket, `final_e_over_n`, `relative_energy`, `best_e_over_n`, and the
tail energy columns are means of the corresponding per-trajectory summary
statistics.  The same convention is used for initial-row diagnostics, so
`initial_e_over_n` is the mean of per-trajectory initial energies.  Thus
`best_e_over_n` is `mean_i min_t E_i(t)/N`, not
`min_t mean_i E_i(t)/N`, and unequal stop-on-cap prefixes are not promoted to a
cycle-aligned ensemble history.  Bond-cap status, detuning coverage, stop
reasons, and elapsed-time throughput are recomputed from the concatenated
trajectory metadata.
"""
function combine_trajectory_bucket(rows)
    length(rows) == 1 && return only(rows)

    rows = sort(rows; by=row -> minimum(row.trajectory_indices))
    indices = concatenate_int_field(rows, :trajectory_indices)
    seeds = concatenate_optional_int_field(rows, :trajectory_seeds)
    unique_indices = unique(indices)
    length(unique_indices) == length(indices) ||
        error(
            "cannot combine split trajectory summaries with duplicate trajectory_indices: " *
            trajectory_indices_label(indices)
        )
    length(seeds) == length(indices) ||
        error(
            "combined trajectory seed count $(length(seeds)) does not match " *
            "trajectory count $(length(indices))"
        )

    base = first(rows)
    completed_steps = concatenate_int_field(rows, :completed_steps_values)
    requested_steps = concatenate_int_field(rows, :requested_steps_values)
    visited_delta_counts = concatenate_int_field(rows, :visited_delta_counts)
    missing_delta_history_count = sum(Int[row.missing_delta_history_count for row in rows])
    length(visited_delta_counts) + missing_delta_history_count == length(indices) ||
        error(
            "combined detuning-history accounting does not match trajectory count: " *
            "$(length(visited_delta_counts)) known + $(missing_delta_history_count) unknown " *
            "for $(length(indices)) trajectories"
        )
    elapsed_values = concatenate_float_field(rows, :elapsed_values)
    elapsed_seconds = all(row -> !isempty(row.elapsed_values), rows) ?
        sum(elapsed_values) :
        NaN
    final_e_values = concatenate_float_field(rows, :final_e_over_n_values)
    final_relative_values = concatenate_float_field(rows, :final_relative_energy_values)
    initial_e_values = concatenate_float_field(rows, :initial_e_over_n_values)
    initial_relative_values = concatenate_float_field(rows, :initial_relative_energy_values)
    initial_overlap_values = concatenate_float_field(rows, :initial_overlap_values)
    best_e_values = concatenate_float_field(rows, :best_e_over_n_values)
    best_relative_values = concatenate_float_field(rows, :best_relative_energy_values)
    tail_e_values = concatenate_float_field(rows, :tail_e_over_n_values)
    tail_relative_values = concatenate_float_field(rows, :tail_relative_energy_values)
    tail_count_values = concatenate_int_field(rows, :tail_count_values)
    stop_reasons = concatenate_string_field(rows, :stop_reason_values)
    source_files = Tuple(concatenate_string_field(rows, :source_files))

    system_saturation_cycle = first_saturation_from_rows(rows, :system_saturation_cycle)
    evolved_saturation_cycle = first_saturation_from_rows(rows, :evolved_saturation_cycle)
    has_tdvp_sweep = any(row -> !(row.peak_tdvp_sweep_max isa Missing), rows)
    tdvp_sweep_saturation_cycle = has_tdvp_sweep ?
        first_saturation_from_rows(rows, :tdvp_sweep_saturation_cycle) :
        missing
    tdvp_sweep_saturation_cycle_for_status =
        tdvp_sweep_saturation_cycle isa Missing ? 0 : tdvp_sweep_saturation_cycle

    final_system_max = maximum(Int[row.final_system_max for row in rows])
    peak_evolved_max = maximum(Int[row.peak_evolved_max for row in rows])
    peak_tdvp_sweep_max = maximum_or_missing([row.peak_tdvp_sweep_max for row in rows])

    return merge(
        base,
        (
            source_files=source_files,
            E0=base.E0,
            trajectory_indices=indices,
            trajectory_seeds=seeds,
            completed_steps_values=completed_steps,
            requested_steps_values=requested_steps,
            visited_delta_counts=visited_delta_counts,
            missing_delta_history_count=missing_delta_history_count,
            elapsed_values=elapsed_values,
            stop_reason_values=stop_reasons,
            initial_e_over_n_values=initial_e_values,
            initial_relative_energy_values=initial_relative_values,
            initial_overlap_values=initial_overlap_values,
            final_e_over_n_values=final_e_values,
            final_relative_energy_values=final_relative_values,
            best_e_over_n_values=best_e_values,
            best_relative_energy_values=best_relative_values,
            tail_e_over_n_values=tail_e_values,
            tail_relative_energy_values=tail_relative_values,
            tail_count_values=tail_count_values,
            file="trajectory_ensemble(traj=$(trajectory_indices_label(indices)))",
            M=length(indices),
            completed_requested="$(range_label(completed_steps))/$(range_label(requested_steps))",
            completed_requested_periods=completed_requested_periods_label(
                completed_steps, requested_steps, base.R, base.schedule
            ),
            visited_detunings=visited_detunings_label_from_counts(
                visited_delta_counts, base.R;
                unknown_count=missing_delta_history_count,
                total_count=length(indices),
            ),
            detuning_coverage=detuning_coverage_status(
                completed_steps, requested_steps, base.R, base.schedule
            ),
            elapsed_total_seconds=elapsed_seconds,
            traj_cycles_per_hour=trajectory_cycles_per_hour(
                completed_steps, elapsed_seconds
            ),
            stop_reason=stop_reason_label(stop_reasons),
            initial_e_over_n=mean(initial_e_values),
            initial_relative_energy=mean(initial_relative_values),
            initial_overlap=mean(initial_overlap_values),
            final_e_over_n=mean(final_e_values),
            relative_energy=mean(final_relative_values),
            best_e_over_n=mean(best_e_values),
            best_relative_energy=mean(best_relative_values),
            tail_e_over_n=mean(tail_e_values),
            tail_relative_energy=mean(tail_relative_values),
            tail_count=range_label(tail_count_values),
            mode_gF=missing,
            mode_gF_source=missing,
            mode_measured_rows=missing,
            mode_last_measured_e_over_n=missing,
            mode_last_measured_abs_err_over_n=missing,
            mode_max_abs_err_over_n=missing,
            system_effective_bond=effective_bond_dimension_label(
                final_system_max, system_saturation_cycle, base.threshold
            ),
            evolved_effective_bond=effective_bond_dimension_label(
                peak_evolved_max, evolved_saturation_cycle, base.threshold
            ),
            tdvp_sweep_effective_bond=has_tdvp_sweep ?
                effective_bond_dimension_label(
                    peak_tdvp_sweep_max,
                    tdvp_sweep_saturation_cycle,
                    base.threshold,
                ) :
                LARGE_N_LABEL_NA,
            truncation_error_history_status=combined_string_status(
                String[row.truncation_error_history_status for row in rows]
            ),
            bond_status=require_largeN_bond_status_label(
                bond_cap_status(
                    system_saturation_cycle,
                    evolved_saturation_cycle,
                    tdvp_sweep_saturation_cycle_for_status,
                )
            ),
            final_system_max=final_system_max,
            final_system_mean=weighted_row_mean(rows, :final_system_mean),
            peak_evolved_max=peak_evolved_max,
            peak_evolved_mean=weighted_row_mean(rows, :peak_evolved_mean),
            peak_tdvp_sweep_max=peak_tdvp_sweep_max,
            system_saturation_cycle=system_saturation_cycle,
            evolved_saturation_cycle=evolved_saturation_cycle,
            tdvp_sweep_saturation_cycle=tdvp_sweep_saturation_cycle,
            q50=weighted_row_mean(rows, :q50),
            q75=weighted_row_mean(rows, :q75),
            q90=weighted_row_mean(rows, :q90),
            q95=weighted_row_mean(rows, :q95),
            frac50=weighted_row_mean(rows, :frac50),
            frac75=weighted_row_mean(rows, :frac75),
            frac90=weighted_row_mean(rows, :frac90),
        ),
    )
end

function combine_trajectory_rows(rows)
    buckets = Dict{Any,Vector{NamedTuple}}()
    key_order = Any[]
    for row in rows
        key = trajectory_ensemble_key(row)
        if !haskey(buckets, key)
            buckets[key] = NamedTuple[]
            push!(key_order, key)
        end
        push!(buckets[key], row)
    end

    combined = NamedTuple[]
    for key in key_order
        push!(combined, combine_trajectory_bucket(buckets[key]))
    end
    return combined
end

function sorted_rows(rows)
    return sort(
        rows;
        by=row -> (
            row.N,
            row.method,
            row.evolution,
            isfinite(row.te) ? row.te : Inf,
            isfinite(row.g) ? row.g : Inf,
            row.randomize_times,
            initial_state_sort_key(row),
            row.R,
            row.file,
        ),
    )
end

function print_markdown(rows)
    println("| file | N | method | evolution | te | g | time protocol | init | R | M | traj | seed | schedule | completed/requested | completed/requested periods | visited detunings | detuning coverage | elapsed_total | traj cycles/hour | stop_reason | delta_protocol | delta_range | delta_factor | Dcap | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | truncation errors | initial E/N | initial relE | initial overlap | final E/N | relE | best E/N | best relE | tail E/N | tail relE | tail n | mode gF | mode source | mode rows | mode last-measured E/N | mode last-measured abs dE/N | mode max abs dE/N | final sys max | final sys mean | peak evolved max | peak evolved mean | peak tdvp sweep max | sys sat | evolved sat | tdvp sweep sat | q50 | q75 | q90 | q95 | frac_ge_0.5D | frac_ge_0.75D | frac_ge_0.9D |")
    println("|---|---:|---|---|---:|---:|---|---|---:|---:|---|---|---|---|---:|---:|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted_rows(rows)
        println(
            "| $(row.file) | $(row.N) | $(row.method) | $(row.evolution) | " *
            "$(format_float(row.te, 3)) | $(format_float(row.g, 12)) | " *
            "$(row.time_protocol) | $(row.init_protocol) | " *
            "$(row.R) | $(row.M) | " *
            "$(trajectory_indices_label(row.trajectory_indices)) | " *
            "$(trajectory_seeds_label(row.method, row.trajectory_indices, row.trajectory_seeds)) | " *
            "$(row.schedule) | $(row.completed_requested) | $(row.completed_requested_periods) | " *
            "$(row.visited_detunings) | $(row.detuning_coverage) | " *
            "$(format_float(row.elapsed_total_seconds, 1)) | " *
            "$(format_float(row.traj_cycles_per_hour, 2)) | $(row.stop_reason) | " *
            "$(row.delta_protocol) | $(row.delta_range) | $(row.delta_factor) | " *
            "$(row.threshold) | " *
            "$(row.system_effective_bond) | $(row.evolved_effective_bond) | " *
            "$(row.tdvp_sweep_effective_bond) | $(row.bond_status) | " *
            "$(row.truncation_error_history_status) | " *
            "$(format_float(row.initial_e_over_n, 8)) | " *
            "$(format_float(row.initial_relative_energy, 5)) | " *
            "$(format_float(row.initial_overlap, 5)) | " *
            "$(format_float(row.final_e_over_n, 8)) | $(format_float(row.relative_energy, 5)) | " *
            "$(format_float(row.best_e_over_n, 8)) | $(format_float(row.best_relative_energy, 5)) | " *
            "$(format_float(row.tail_e_over_n, 8)) | $(format_float(row.tail_relative_energy, 5)) | " *
            "$(row.tail_count) | " *
            "$(format_string_or_na(row.mode_gF)) | $(format_string_or_na(row.mode_gF_source)) | " *
            "$(format_string_or_na(row.mode_measured_rows)) | " *
            "$(format_float_or_na(row.mode_last_measured_e_over_n, 8)) | " *
            "$(format_error_float_or_na(row.mode_last_measured_abs_err_over_n, 3)) | " *
            "$(format_error_float_or_na(row.mode_max_abs_err_over_n, 3)) | " *
            "$(row.final_system_max) | $(format_float(row.final_system_mean, 2)) | " *
            "$(row.peak_evolved_max) | $(format_float(row.peak_evolved_mean, 2)) | " *
            "$(format_integer_or_na(row.peak_tdvp_sweep_max)) | " *
            "$(saturation_cycle_label(row.system_saturation_cycle)) | " *
            "$(saturation_cycle_label(row.evolved_saturation_cycle)) | " *
            "$(saturation_cycle_label(row.tdvp_sweep_saturation_cycle)) | " *
            "$(format_float(row.q50, 1)) | $(format_float(row.q75, 1)) | " *
            "$(format_float(row.q90, 1)) | $(format_float(row.q95, 1)) | " *
            "$(format_float(row.frac50, 2)) | $(format_float(row.frac75, 2)) | " *
            "$(format_float(row.frac90, 2)) |"
        )
    end
end

function print_compact_markdown(rows)
    println("| file | N | method | evolution | te | g | time protocol | init | R | M | traj | seed | schedule | completed/requested | completed/requested periods | visited detunings | detuning coverage | initial E/N | initial overlap | final E/N | best E/N | mode max abs dE/N | Dcap | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | truncation errors | elapsed_total | traj cycles/hour | stop_reason |")
    println("|---|---:|---|---|---:|---:|---|---|---:|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|")
    for row in sorted_rows(rows)
        println(
            "| $(row.file) | $(row.N) | $(row.method) | $(row.evolution) | " *
            "$(format_float(row.te, 3)) | $(format_float(row.g, 12)) | " *
            "$(row.time_protocol) | $(row.init_protocol) | " *
            "$(row.R) | $(row.M) | " *
            "$(trajectory_indices_label(row.trajectory_indices)) | " *
            "$(trajectory_seeds_label(row.method, row.trajectory_indices, row.trajectory_seeds)) | " *
            "$(row.schedule) | $(row.completed_requested) | " *
            "$(row.completed_requested_periods) | $(row.visited_detunings) | " *
            "$(row.detuning_coverage) | " *
            "$(format_float(row.initial_e_over_n, 8)) | " *
            "$(format_float(row.initial_overlap, 5)) | " *
            "$(format_float(row.final_e_over_n, 8)) | " *
            "$(format_float(row.best_e_over_n, 8)) | " *
            "$(format_error_float_or_na(row.mode_max_abs_err_over_n, 3)) | $(row.threshold) | " *
            "$(row.system_effective_bond) | $(row.evolved_effective_bond) | " *
            "$(row.tdvp_sweep_effective_bond) | $(row.bond_status) | " *
            "$(row.truncation_error_history_status) | " *
            "$(format_float(row.elapsed_total_seconds, 1)) | " *
            "$(format_float(row.traj_cycles_per_hour, 2)) | $(row.stop_reason) |"
        )
    end
end

function parse_args(args)
    compact = false
    combine_trajectories = false
    skip_invalid = false
    paths = String[]
    for arg in args
        if arg == "--compact"
            compact = true
        elseif arg == "--combine-trajectories"
            combine_trajectories = true
        elseif arg == "--skip-invalid"
            skip_invalid = true
        elseif startswith(arg, "--")
            throw(ArgumentError("unknown option: $arg"))
        else
            push!(paths, arg)
        end
    end
    return (
        paths=paths,
        compact=compact,
        combine_trajectories=combine_trajectories,
        skip_invalid=skip_invalid,
    )
end

skip_invalid_catches_error(err) = !(err isa InterruptException)

function summarize_largeN_bond_dimensions_main(args=ARGS)
    if isempty(args) || any(arg -> arg in ("-h", "--help"), args)
        usage()
        return isempty(args) ? 1 : 0
    end

    parsed = parse_args(args)
    rows = NamedTuple[]
    for path in parsed.paths
        isfile(path) || error("not a file: $path")
        try
            append!(rows, summarize_file(path))
        catch err
            parsed.skip_invalid && skip_invalid_catches_error(err) || rethrow()
            error_summary = first(split(sprint(showerror, err), '\n'))
            @warn "Skipping invalid large-N campaign input" path error=error_summary
        end
    end
    isempty(rows) && error("no large-N campaign runs found")
    parsed.combine_trajectories && (rows = combine_trajectory_rows(rows))
    parsed.compact ? print_compact_markdown(rows) : print_markdown(rows)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(summarize_largeN_bond_dimensions_main())
end
