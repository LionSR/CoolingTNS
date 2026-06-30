if !isdefined(@__MODULE__, :_COOLINGTNS_LARGEN_SCALING_HELPERS_INCLUDED)
const _COOLINGTNS_LARGEN_SCALING_HELPERS_INCLUDED = true

using Statistics

const LARGE_N_TRAJECTORY_SEED_N_STRIDE = 1_000_000
const LARGE_N_TRAJECTORY_SEED_R_STRIDE = 10_000
const LARGE_N_TRAJECTORY_SEED_RULE =
    "trajectory_seed = base_seed + 1_000_000*N + 10_000*R + trajectory; " *
    "valid for 1 <= R < 100 and 1 <= trajectory < 10000"

# Persisted HDF5 detuning metadata keys and values.
const LARGE_N_DETUNING_REFERENCE_GAP_SOURCE_KEY = "detuning_reference_gap_source"
const LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY = "detuning_protocol_source"
const LARGE_N_DETUNING_REFERENCE_GAP_KEY = "detuning_reference_gap"
const LARGE_N_DETUNING_DELTA_MIN_KEY = "detuning_delta_min"
const LARGE_N_DETUNING_DELTA_MAX_KEY = "detuning_delta_max"
const LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY = "detuning_delta_max_factor"
const LARGE_N_DETUNING_FIXED_ACROSS_DMAX_KEY = "detuning_fixed_across_dmax"

const LARGE_N_DETUNING_REFERENCE_SETUP_GAP = "setup_gap"
const LARGE_N_DETUNING_REFERENCE_ISING_MODE_PAIR = "ising_mode_pair_reference"
const LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE = "gap_scaled_range"
const LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE = "fixed_range"

# Generic reader-facing summary labels used by large-N validation tables.
# These distinguish absent, unknown, empty, and legacy provenance without
# changing the underlying numerical evidence.
const LARGE_N_LABEL_NA = "n/a"
const LARGE_N_LABEL_UNKNOWN = "unknown"
const LARGE_N_LABEL_NONE = "none"
const LARGE_N_LABEL_MISSING = "missing"
const LARGE_N_LABEL_LEGACY_MISSING = "legacy_missing"

# Persisted HDF5 evolution and bond-cap diagnostic keys for large-N campaigns.
const LARGE_N_EVOLUTION_METHOD_KEY = "evolution_method"
const LARGE_N_SYSTEM_SOLVE_REUSED_ACROSS_R_KEY = "system_solve_reused_across_R"
const LARGE_N_BOND_SATURATION_THRESHOLD_KEY = "bond_saturation_threshold"
const LARGE_N_SYSTEM_SATURATION_CYCLE_KEY = "system_saturation_cycle"
const LARGE_N_EVOLVED_SATURATION_CYCLE_KEY = "evolved_saturation_cycle"
const LARGE_N_SYSTEM_MAX_BOND_KEY = "system_max_bond"
const LARGE_N_SYSTEM_MEAN_BOND_KEY = "system_mean_bond"
const LARGE_N_EVOLVED_MAX_BOND_KEY = "evolved_max_bond"
const LARGE_N_EVOLVED_MEAN_BOND_KEY = "evolved_mean_bond"
const LARGE_N_TDVP_SWEEP_MAX_BOND_KEY = "tdvp_sweep_max_bond"
const LARGE_N_TDVP_SWEEP_SATURATION_CYCLE_KEY = "tdvp_sweep_saturation_cycle"
const LARGE_N_ELAPSED_SECONDS_KEY = "elapsed_seconds"
const LARGE_N_STOP_REASONS_KEY = "stop_reasons"
const LARGE_N_TRAJECTORY_SEED_RULE_KEY = "trajectory_seed_rule"
const LARGE_N_TRAJECTORY_SEEDS_KEY = "trajectory_seeds"
const LARGE_N_TRAJECTORY_INDICES_KEY = "trajectory_indices"
const LARGE_N_DELTA_LISTS_KEY = "delta_lists"
const LARGE_N_DELTA_LIST_FIRST_TRAJECTORY_KEY = "delta_list_first_trajectory"
const LARGE_N_DELTA_LIST_IS_COMMON_KEY = "delta_list_is_common"
const LARGE_N_TE_LISTS_KEY = "te_lists"
const LARGE_N_TE_LIST_FIRST_TRAJECTORY_KEY = "te_list_first_trajectory"
const LARGE_N_TE_LIST_IS_COMMON_KEY = "te_list_is_common"
const LARGE_N_FINAL_BOND_DIMS_GROUP = "final_bond_dims"
const LARGE_N_FINAL_BOND_DIMS_TRAJECTORY_PREFIX = "trajectory_"

# Reader-facing detuning-coverage labels.  HDF5 campaign summaries and
# interrupted progress-CSV summaries use these labels to distinguish a completed
# multi-frequency grid pass from a short prefix.
const LARGE_N_DETUNING_COVERAGE_NA = LARGE_N_LABEL_NA
const LARGE_N_DETUNING_COVERAGE_SINGLE_DETUNING = "single_detuning"
const LARGE_N_DETUNING_COVERAGE_FULL_GRID = "full_grid_observed"
const LARGE_N_DETUNING_COVERAGE_REQUESTED_PARTIAL_GRID = "requested_partial_grid"
const LARGE_N_DETUNING_COVERAGE_STOPPED_PARTIAL_GRID = "stopped_partial_grid"
const LARGE_N_DETUNING_COVERAGE_PARTIAL_GRID_OBSERVED = "partial_grid_observed"
const LARGE_N_DETUNING_COVERAGE_NO_COMPLETED_CYCLES = "no_completed_cycles"
const LARGE_N_DETUNING_COVERAGE_MISSING_DETUNING_VALUES = "missing_detuning_values"

# Internal trajectory-row keys used before ensemble aggregation.  They are not
# HDF5 dataset names, but keeping them adjacent to the persisted bond keys
# prevents writer-side drift in the large-N diagnostics.
const LARGE_N_ROW_SYSTEM_MAX_BOND_KEY = "sys_maxbond"
const LARGE_N_ROW_SYSTEM_MEAN_BOND_KEY = "sys_meanbond"
const LARGE_N_ROW_EVOLVED_MAX_BOND_KEY = "evolved_maxbond"
const LARGE_N_ROW_EVOLVED_MEAN_BOND_KEY = "evolved_meanbond"
const LARGE_N_ROW_TDVP_SWEEP_MAX_BOND_KEY = "tdvp_sweep_maxbond"

# Bond-cap status labels.  These strings are reader-facing evidence labels in
# the large-N HDF5 and progress-CSV summaries; keep their construction
# centralized so cap-limited runs cannot be mislabeled by one reader.
const LARGE_N_BOND_CAP_SOURCE_SYSTEM = "system"
const LARGE_N_BOND_CAP_SOURCE_EVOLVED = "evolved"
const LARGE_N_BOND_CAP_SOURCE_TDVP_SWEEP = "tdvp_sweep"
const LARGE_N_BOND_CAP_SOURCES = (
    LARGE_N_BOND_CAP_SOURCE_SYSTEM,
    LARGE_N_BOND_CAP_SOURCE_EVOLVED,
    LARGE_N_BOND_CAP_SOURCE_TDVP_SWEEP,
)
const LARGE_N_BOND_STATUS_NO_CAP_HIT = "no_cap_hit"
const LARGE_N_BOND_STATUS_PREFIX = "not_converged"
const LARGE_N_BOND_STATUS_SUFFIX = "cap"

function _largeN_bond_status_label(hit_sources)
    sources = String.(collect(hit_sources))
    isempty(sources) && return LARGE_N_BOND_STATUS_NO_CAP_HIT
    length(unique(sources)) == length(sources) || throw(ArgumentError(
        "large-N bond-cap sources must be unique; got $(join(sources, ", "))"
    ))
    unknown = [source for source in sources if !(source in LARGE_N_BOND_CAP_SOURCES)]
    isempty(unknown) || throw(ArgumentError(
        "unknown large-N bond-cap source(s): $(join(unknown, ", ")); expected one of " *
        join(LARGE_N_BOND_CAP_SOURCES, ", ")
    ))
    ordered = [
        source for source in LARGE_N_BOND_CAP_SOURCES
        if source in sources
    ]
    return "$(LARGE_N_BOND_STATUS_PREFIX)_$(join(ordered, "_and_"))_$(LARGE_N_BOND_STATUS_SUFFIX)"
end

const LARGE_N_BOND_STATUS_SYSTEM_CAP = _largeN_bond_status_label((
    LARGE_N_BOND_CAP_SOURCE_SYSTEM,
))
const LARGE_N_BOND_STATUS_EVOLVED_CAP = _largeN_bond_status_label((
    LARGE_N_BOND_CAP_SOURCE_EVOLVED,
))
const LARGE_N_BOND_STATUS_TDVP_SWEEP_CAP = _largeN_bond_status_label((
    LARGE_N_BOND_CAP_SOURCE_TDVP_SWEEP,
))
const LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_CAP = _largeN_bond_status_label((
    LARGE_N_BOND_CAP_SOURCE_SYSTEM,
    LARGE_N_BOND_CAP_SOURCE_EVOLVED,
))
const LARGE_N_BOND_STATUS_SYSTEM_AND_TDVP_SWEEP_CAP = _largeN_bond_status_label((
    LARGE_N_BOND_CAP_SOURCE_SYSTEM,
    LARGE_N_BOND_CAP_SOURCE_TDVP_SWEEP,
))
const LARGE_N_BOND_STATUS_EVOLVED_AND_TDVP_SWEEP_CAP = _largeN_bond_status_label((
    LARGE_N_BOND_CAP_SOURCE_EVOLVED,
    LARGE_N_BOND_CAP_SOURCE_TDVP_SWEEP,
))
const LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_AND_TDVP_SWEEP_CAP =
    _largeN_bond_status_label((
        LARGE_N_BOND_CAP_SOURCE_SYSTEM,
        LARGE_N_BOND_CAP_SOURCE_EVOLVED,
        LARGE_N_BOND_CAP_SOURCE_TDVP_SWEEP,
    ))

const LARGE_N_BOND_STATUSES = (
    LARGE_N_BOND_STATUS_NO_CAP_HIT,
    LARGE_N_BOND_STATUS_SYSTEM_CAP,
    LARGE_N_BOND_STATUS_EVOLVED_CAP,
    LARGE_N_BOND_STATUS_TDVP_SWEEP_CAP,
    LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_CAP,
    LARGE_N_BOND_STATUS_SYSTEM_AND_TDVP_SWEEP_CAP,
    LARGE_N_BOND_STATUS_EVOLVED_AND_TDVP_SWEEP_CAP,
    LARGE_N_BOND_STATUS_SYSTEM_AND_EVOLVED_AND_TDVP_SWEEP_CAP,
)

"""
    require_largeN_bond_status_label(status) -> String

Validate a machine-readable large-N bond-cap status label.

The status is not a cooling-success metric.  It records only which effective
bond-dimension histories reached the method-specific cap, so malformed labels
should be rejected before a summary table is used as physical evidence.
"""
function require_largeN_bond_status_label(status)
    label = String(status)
    label in LARGE_N_BOND_STATUSES && return label
    throw(ArgumentError(
        "unknown large-N bond status '$label'; expected one of " *
        join(LARGE_N_BOND_STATUSES, ", ")
    ))
end

# Progress CSV stage labels.  These labels define the physical meaning of a
# flushed observer row, and are shared by the campaign writer and recovery
# summarizer.
const LARGE_N_PROGRESS_STAGE_INITIAL = "initial"
const LARGE_N_PROGRESS_STAGE_PREPARED = "prepared"
const LARGE_N_PROGRESS_STAGE_EVOLVED = "evolved"
const LARGE_N_PROGRESS_STAGE_UPDATED = "updated"
const LARGE_N_PROGRESS_STAGE_TDVP_SWEEP = "tdvp_sweep"
const LARGE_N_PROGRESS_STAGES = (
    LARGE_N_PROGRESS_STAGE_INITIAL,
    LARGE_N_PROGRESS_STAGE_PREPARED,
    LARGE_N_PROGRESS_STAGE_EVOLVED,
    LARGE_N_PROGRESS_STAGE_UPDATED,
    LARGE_N_PROGRESS_STAGE_TDVP_SWEEP,
)

# Persisted progress CSV schema.  These columns are written by the large-N
# campaign driver and read by the interrupted-run recovery summarizer.
const LARGE_N_PROGRESS_CSV_COLUMNS = (
    "timestamp",
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
    "stage",
    "step",
    "cycle",
    "delta",
    "te",
    "energy_per_site",
    "relative_energy",
    "overlap",
    LARGE_N_SYSTEM_MAX_BOND_KEY,
    LARGE_N_SYSTEM_MEAN_BOND_KEY,
    LARGE_N_EVOLVED_MAX_BOND_KEY,
    LARGE_N_EVOLVED_MEAN_BOND_KEY,
    "tdvp_sweep",
    "tdvp_time",
    LARGE_N_ELAPSED_SECONDS_KEY,
)

"""
    largeN_progress_stage(stage::Symbol) -> String

Return the persisted progress-CSV stage label for a cooling observer stage.
Unknown observer stages are rejected explicitly so that recovery summaries do
not silently misclassify the physical row type.
"""
function largeN_progress_stage(stage::Symbol)
    stage === :initial && return LARGE_N_PROGRESS_STAGE_INITIAL
    stage === :prepared && return LARGE_N_PROGRESS_STAGE_PREPARED
    stage === :evolved && return LARGE_N_PROGRESS_STAGE_EVOLVED
    stage === :updated && return LARGE_N_PROGRESS_STAGE_UPDATED
    throw(ArgumentError("unknown large-N progress observer stage: $stage"))
end

"""
    require_largeN_progress_stage_label(stage) -> String

Return `stage` as a string after checking that it is one of the persisted
large-N progress CSV labels.  This validates external CSV input and direct
writer calls against the same label set used by `largeN_progress_stage`.
"""
function require_largeN_progress_stage_label(stage)
    label = String(stage)
    label in LARGE_N_PROGRESS_STAGES && return label
    throw(ArgumentError(
        "unknown large-N progress CSV stage '$label'; expected one of " *
        join(LARGE_N_PROGRESS_STAGES, ", ")
    ))
end

"""
    progress_detuning_coverage_status(visited_count, R, completed_cycles) -> String

Return a reader-facing detuning-coverage label for an interrupted progress trace
when only the completed update rows are available.  The progress CSV does not
store the originally requested number of cycles, so a nonzero prefix shorter
than one full grid is reported as `partial_grid_observed`.
"""
function progress_detuning_coverage_status(
    visited_count::Integer, R::Integer, completed_cycles::Integer,
)
    R > 0 || return LARGE_N_DETUNING_COVERAGE_NA
    completed_cycles <= 0 && return LARGE_N_DETUNING_COVERAGE_NO_COMPLETED_CYCLES
    visited_count <= 0 && return LARGE_N_DETUNING_COVERAGE_MISSING_DETUNING_VALUES
    R == 1 && return LARGE_N_DETUNING_COVERAGE_SINGLE_DETUNING
    visited_count >= R && return LARGE_N_DETUNING_COVERAGE_FULL_GRID
    return LARGE_N_DETUNING_COVERAGE_PARTIAL_GRID_OBSERVED
end

# Progress rows are grouped by the job identity columns when recovering an
# interrupted run from CSV.  The coupling strength `g` is part of this identity
# because serial `--g-values` scans may share one progress CSV path.  The per-row
# `te` value is intentionally not part of this key: randomized-time runs draw a
# different `te` per cycle, and fixed `--te-values` scans are emitted as separate
# planned jobs with distinct paths.
const LARGE_N_PROGRESS_GROUP_COLUMNS = (
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

"""
    largeN_trajectory_seed(base_seed, N, R, trajectory)

Return the deterministic seed used for one large-N trajectory.  The formula is
part of the on-disk provenance contract because two campaign files with the
same base seed and the same `(N,R,trajectory)` label should use the same
stochastic trajectory, independent of how other frequency counts are grouped.
"""
function largeN_trajectory_seed(base_seed::Integer, N::Integer, R::Integer,
                                trajectory::Integer)
    R >= 1 || throw(ArgumentError("R must be positive, got $R"))
    R < div(LARGE_N_TRAJECTORY_SEED_N_STRIDE, LARGE_N_TRAJECTORY_SEED_R_STRIDE) ||
        throw(ArgumentError("R must be less than 100 for the stored seed rule, got $R"))
    trajectory >= 1 ||
        throw(ArgumentError("trajectory must be positive, got $trajectory"))
    trajectory < LARGE_N_TRAJECTORY_SEED_R_STRIDE ||
        throw(ArgumentError(
            "trajectory must be less than $(LARGE_N_TRAJECTORY_SEED_R_STRIDE) " *
            "for the stored seed rule, got $trajectory"
        ))
    return Int(base_seed) + LARGE_N_TRAJECTORY_SEED_N_STRIDE * Int(N) +
           LARGE_N_TRAJECTORY_SEED_R_STRIDE * Int(R) + Int(trajectory)
end

"""
    largeN_detuning_protocol(gap; delta_min, delta_max, delta_max_factor)

Return the detuning interval used by a large-N validation campaign.  A fixed
range is an explicit physical protocol; otherwise the range is derived from the
DMRG gap estimate as `[gap, delta_max_factor * gap]`.  A fixed range records
the reference gap estimate for provenance, but does not use it to define the
detunings.
"""
function largeN_detuning_protocol(
    gap::Real;
    delta_min=nothing,
    delta_max=nothing,
    delta_max_factor::Real=6.0,
)
    if (delta_min === nothing) != (delta_max === nothing)
        throw(ArgumentError("delta_min and delta_max must be supplied together"))
    end

    reference_gap = Float64(gap)
    isfinite(reference_gap) ||
        throw(ArgumentError("reference gap must be finite, got $gap"))

    if delta_min === nothing
        reference_gap > 0 ||
            throw(ArgumentError("reference gap must be positive for a gap-scaled detuning range, got $gap"))
        factor = Float64(delta_max_factor)
        isfinite(factor) && factor >= 1 ||
            throw(ArgumentError("delta_max_factor must be finite and at least 1, got $factor"))
        return (
            source=LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE,
            reference_gap=reference_gap,
            delta_min=reference_gap,
            delta_max=factor * reference_gap,
            delta_max_factor=factor,
            fixed_across_dmax=false,
        )
    end

    delta_min_value = Float64(delta_min)
    delta_max_value = Float64(delta_max)
    isfinite(delta_min_value) &&
        isfinite(delta_max_value) &&
        delta_min_value > 0 &&
        delta_max_value >= delta_min_value ||
        throw(ArgumentError("fixed detuning range must satisfy 0 < delta_min <= delta_max"))
    return (
        source=LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE,
        reference_gap=reference_gap,
        delta_min=delta_min_value,
        delta_max=delta_max_value,
        delta_max_factor=NaN,
        fixed_across_dmax=true,
    )
end

"""
    largeN_detuning_protocol(gap, cfg)

Construct the large-N detuning protocol from parsed campaign-driver options.
"""
function largeN_detuning_protocol(gap::Real, cfg)
    return largeN_detuning_protocol(
        gap;
        delta_min=cfg["delta_min"],
        delta_max=cfg["delta_max"],
        delta_max_factor=cfg["delta_max_factor"],
    )
end

"""
    largeN_delta_values(protocol, R)

Return the `R` bath detunings for a stored large-N protocol.  For `R=1`, the
single detuning is the low end of the interval; this matches the previous
single-frequency convention used by the validation driver.
"""
function largeN_delta_values(protocol, R::Integer)
    R >= 1 || throw(ArgumentError("R must be positive, got $R"))
    R == 1 && return [protocol.delta_min]
    return collect(range(protocol.delta_min, protocol.delta_max; length=Int(R)))
end

"""
    largeN_method_kind_from_name(method_name)

Normalize the large-N diagnostic method name stored in HDF5 files and progress
CSVs.  This helper intentionally does not import `CoolingTNS`, so the
interrupted-run progress CSV summarizer can remain lightweight.
"""
function largeN_method_kind_from_name(method_name::AbstractString)
    normalized = lowercase(method_name)
    normalized == "mcwf" && return :mcwf
    normalized == "mpo" && return :mpo
    throw(ArgumentError(
        "unknown large-N method '$method_name'; expected 'mcwf' or 'mpo'"
    ))
end

"""
    write_largeN_detuning_protocol(parent, protocol)

Write the on-disk HDF5 metadata contract for a large-N detuning protocol:
`detuning_protocol_source`, `detuning_reference_gap`, `detuning_delta_min`,
`detuning_delta_max`, `detuning_delta_max_factor`, and
`detuning_fixed_across_dmax`.
"""
function write_largeN_detuning_protocol(parent, protocol)
    write(parent, LARGE_N_DETUNING_PROTOCOL_SOURCE_KEY, protocol.source)
    write(parent, LARGE_N_DETUNING_REFERENCE_GAP_KEY, protocol.reference_gap)
    write(parent, LARGE_N_DETUNING_DELTA_MIN_KEY, protocol.delta_min)
    write(parent, LARGE_N_DETUNING_DELTA_MAX_KEY, protocol.delta_max)
    write(parent, LARGE_N_DETUNING_DELTA_MAX_FACTOR_KEY, protocol.delta_max_factor)
    write(parent, LARGE_N_DETUNING_FIXED_ACROSS_DMAX_KEY, protocol.fixed_across_dmax)
end

"""
    first_bond_saturation_cycle(maxbond, saturation_threshold)

Return the first 1-based cooling cycle whose recorded maximum bond dimension
reaches `saturation_threshold`.  The first entry of `maxbond` is the
pre-cooling initial measurement, so it is skipped.  Return `0` if the threshold
is never reached.
"""
function first_bond_saturation_cycle(maxbond::AbstractVector{<:Integer},
                                     saturation_threshold::Integer)
    for idx in 2:length(maxbond)
        maxbond[idx] >= saturation_threshold && return idx - 1
    end
    return 0
end

function first_recorded_saturation_cycle(cycles::AbstractVector{<:Integer})
    recorded = filter(>(0), cycles)
    return isempty(recorded) ? 0 : minimum(recorded)
end

saturation_cycle_label(cycle::Integer) = cycle == 0 ? LARGE_N_LABEL_NONE : string(cycle)
saturation_cycle_label(::Missing) = LARGE_N_LABEL_NA

"""
    bond_cap_status(system_saturation_cycle, evolved_saturation_cycle,
                    tdvp_sweep_saturation_cycle=0)

Return a machine-readable bond-cap diagnostic for a large-N cooling run.  This
status is only a bond-dimension statement: `no_cap_hit` does not imply energy
or trajectory convergence.
"""
function bond_cap_status(system_saturation_cycle::Integer,
                         evolved_saturation_cycle::Integer,
                         tdvp_sweep_saturation_cycle::Integer=0)
    hit_sources = String[]
    system_saturation_cycle > 0 && push!(hit_sources, LARGE_N_BOND_CAP_SOURCE_SYSTEM)
    evolved_saturation_cycle > 0 && push!(hit_sources, LARGE_N_BOND_CAP_SOURCE_EVOLVED)
    tdvp_sweep_saturation_cycle > 0 &&
        push!(hit_sources, LARGE_N_BOND_CAP_SOURCE_TDVP_SWEEP)
    return _largeN_bond_status_label(hit_sources)
end

"""
    effective_bond_dimension_label(observed_max, saturation_cycle, saturation_threshold)

Return a conservative effective-bond-dimension label.  If the run reached the
method-specific saturation threshold, the observed bond dimension is only a
lower bound on the converged bond dimension, so the label is written as
`>=D`.  Otherwise the largest observed bond dimension is reported directly.
"""
function effective_bond_dimension_label(observed_max::Integer,
                                        saturation_cycle::Integer,
                                        saturation_threshold::Integer)
    if saturation_cycle > 0
        return ">=$(max(observed_max, saturation_threshold))"
    end
    return string(observed_max)
end

"""
    bond_dimension_quantiles(link_dims, probabilities)

Return quantiles of a final MPS/MPO link-dimension vector as `Float64` values.
An empty link list returns `NaN` for every requested probability.
"""
function bond_dimension_quantiles(link_dims::AbstractVector{<:Integer},
                                  probabilities::AbstractVector{<:Real})
    isempty(link_dims) && return fill(NaN, length(probabilities))
    return quantile(Float64.(link_dims), Float64.(probabilities))
end

"""
    bond_dimension_fraction_at_least(link_dims, threshold)

Return the fraction of final links whose dimension is at least `threshold`.
An empty link list returns `NaN`.
"""
function bond_dimension_fraction_at_least(link_dims::AbstractVector{<:Integer},
                                          threshold::Real)
    isempty(link_dims) && return NaN
    return count(d -> d >= threshold, link_dims) / length(link_dims)
end

"""
    bond_dimension_threshold_fractions(link_dims, saturation_threshold, fractions)

For each `fraction`, return the fraction of final links whose dimension is at
least `fraction * saturation_threshold`.  These are the `frac_ge_*D` entries in
the large-N bond-dimension summary table.
"""
function bond_dimension_threshold_fractions(link_dims::AbstractVector{<:Integer},
                                            saturation_threshold::Real,
                                            fractions::AbstractVector{<:Real})
    return [
        bond_dimension_fraction_at_least(link_dims, fraction * saturation_threshold)
        for fraction in fractions
    ]
end

"""
    bond_history_matrix(history)

Return a bond-history array as a `steps x trajectories` matrix.  A vector is
interpreted as one trajectory.
"""
bond_history_matrix(history) = ndims(history) == 1 ? reshape(history, :, 1) : history

"""
    final_system_max_bond(system_maxbond)

Return the largest final system-state bond dimension over all trajectories.
"""
function final_system_max_bond(system_maxbond)
    history = bond_history_matrix(system_maxbond)
    return maximum(history[end, :])
end

"""
    final_system_mean_bond(system_meanbond)

Return the trajectory average of the final system-state mean bond dimension.
"""
function final_system_mean_bond(system_meanbond)
    history = bond_history_matrix(system_meanbond)
    return mean(history[end, :])
end

"""
    peak_evolved_max_bond(evolved_maxbond)

Return the largest evolved system-bath bond dimension over evolved cooling
steps, excluding the initial row that has no evolved state.
"""
function peak_evolved_max_bond(evolved_maxbond)
    history = bond_history_matrix(evolved_maxbond)
    size(history, 1) >= 2 || return 0
    return maximum(history[2:end, :])
end

"""
    peak_evolved_mean_bond(evolved_meanbond)

Return the peak, over evolved cooling steps, of the trajectory-averaged
evolved system-bath mean bond dimension.  The initial row is excluded because
it has no evolved state.
"""
function peak_evolved_mean_bond(evolved_meanbond)
    history = bond_history_matrix(evolved_meanbond)
    size(history, 1) >= 2 || return NaN
    return maximum(vec(mean(history[2:end, :]; dims=2)))
end

end
