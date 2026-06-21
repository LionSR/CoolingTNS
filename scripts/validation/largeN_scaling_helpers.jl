if !isdefined(@__MODULE__, :_COOLINGTNS_LARGEN_SCALING_HELPERS_INCLUDED)
const _COOLINGTNS_LARGEN_SCALING_HELPERS_INCLUDED = true

using Statistics

const LARGE_N_TRAJECTORY_SEED_N_STRIDE = 1_000_000
const LARGE_N_TRAJECTORY_SEED_R_STRIDE = 10_000
const LARGE_N_TRAJECTORY_SEED_RULE =
    "trajectory_seed = base_seed + 1_000_000*N + 10_000*R + trajectory; " *
    "valid for 1 <= R < 100 and 1 <= trajectory < 10000"

const LARGE_N_DETUNING_REFERENCE_SETUP_GAP = "setup_gap"
const LARGE_N_DETUNING_REFERENCE_ISING_MODE_PAIR = "ising_mode_pair_reference"
const LARGE_N_DETUNING_PROTOCOL_GAP_SCALED_RANGE = "gap_scaled_range"
const LARGE_N_DETUNING_PROTOCOL_FIXED_RANGE = "fixed_range"

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
    write(parent, "detuning_protocol_source", protocol.source)
    write(parent, "detuning_reference_gap", protocol.reference_gap)
    write(parent, "detuning_delta_min", protocol.delta_min)
    write(parent, "detuning_delta_max", protocol.delta_max)
    write(parent, "detuning_delta_max_factor", protocol.delta_max_factor)
    write(parent, "detuning_fixed_across_dmax", protocol.fixed_across_dmax)
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

saturation_cycle_label(cycle::Integer) = cycle == 0 ? "none" : string(cycle)
saturation_cycle_label(::Missing) = "n/a"

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
    system_saturation_cycle > 0 && push!(hit_sources, "system")
    evolved_saturation_cycle > 0 && push!(hit_sources, "evolved")
    tdvp_sweep_saturation_cycle > 0 && push!(hit_sources, "tdvp_sweep")
    isempty(hit_sources) && return "no_cap_hit"
    return "not_converged_$(join(hit_sources, "_and_"))_cap"
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
