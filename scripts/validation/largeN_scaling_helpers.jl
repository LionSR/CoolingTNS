using Statistics

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

Return the largest transient system-bath bond dimension over evolved cooling
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
transient system-bath mean bond dimension.  The initial row is excluded because
it has no evolved state.
"""
function peak_evolved_mean_bond(evolved_meanbond)
    history = bond_history_matrix(evolved_meanbond)
    size(history, 1) >= 2 || return NaN
    return maximum(vec(mean(history[2:end, :]; dims=2)))
end
