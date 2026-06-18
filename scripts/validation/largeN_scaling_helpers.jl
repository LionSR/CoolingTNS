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
