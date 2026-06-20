"""
    multi_frequency_schedules.jl

Single source of truth for multi-frequency detuning schedule names.
"""

const MULTI_FREQUENCY_SCHEDULES = (:round_robin, :descending, :random)

"""
    validate_multi_frequency_schedule(schedule::Symbol) -> Symbol

Return `schedule` if it is a supported multi-frequency detuning schedule.
"""
function validate_multi_frequency_schedule(schedule::Symbol)::Symbol
    schedule in MULTI_FREQUENCY_SCHEDULES && return schedule
    throw(ArgumentError(
        "schedule must be one of $(join(string.(MULTI_FREQUENCY_SCHEDULES), ", ")), got $schedule",
    ))
end

"""
    parse_multi_frequency_schedule(schedule) -> Symbol

Parse and validate a schedule name from CLI or HDF5-style string metadata.
"""
parse_multi_frequency_schedule(schedule::Symbol)::Symbol =
    validate_multi_frequency_schedule(schedule)

parse_multi_frequency_schedule(schedule::AbstractString)::Symbol =
    validate_multi_frequency_schedule(Symbol(schedule))

"""
    multi_frequency_schedule_token(schedule) -> String

Short stable token for filenames that summarize a multi-frequency schedule.
"""
function multi_frequency_schedule_token(schedule)::String
    schedule = parse_multi_frequency_schedule(schedule)
    schedule === :round_robin && return "rr"
    schedule === :descending && return "desc"
    schedule === :random && return "rand"
    error("unreachable schedule token for $schedule")
end
