#!/usr/bin/env julia
"""
Summarize large-N progress CSV files written by
`run_largeN_multifrequency_tn_scaling.jl`.

This script is meant for interrupted MCWF+TDVP runs where the HDF5 summary may
not have been written.  It reads the flushed progress CSV and reports the
energy trace, cap onset, and per-sweep TDVP timing.

Example:

    julia --project=. scripts/validation/summarize_tdvp_progress_csv.jl \
        /tmp/coolingtns_tdvp_progress/progress.csv
"""

module TDVPProgressCSVSummary

using Printf

include(joinpath(@__DIR__, "largeN_scaling_helpers.jl"))

const PROGRESS_GROUP_COLUMNS = [
    "N",
    "method",
    "evolution",
    "R",
    "trajectory",
    "seed",
    "Dmax",
    "cutoff",
    "tau",
]

function usage()
    println(
        "usage: julia --project=. scripts/validation/summarize_tdvp_progress_csv.jl " *
        "[--cap D] PROGRESS.csv [PROGRESS2.csv ...]"
    )
end

function parse_csv_line(line::AbstractString)
    fields = String[]
    io = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if in_quotes
            if c == '"'
                next_i = nextind(line, i)
                if next_i <= lastindex(line) && line[next_i] == '"'
                    print(io, '"')
                    i = next_i
                else
                    in_quotes = false
                end
            else
                print(io, c)
            end
        elseif c == '"'
            in_quotes = true
        elseif c == ','
            push!(fields, String(take!(io)))
        else
            print(io, c)
        end
        i = nextind(line, i)
    end
    in_quotes && throw(ArgumentError("unterminated quoted CSV field"))
    push!(fields, String(take!(io)))
    return fields
end

function read_progress_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("progress CSV is empty: $path"))
    header = parse_csv_line(lines[1])
    rows = Vector{Dict{String,String}}()
    for (line_number, line) in enumerate(lines[2:end])
        isempty(line) && continue
        values = parse_csv_line(line)
        length(values) == length(header) || throw(ArgumentError(
            "row $(line_number + 1) in $path has $(length(values)) fields, " *
            "but the header has $(length(header))"
        ))
        push!(rows, Dict(zip(header, values)))
    end
    return header, rows
end

progress_cell(row, name::AbstractString) = get(row, name, "")

function progress_float(row, name::AbstractString)
    value = progress_cell(row, name)
    isempty(value) && return NaN
    return parse(Float64, value)
end

function progress_int(row, name::AbstractString)
    value = progress_cell(row, name)
    isempty(value) && return 0
    return parse(Int, value)
end

function group_key(row)
    return Tuple(progress_cell(row, col) for col in PROGRESS_GROUP_COLUMNS)
end

function group_label(key)
    return NamedTuple{Tuple(Symbol.(PROGRESS_GROUP_COLUMNS))}(key)
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

function cap_label(cycle::Integer, sweep)
    cycle == 0 && return "none"
    sweep === nothing && return string(cycle)
    return "$(cycle):$(sweep)"
end

function default_progress_cap(method::AbstractString, dmax::Integer)
    # This lightweight CLI intentionally avoids importing CoolingTNS.  The
    # factors mirror the project-wide dispatch rule in src/parameter_types.jl:
    # `tn_method_maxdim(::MonteCarloWavefunction, Dmax) = Dmax` and
    # `tn_method_maxdim(::DensityMatrix, Dmax) = 4Dmax`.
    method_name = lowercase(method)
    method_name == "mcwf" && return dmax
    method_name == "mpo" && return 4 * dmax
    throw(ArgumentError(
        "unknown method '$method' in progress CSV; pass --cap D explicitly"
    ))
end

function peak_evolved_bond(rows)
    peak = 0
    for row in rows
        value = progress_float(row, "evolved_max_bond")
        isfinite(value) || continue
        peak = max(peak, Int(round(value)))
    end
    return peak
end

function summarize_progress_group(file_name::AbstractString, key, rows; cap=nothing)
    label = group_label(key)
    threshold = cap === nothing ?
        default_progress_cap(label.method, parse(Int, label.Dmax)) :
        Int(cap)
    updates = [row for row in rows if progress_cell(row, "stage") == "updated"]

    system_cap_cycle = 0
    for row in updates
        if progress_float(row, "system_max_bond") >= threshold
            system_cap_cycle = progress_int(row, "cycle")
            break
        end
    end

    transient_cap_cycle = 0
    transient_cap_sweep = nothing
    for row in rows
        stage = progress_cell(row, "stage")
        stage in ("tdvp_sweep", "evolved", "updated", "prepared") || continue
        if progress_float(row, "evolved_max_bond") >= threshold
            transient_cap_cycle = progress_int(row, "cycle")
            if stage == "tdvp_sweep"
                transient_cap_sweep = progress_int(row, "tdvp_sweep")
            end
            break
        end
    end

    prepared_or_sweep_elapsed = Dict{Int,Float64}()
    max_sweep_increment = NaN
    max_sweep_cycle = 0
    max_sweep = 0
    for row in rows
        stage = progress_cell(row, "stage")
        cycle = progress_int(row, "cycle")
        elapsed = progress_float(row, "elapsed_seconds")
        if stage == "prepared"
            prepared_or_sweep_elapsed[cycle] = elapsed
        elseif stage == "tdvp_sweep" && haskey(prepared_or_sweep_elapsed, cycle)
            increment = elapsed - prepared_or_sweep_elapsed[cycle]
            prepared_or_sweep_elapsed[cycle] = elapsed
            if !isfinite(max_sweep_increment) || increment > max_sweep_increment
                max_sweep_increment = increment
                max_sweep_cycle = cycle
                max_sweep = progress_int(row, "tdvp_sweep")
            end
        end
    end

    completed_cycles = isempty(updates) ? 0 : maximum(progress_int(row, "cycle") for row in updates)
    final_update = isempty(updates) ? nothing : updates[end]
    final_energy = final_update === nothing ? NaN : progress_float(final_update, "energy_per_site")
    final_system_max = final_update === nothing ? 0 : Int(round(progress_float(final_update, "system_max_bond")))
    peak_evolved = peak_evolved_bond(rows)
    last_row = isempty(rows) ? nothing : rows[end]
    last_stage = last_row === nothing ? "none" : progress_cell(last_row, "stage")
    last_step = last_row === nothing ? 0 : progress_int(last_row, "step")
    last_cycle = last_row === nothing ? 0 : progress_int(last_row, "cycle")
    status = bond_cap_status(system_cap_cycle, transient_cap_cycle)

    return (
        file=basename(file_name),
        N=parse(Int, label.N),
        method=label.method,
        evolution=label.evolution,
        R=parse(Int, label.R),
        trajectory=parse(Int, label.trajectory),
        seed=parse(Int, label.seed),
        threshold=threshold,
        completed_cycles=completed_cycles,
        final_energy=final_energy,
        system_effective_bond=effective_bond_dimension_label(
            max(final_system_max, 0), system_cap_cycle, threshold,
        ),
        evolved_effective_bond=effective_bond_dimension_label(
            max(peak_evolved, 0), transient_cap_cycle, threshold,
        ),
        bond_status=status,
        system_cap_cycle=system_cap_cycle,
        transient_cap_cycle=transient_cap_cycle,
        transient_cap_sweep=transient_cap_sweep,
        max_sweep_increment=max_sweep_increment,
        max_sweep_cycle=max_sweep_cycle,
        max_sweep=max_sweep,
        last_step=last_step,
        last_cycle=last_cycle,
        last_stage=last_stage,
        updates=updates,
    )
end

function summarize_progress_file(path::AbstractString; cap=nothing)
    _, rows = read_progress_csv(path)
    groups = Dict{Any,Vector{Dict{String,String}}}()
    for row in rows
        push!(get!(groups, group_key(row), Vector{Dict{String,String}}()), row)
    end
    return [
        summarize_progress_group(path, key, group_rows; cap=cap)
        for (key, group_rows) in sort(
            collect(groups);
            by=pair -> (
                parse(Int, group_label(pair.first).N),
                group_label(pair.first).method,
                group_label(pair.first).evolution,
                parse(Int, group_label(pair.first).R),
                parse(Int, group_label(pair.first).trajectory),
            ),
        )
    ]
end

function summarize_progress_files(paths::AbstractVector{<:AbstractString}; cap=nothing)
    rows = NamedTuple[]
    for path in paths
        append!(rows, summarize_progress_file(path; cap=cap))
    end
    return rows
end

function print_summary_table(rows)
    println("| file | N | method | evolution | R | traj | seed | Dcap | completed cycles | final E/N | Dsys_eff | Dsb_eff | bond_status | sys cap | evolved cap | max sweep dt | max sweep at | last step | last cycle | last stage |")
    println("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|---:|---:|---|")
    for row in rows
        max_sweep_label = row.max_sweep_cycle == 0 ? "none" : "$(row.max_sweep_cycle):$(row.max_sweep)"
        println(
            "| $(row.file) | $(row.N) | $(row.method) | $(row.evolution) | " *
            "$(row.R) | $(row.trajectory) | $(row.seed) | $(row.threshold) | " *
            "$(row.completed_cycles) | $(format_float(row.final_energy, 8)) | " *
            "$(row.system_effective_bond) | $(row.evolved_effective_bond) | " *
            "$(row.bond_status) | $(saturation_cycle_label(row.system_cap_cycle)) | " *
            "$(cap_label(row.transient_cap_cycle, row.transient_cap_sweep)) | " *
            "$(format_float(row.max_sweep_increment, 1)) | $(max_sweep_label) | " *
            "$(row.last_step) | $(row.last_cycle) | $(row.last_stage) |"
        )
    end
end

function print_energy_trace(rows)
    println()
    println("| R | traj | cycle | delta | E/N | system max bond | evolved max bond | elapsed |")
    println("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows
        for update in row.updates
            println(
                "| $(row.R) | $(row.trajectory) | $(progress_cell(update, "cycle")) | " *
                "$(format_float(progress_float(update, "delta"), 8)) | " *
                "$(format_float(progress_float(update, "energy_per_site"), 8)) | " *
                "$(progress_cell(update, "system_max_bond")) | " *
                "$(progress_cell(update, "evolved_max_bond")) | " *
                "$(format_float(progress_float(update, "elapsed_seconds"), 1)) |"
            )
        end
    end
end

function print_markdown(rows)
    print_summary_table(rows)
    print_energy_trace(rows)
end

function parse_args(args)
    cap = nothing
    paths = String[]
    i = 1
    while i <= length(args)
        if args[i] == "--cap"
            i == length(args) && throw(ArgumentError("--cap requires a value"))
            cap = parse(Int, args[i + 1])
            i += 2
        elseif startswith(args[i], "--")
            throw(ArgumentError("unknown option: $(args[i])"))
        else
            push!(paths, args[i])
            i += 1
        end
    end
    return (paths=paths, cap=cap)
end

function summarize_tdvp_progress_csv_main(args=ARGS)
    parsed = parse_args(args)
    if isempty(parsed.paths)
        usage()
        return 1
    end
    rows = summarize_progress_files(parsed.paths; cap=parsed.cap)
    print_markdown(rows)
    return 0
end

end # module TDVPProgressCSVSummary

if abspath(PROGRAM_FILE) == @__FILE__
    exit(TDVPProgressCSVSummary.summarize_tdvp_progress_csv_main())
end
