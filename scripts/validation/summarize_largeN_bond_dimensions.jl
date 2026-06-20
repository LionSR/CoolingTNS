#!/usr/bin/env julia
"""
Summarize effective bond dimensions from large-N tensor-network campaign HDF5 files.

Example:

    julia --project=. scripts/validation/summarize_largeN_bond_dimensions.jl \
        /tmp/coolingtns_largeN_mcwf_N64_R1_steps4_Dmax320.h5 \
        /tmp/coolingtns_largeN_mcwf_N64_R2-5-10_steps4_Dmax320.h5

    julia --project=. scripts/validation/summarize_largeN_bond_dimensions.jl \
        --compact /tmp/coolingtns_largeN_mcwf_N64_R1_steps4_Dmax320.h5

The output is a Markdown table.  When present, the stored detuning protocol is
shown next to the bond-dimension diagnostics, so fixed-detuning cutoff sweeps
can be audited from the summary alone.  The `delta_range` column is the stored
protocol interval; for `R=1`, the campaign samples only the lower endpoint.  For
multi-trajectory data, final-link quantiles and threshold fractions are computed
per trajectory and then averaged over trajectories.  Stop-on-cap provenance is
read directly from the HDF5 fields `requested_steps`, `completed_steps`,
`stop_reasons`, and `elapsed_seconds` when available.  The elapsed column is a
sum over trajectory elapsed times, matching the sequential campaign driver.
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
        "[--compact] FILE.h5 [FILE2.h5 ...]"
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
format_integer_or_na(::Missing) = "n/a"
format_float_or_na(value::Real, digits::Int=2) = format_float(value, digits)
format_float_or_na(::Missing, digits::Int=2) = "n/a"
format_string_or_na(value) = string(value)
format_string_or_na(::Missing) = "n/a"

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

function read_group_value(primary_group, fallback_group, key::AbstractString, default)
    haskey(primary_group, key) && return read(primary_group[key])
    haskey(fallback_group, key) && return read(fallback_group[key])
    return default
end

function detuning_interval_label(delta_min::Real, delta_max::Real)
    isfinite(delta_min) && isfinite(delta_max) ||
        return "unknown"
    return "[$(format_float(delta_min, 8)),$(format_float(delta_max, 8))]"
end

function delta_values_interval_label(run_group)
    haskey(run_group, "delta_values") || return "unknown"
    delta_values = Float64.(read(run_group["delta_values"]))
    isempty(delta_values) && return "unknown"
    return detuning_interval_label(minimum(delta_values), maximum(delta_values))
end

function detuning_factor_label(source::AbstractString, factor::Real)
    source == "gap_scaled_range" && isfinite(factor) &&
        return format_float(factor, 3)
    source == "fixed_range" && return "n/a"
    return "unknown"
end

function detuning_protocol_summary(method_group, run_group)
    source = String(
        read_group_value(
            run_group,
            method_group,
            "detuning_protocol_source",
            "unknown",
        ),
    )
    delta_min = Float64(
        read_group_value(run_group, method_group, "detuning_delta_min", NaN),
    )
    delta_max = Float64(
        read_group_value(run_group, method_group, "detuning_delta_max", NaN),
    )
    factor = Float64(
        read_group_value(run_group, method_group, "detuning_delta_max_factor", NaN),
    )

    interval = source == "unknown" ?
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
        mode_final_e_over_n=missing,
        mode_final_abs_err=missing,
        mode_max_abs_err=missing,
    )
end

function mode_measurement_row_label(n_measured::Integer, n_rows::Integer)
    n_rows <= 0 && return "n/a"
    return "$(n_measured)/$(n_rows)"
end

function mode_reconstruction_summary(root, run_group, N::Integer, energy_mean)
    mode_keys = (
        RESULT_MODE_HK,
        RESULT_MODE_K_INDICES,
        RESULT_MODE_MEASUREMENT_CYCLES,
        RESULT_MODE_GF,
        RESULT_MODE_GF_SOURCE,
    )
    all(key -> haskey(run_group, key), mode_keys) ||
        return missing_mode_reconstruction_summary()

    mode_gF = Int(read(run_group[RESULT_MODE_GF]))
    mode_gF_source = String(read(run_group[RESULT_MODE_GF_SOURCE]))

    model = haskey(root, "model") ? String(read(root["model"])) : "unknown"
    bc = haskey(root, "bc") ? String(read(root["bc"])) : "unknown"
    if model != "ising" || !(bc in ("periodic", "antiperiodic"))
        error(
            "mode measurements are present, but the file describes model='$model' " *
            "with bc='$bc'; mode-energy reconstruction is defined here only for " *
            "the integrable Ising chain with periodic or antiperiodic spin BC"
        )
    end
    haskey(root, "J") && haskey(root, "h") ||
        error("mode measurements are present, but root datasets J and h are missing")

    ham_params = IsingParameters(N, Float64(read(root["J"])), Float64(read(root["h"])), Symbol(bc))
    mode_hk = Float64.(read(run_group[RESULT_MODE_HK]))
    mode_hk isa AbstractMatrix ||
        error("$RESULT_MODE_HK must be a steps-by-modes matrix")
    k_indices = Float64.(vec(read(run_group[RESULT_MODE_K_INDICES])))
    cycles = Int.(scalar_or_vector(read(run_group[RESULT_MODE_MEASUREMENT_CYCLES])))
    rows = cycles .+ 1
    valid_rows = [
        row for row in rows
        if 1 <= row <= size(mode_hk, 1) &&
           row <= length(energy_mean) &&
           isfinite(energy_mean[row]) &&
           all(isfinite, view(mode_hk, row, :))
    ]

    measured_label = mode_measurement_row_label(length(valid_rows), length(energy_mean))
    isempty(valid_rows) && return (
        mode_gF=mode_gF,
        mode_gF_source=mode_gF_source,
        mode_measured_rows=measured_label,
        mode_final_e_over_n=missing,
        mode_final_abs_err=missing,
        mode_max_abs_err=missing,
    )

    mode_energy = ising_energy_from_mode_hk(k_indices, mode_hk[valid_rows, :], ham_params)
    direct_energy = Float64.(energy_mean[valid_rows])
    abs_error = abs.(mode_energy .- direct_energy)
    return (
        mode_gF=mode_gF,
        mode_gF_source=mode_gF_source,
        mode_measured_rows=measured_label,
        mode_final_e_over_n=mode_energy[end] / N,
        mode_final_abs_err=abs_error[end],
        mode_max_abs_err=maximum(abs_error),
    )
end

energy_tail_start(nsteps::Integer) = max(1, nsteps - ENERGY_TAIL_WINDOW + 1)

function range_label(values::AbstractVector{<:Integer})
    isempty(values) && return "n/a"
    lo, hi = extrema(values)
    lo == hi && return string(lo)
    return "$lo-$hi"
end

function stop_reason_label(reasons::AbstractVector{<:AbstractString})
    values = String.(reasons)
    nonempty = filter(!isempty, values)
    isempty(nonempty) && return "none"
    unique_reasons = sort(unique(nonempty))
    length(unique_reasons) == 1 && length(nonempty) == length(values) &&
        return only(unique_reasons)
    return join(
        ["$(reason)x$(count(==(reason), nonempty))/$(length(values))" for reason in unique_reasons],
        "+",
    )
end

function method_from_name(method_name::AbstractString)
    # HDF5 stores method names as strings; the cap itself is still determined
    # by the library dispatch rule in `tn_method_maxdim`.
    method_name == "mcwf" && return MonteCarloWavefunction()
    method_name == "mpo" && return DensityMatrix()
    error("unknown method '$method_name' in campaign file")
end

function saturation_threshold_for(root, method_group, run_group, method_name::AbstractString)
    haskey(run_group, "bond_saturation_threshold") &&
        return Int(read(run_group["bond_saturation_threshold"]))
    haskey(method_group, "bond_saturation_threshold") &&
        return Int(read(method_group["bond_saturation_threshold"]))
    return tn_method_maxdim(method_from_name(method_name), Int(read(root["Dmax"])))
end

function final_link_dimensions(run_group)
    !haskey(run_group, "final_bond_dims") && return Vector{Vector{Int}}()
    bond_group = run_group["final_bond_dims"]
    names = sort(
        String.(keys(bond_group));
        by=name -> parse(Int, replace(name, "trajectory_" => "")),
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

function summarize_run(file_name::AbstractString, root, n_group_name::AbstractString,
                       method_name::AbstractString, method_group, r_group_name::AbstractString,
                       run_group)
    N = Int(read(root[n_group_name]["N"]))
    R = parse(Int, r_group_name[2:end])
    M = Int(read(run_group["M"]))
    evolution = String(read_group_value(method_group, root, "evolution_method", "unknown"))
    threshold = saturation_threshold_for(root, method_group, run_group, method_name)

    energy_mean = read(run_group["E_mean"])
    relative_energy_mean = read(run_group["relative_energy_mean"])
    inferred_completed_steps = max(length(energy_mean) - 1, 0)
    default_requested_steps = haskey(root, "steps") ?
        Int(read(root["steps"])) :
        inferred_completed_steps
    requested_steps_values = read_integer_vector(
        run_group,
        "requested_steps",
        fill(default_requested_steps, M),
    )
    completed_steps_values = read_integer_vector(
        run_group,
        "completed_steps",
        fill(inferred_completed_steps, M),
    )
    elapsed_values = read_float_vector(run_group, "elapsed_seconds")
    elapsed_seconds = isempty(elapsed_values) ? NaN : sum(elapsed_values)
    stop_reasons = read_string_vector(run_group, "stop_reasons")
    system_max_bond = bond_history_matrix(read(run_group["system_max_bond"]))
    system_mean_bond = bond_history_matrix(read(run_group["system_mean_bond"]))
    evolved_max_bond = bond_history_matrix(read(run_group["evolved_max_bond"]))
    evolved_mean_bond = bond_history_matrix(read(run_group["evolved_mean_bond"]))
    tdvp_sweep_dataset_present = haskey(run_group, "tdvp_sweep_max_bond")
    tdvp_sweep_max_bond = tdvp_sweep_dataset_present ?
        bond_history_matrix(read(run_group["tdvp_sweep_max_bond"])) :
        Matrix{Int}(undef, 0, 0)
    tdvp_sweep_saturation_dataset_cycle = first_saturation_from_dataset(
        run_group, "tdvp_sweep_saturation_cycle"
    )
    has_tdvp_sweep_history =
        !isempty(tdvp_sweep_max_bond) &&
        (any(value -> value != 0, tdvp_sweep_max_bond) ||
         tdvp_sweep_saturation_dataset_cycle > 0)

    final_e_over_n = energy_mean[end] / N
    final_relative_energy = relative_energy_mean[end]
    best_e_over_n = minimum(energy_mean) / N
    best_relative_energy = minimum(relative_energy_mean)
    tail_start = energy_tail_start(length(energy_mean))
    tail_count = length(energy_mean) - tail_start + 1
    tail_e_over_n = mean(energy_mean[tail_start:end]) / N
    tail_relative_energy = mean(relative_energy_mean[tail_start:end])
    final_system_max = final_system_max_bond(system_max_bond)
    final_system_mean = final_system_mean_bond(system_mean_bond)
    peak_evolved_max = peak_evolved_max_bond(evolved_max_bond)
    peak_evolved_mean = peak_evolved_mean_bond(evolved_mean_bond)
    peak_tdvp_sweep_max = has_tdvp_sweep_history ?
        maximum(tdvp_sweep_max_bond) :
        missing

    system_saturation_cycle = first_saturation_from_dataset(run_group, "system_saturation_cycle")
    system_saturation_cycle == 0 &&
        (system_saturation_cycle = first_saturation_from_history(system_max_bond, threshold))
    evolved_saturation_cycle = first_saturation_from_dataset(run_group, "evolved_saturation_cycle")
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
        "n/a"
    bond_status = bond_cap_status(
        system_saturation_cycle, evolved_saturation_cycle,
        tdvp_sweep_saturation_cycle_for_status,
    )

    link_dims = final_link_dimensions(run_group)
    quantiles = mean_link_quantiles(link_dims)
    fractions = mean_link_threshold_fractions(link_dims, threshold)
    detuning = detuning_protocol_summary(method_group, run_group)
    mode_summary = mode_reconstruction_summary(root, run_group, N, energy_mean)

    return (
        file=basename(file_name),
        N=N,
        method=method_name,
        evolution=evolution,
        R=R,
        M=M,
        completed_requested="$(range_label(completed_steps_values))/$(range_label(requested_steps_values))",
        elapsed_total_seconds=elapsed_seconds,
        stop_reason=stop_reason_label(stop_reasons),
        delta_protocol=detuning.delta_protocol,
        delta_range=detuning.delta_range,
        delta_factor=detuning.delta_factor,
        threshold=threshold,
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
        mode_final_e_over_n=mode_summary.mode_final_e_over_n,
        mode_final_abs_err=mode_summary.mode_final_abs_err,
        mode_max_abs_err=mode_summary.mode_max_abs_err,
        system_effective_bond=system_effective_bond,
        evolved_effective_bond=evolved_effective_bond,
        tdvp_sweep_effective_bond=tdvp_sweep_effective_bond,
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
            startswith(n_group_name, "N") || continue
            root[n_group_name] isa HDF5.Group || continue
            haskey(root[n_group_name], "N") || continue
            n_group = root[n_group_name]
            for method_name in sort(String.(keys(n_group)))
                n_group[method_name] isa HDF5.Group || continue
                method_group = n_group[method_name]
                for r_group_name in sort(String.(keys(method_group)); by=name -> startswith(name, "R") ? parse(Int, name[2:end]) : typemax(Int))
                    startswith(r_group_name, "R") || continue
                    method_group[r_group_name] isa HDF5.Group || continue
                    push!(
                        rows,
                        summarize_run(path, root, n_group_name, method_name, method_group,
                                      r_group_name, method_group[r_group_name]),
                    )
                end
            end
        end
    end
    return rows
end

function sorted_rows(rows)
    return sort(rows; by=row -> (row.N, row.method, row.evolution, row.R, row.file))
end

function print_markdown(rows)
    println("| file | N | method | evolution | R | M | completed/requested | elapsed_total | stop_reason | delta_protocol | delta_range | delta_factor | Dcap | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | final E/N | relE | best E/N | best relE | tail E/N | tail relE | tail n | mode gF | mode source | mode rows | mode final E/N | mode final abs dE | mode max abs dE | final sys max | final sys mean | peak evolved max | peak evolved mean | peak tdvp sweep max | sys sat | evolved sat | tdvp sweep sat | q50 | q75 | q90 | q95 | frac_ge_0.5D | frac_ge_0.75D | frac_ge_0.9D |")
    println("|---|---:|---|---|---:|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted_rows(rows)
        println(
            "| $(row.file) | $(row.N) | $(row.method) | $(row.evolution) | $(row.R) | $(row.M) | " *
            "$(row.completed_requested) | $(format_float(row.elapsed_total_seconds, 1)) | " *
            "$(row.stop_reason) | " *
            "$(row.delta_protocol) | $(row.delta_range) | $(row.delta_factor) | " *
            "$(row.threshold) | " *
            "$(row.system_effective_bond) | $(row.evolved_effective_bond) | " *
            "$(row.tdvp_sweep_effective_bond) | $(row.bond_status) | " *
            "$(format_float(row.final_e_over_n, 8)) | $(format_float(row.relative_energy, 5)) | " *
            "$(format_float(row.best_e_over_n, 8)) | $(format_float(row.best_relative_energy, 5)) | " *
            "$(format_float(row.tail_e_over_n, 8)) | $(format_float(row.tail_relative_energy, 5)) | " *
            "$(row.tail_count) | " *
            "$(format_string_or_na(row.mode_gF)) | $(format_string_or_na(row.mode_gF_source)) | " *
            "$(format_string_or_na(row.mode_measured_rows)) | " *
            "$(format_float_or_na(row.mode_final_e_over_n, 8)) | " *
            "$(format_float_or_na(row.mode_final_abs_err, 3)) | " *
            "$(format_float_or_na(row.mode_max_abs_err, 3)) | " *
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
    println("| file | N | method | evolution | R | M | completed/requested | final E/N | best E/N | mode max abs dE | Dcap | Dsys_eff | Dsb_eff | Dtdvp_sweep_eff | bond_status | elapsed_total | stop_reason |")
    println("|---|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|")
    for row in sorted_rows(rows)
        println(
            "| $(row.file) | $(row.N) | $(row.method) | $(row.evolution) | " *
            "$(row.R) | $(row.M) | $(row.completed_requested) | " *
            "$(format_float(row.final_e_over_n, 8)) | " *
            "$(format_float(row.best_e_over_n, 8)) | " *
            "$(format_float_or_na(row.mode_max_abs_err, 3)) | $(row.threshold) | " *
            "$(row.system_effective_bond) | $(row.evolved_effective_bond) | " *
            "$(row.tdvp_sweep_effective_bond) | $(row.bond_status) | " *
            "$(format_float(row.elapsed_total_seconds, 1)) | $(row.stop_reason) |"
        )
    end
end

function parse_args(args)
    compact = false
    paths = String[]
    for arg in args
        if arg == "--compact"
            compact = true
        elseif startswith(arg, "--")
            throw(ArgumentError("unknown option: $arg"))
        else
            push!(paths, arg)
        end
    end
    return (paths=paths, compact=compact)
end

function summarize_largeN_bond_dimensions_main(args=ARGS)
    if isempty(args) || any(arg -> arg in ("-h", "--help"), args)
        usage()
        return isempty(args) ? 1 : 0
    end

    parsed = parse_args(args)
    rows = NamedTuple[]
    for path in parsed.paths
        isfile(path) || error("not a file: $path")
        append!(rows, summarize_file(path))
    end
    isempty(rows) && error("no large-N campaign runs found")
    parsed.compact ? print_compact_markdown(rows) : print_markdown(rows)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(summarize_largeN_bond_dimensions_main())
end
