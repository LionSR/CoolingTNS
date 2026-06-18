#!/usr/bin/env julia
"""
Summarize effective bond dimensions from large-N tensor-network campaign HDF5 files.

Example:

    julia --project=. scripts/validation/summarize_largeN_bond_dimensions.jl \
        /tmp/coolingtns_largeN_mcwf_N64_R1_steps4_Dmax320.h5 \
        /tmp/coolingtns_largeN_mcwf_N64_R2-5-10_steps4_Dmax320.h5

The output is a Markdown table.  For multi-trajectory data, final-link
quantiles and threshold fractions are computed per trajectory and then averaged
over trajectories.
"""

using CoolingTNS
using HDF5
using Printf
using Statistics

include(joinpath(@__DIR__, "largeN_scaling_helpers.jl"))

const LINK_QUANTILE_PROBABILITIES = [0.50, 0.75, 0.90, 0.95]
const LINK_THRESHOLD_FRACTIONS = [0.50, 0.75, 0.90]

function usage()
    println(
        "usage: julia --project=. scripts/validation/summarize_largeN_bond_dimensions.jl " *
        "FILE.h5 [FILE2.h5 ...]"
    )
end

function format_float(value::Real, digits::Int=2)
    !isfinite(value) && return "NaN"
    digits == 1 && return @sprintf("%.1f", value)
    digits == 2 && return @sprintf("%.2f", value)
    digits == 5 && return @sprintf("%.5f", value)
    digits == 8 && return @sprintf("%.8f", value)
    return string(round(Float64(value); digits=digits))
end

function method_from_name(method_name::AbstractString)
    # HDF5 stores method names as strings; the cap itself is still determined
    # by the library dispatch rule in `tn_trotter_maxdim`.
    method_name == "mcwf" && return MonteCarloWavefunction()
    method_name == "mpo" && return DensityMatrix()
    error("unknown method '$method_name' in campaign file")
end

function saturation_threshold_for(root, method_group, run_group, method_name::AbstractString)
    haskey(run_group, "bond_saturation_threshold") &&
        return Int(read(run_group["bond_saturation_threshold"]))
    haskey(method_group, "bond_saturation_threshold") &&
        return Int(read(method_group["bond_saturation_threshold"]))
    return tn_trotter_maxdim(method_from_name(method_name), Int(read(root["Dmax"])))
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
    threshold = saturation_threshold_for(root, method_group, run_group, method_name)

    energy_mean = read(run_group["E_mean"])
    relative_energy_mean = read(run_group["relative_energy_mean"])
    system_max_bond = bond_history_matrix(read(run_group["system_max_bond"]))
    system_mean_bond = bond_history_matrix(read(run_group["system_mean_bond"]))
    evolved_max_bond = bond_history_matrix(read(run_group["evolved_max_bond"]))
    evolved_mean_bond = bond_history_matrix(read(run_group["evolved_mean_bond"]))

    final_e_over_n = energy_mean[end] / N
    final_relative_energy = relative_energy_mean[end]
    final_system_max = final_system_max_bond(system_max_bond)
    final_system_mean = final_system_mean_bond(system_mean_bond)
    peak_evolved_max = peak_evolved_max_bond(evolved_max_bond)
    peak_evolved_mean = peak_evolved_mean_bond(evolved_mean_bond)

    system_saturation_cycle = first_saturation_from_dataset(run_group, "system_saturation_cycle")
    system_saturation_cycle == 0 &&
        (system_saturation_cycle = first_saturation_from_history(system_max_bond, threshold))
    evolved_saturation_cycle = first_saturation_from_dataset(run_group, "evolved_saturation_cycle")
    evolved_saturation_cycle == 0 &&
        (evolved_saturation_cycle = first_saturation_from_history(evolved_max_bond, threshold))
    system_effective_bond = effective_bond_dimension_label(
        final_system_max, system_saturation_cycle, threshold
    )
    evolved_effective_bond = effective_bond_dimension_label(
        peak_evolved_max, evolved_saturation_cycle, threshold
    )

    link_dims = final_link_dimensions(run_group)
    quantiles = mean_link_quantiles(link_dims)
    fractions = mean_link_threshold_fractions(link_dims, threshold)

    return (
        file=basename(file_name),
        N=N,
        method=method_name,
        R=R,
        M=M,
        threshold=threshold,
        final_e_over_n=final_e_over_n,
        relative_energy=final_relative_energy,
        system_effective_bond=system_effective_bond,
        evolved_effective_bond=evolved_effective_bond,
        final_system_max=final_system_max,
        final_system_mean=final_system_mean,
        peak_evolved_max=peak_evolved_max,
        peak_evolved_mean=peak_evolved_mean,
        system_saturation_cycle=system_saturation_cycle,
        evolved_saturation_cycle=evolved_saturation_cycle,
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

function print_markdown(rows)
    println("| file | N | method | R | M | Dcap | Dsys_eff | Dsb_eff | final E/N | relE | final sys max | final sys mean | peak evolved max | peak evolved mean | sys sat | evolved sat | q50 | q75 | q90 | q95 | frac_ge_0.5D | frac_ge_0.75D | frac_ge_0.9D |")
    println("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sort(rows; by=row -> (row.N, row.method, row.R, row.file))
        println(
            "| $(row.file) | $(row.N) | $(row.method) | $(row.R) | $(row.M) | $(row.threshold) | " *
            "$(row.system_effective_bond) | $(row.evolved_effective_bond) | " *
            "$(format_float(row.final_e_over_n, 8)) | $(format_float(row.relative_energy, 5)) | " *
            "$(row.final_system_max) | $(format_float(row.final_system_mean, 2)) | " *
            "$(row.peak_evolved_max) | $(format_float(row.peak_evolved_mean, 2)) | " *
            "$(saturation_cycle_label(row.system_saturation_cycle)) | " *
            "$(saturation_cycle_label(row.evolved_saturation_cycle)) | " *
            "$(format_float(row.q50, 1)) | $(format_float(row.q75, 1)) | " *
            "$(format_float(row.q90, 1)) | $(format_float(row.q95, 1)) | " *
            "$(format_float(row.frac50, 2)) | $(format_float(row.frac75, 2)) | " *
            "$(format_float(row.frac90, 2)) |"
        )
    end
end

function main(args=ARGS)
    if isempty(args) || any(arg -> arg in ("-h", "--help"), args)
        usage()
        return isempty(args) ? 1 : 0
    end

    rows = NamedTuple[]
    for path in args
        isfile(path) || error("not a file: $path")
        append!(rows, summarize_file(path))
    end
    isempty(rows) && error("no large-N campaign runs found")
    print_markdown(rows)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
