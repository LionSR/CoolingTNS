"""
General plotting functions for cooling results visualization.

Standalone plotting script. Usage:
    julia --project=. scripts/plotting/plotting.jl
or:
    include("scripts/plotting/plotting.jl")
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

# Import CoolingTNS types and functions needed by this file
using CoolingTNS: HamiltonianParameters, CouplingParameters, UnifiedSimulationParameters,
    CoolingBackend, parse_hamiltonian_name, create_filename, mean_last_window,
    create_search_name_part, RESULT_ENERGY, RESULT_GROUND_STATE_OVERLAP,
    RESULT_MOMENTUM_DISTRIBUTION, RESULT_K_VALUES

_ham_params_with_N(template::HamiltonianParameters, N::Int) =
    HamiltonianParameters(template.model, N, template.params, template.bc)

# Resolve the actual HDF5 file to read.
#
# - Standard runs: `Results/<filename_prefix>.h5`
# - Optimization runs: `ResultsOpt/<filename_prefix>_<search_name_part>.h5`
#   If `search_params` is not provided, try to infer by picking the most recently
#   modified match with the right prefix.
function _results_h5_path(
    directory::AbstractString,
    filename_prefix::AbstractString;
    is_optimization::Bool=false,
    search_params=nothing,
)
    isdir(directory) || return nothing

    if !is_optimization
        path = joinpath(directory, filename_prefix * ".h5")
        return isfile(path) ? path : nothing
    end

    if search_params !== nothing
        filename = filename_prefix * "_" * create_search_name_part(search_params)
        path = joinpath(directory, filename * ".h5")
        return isfile(path) ? path : nothing
    end

    candidates = filter(
        f -> startswith(f, filename_prefix * "_") && endswith(f, ".h5"),
        readdir(directory),
    )
    isempty(candidates) && return nothing

    paths = joinpath.(Ref(directory), candidates)
    mtimes = map(p -> stat(p).mtime, paths)
    return paths[argmax(mtimes)]
end

function _with_pe(sim_params::UnifiedSimulationParameters{S,E}, pe::Real) where {S,E}
    return UnifiedSimulationParameters(
        sim_params.sim_method,
        sim_params.evolution_method;
        Dmax=sim_params.Dmax,
        cutoff=sim_params.cutoff,
        tau=sim_params.tau,
        pe=Float64(pe),
        n_trajectories=sim_params.n_trajectories,
        parallel=sim_params.parallel,
        trotter_steps=sim_params.trotter_steps,
        maxiter=sim_params.maxiter,
        normalize=sim_params.normalize,
    )
end

function _read_final_metrics(filename::AbstractString, N::Int)
    e0, edensity_final, gs_overlap_final, E_list, GS_overlap_list = safe_read_keys(
        filename,
        "e₀",
        "Edensity_final",
        "GS_overlap_final",
        RESULT_ENERGY,
        RESULT_GROUND_STATE_OVERLAP,
    )
    e0 === nothing && return nothing

    e0 = Float64(_maybe_scalar(e0))

    edensity = if edensity_final !== nothing
        Float64(_maybe_scalar(edensity_final))
    elseif E_list !== nothing
        Float64(_maybe_scalar(E_list[end])) / N
    else
        nothing
    end

    overlap = if gs_overlap_final !== nothing
        Float64(_maybe_scalar(gs_overlap_final))
    elseif GS_overlap_list !== nothing
        Float64(_maybe_scalar(GS_overlap_list[end]))
    else
        nothing
    end

    (edensity === nothing || overlap === nothing) && return nothing
    return e0, edensity, overlap
end

function plot_energy_and_overlap(E_list, GS_overlap_list, e0, N, filename; moving_average=false, output_dir="Results")
    plt = get_pyplot()

    steps = length(E_list) - 1

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    ax = axs[0]
    ax.plot(1:steps+1, E_list / N, alpha=0.75, marker="o", label="Cooling")
    if moving_average
        window_size = 10
        E_ma = [CoolingTNS.mean_last_window(E_list[1:i], window_size) for i in 1:length(E_list)]
        ax.plot(1:steps+1, E_ma / N, alpha=0.75, marker="o", label="Cooling (MA=$(window_size))")
    end
    ax.set_xlabel("Steps")
    ax.set_ylabel(L"Energy density $E/N$")
    ax.axhline(y=e0 / N, xmin=0, xmax=1, linewidth=1.5, color="black", label=L"$E_0/N$")
    ax.legend()

    ax = axs[1]
    ax.plot(1:steps+1, GS_overlap_list, marker="o", alpha=0.75, color="grey", label="Cooling")
    if moving_average
        GS_ma = [CoolingTNS.mean_last_window(GS_overlap_list[1:i], window_size) for i in 1:length(GS_overlap_list)]
        ax.plot(1:steps+1, GS_ma, marker="o", alpha=0.75, color="black", label="Cooling (MA=$(window_size))")
    end
    ax.set_xlabel("Steps")
    ax.set_ylabel("Ground state overlap")
    ax.legend()

    mkpath(output_dir)
    fig.savefig(joinpath(output_dir, "$(filename).pdf"), dpi=300)
end

function plot_vs_N(
    ham_name::AbstractString,
    coupling_params::CouplingParameters,
    sim_params::UnifiedSimulationParameters,
    backend::CoolingBackend,
    N_values::Vector{Int};
    is_optimization::Bool=false,
    search_params=nothing,
)
    return plot_vs_N(
        parse_hamiltonian_name(ham_name),
        coupling_params,
        sim_params,
        backend,
        N_values;
        is_optimization=is_optimization,
        search_params=search_params,
    )
end

function plot_vs_N(
    ham_template::HamiltonianParameters,
    coupling_params::CouplingParameters,
    sim_params::UnifiedSimulationParameters,
    backend::CoolingBackend,
    N_values::Vector{Int};
    is_optimization::Bool=false,
    search_params=nothing,
)
    plt = get_pyplot()

    energy_densities = Float64[]
    final_overlaps = Float64[]
    valid_N_values = Int[]
    e0_values = Float64[]

    directory = is_optimization ? "ResultsOpt" : "Results"
    prefix = is_optimization ? "Optimize" : ""

    for N in N_values
        ham_params = _ham_params_with_N(ham_template, N)
        filename_prefix = prefix * create_filename(ham_params, coupling_params, sim_params, backend)

        full_filename = _results_h5_path(
            directory,
            filename_prefix;
            is_optimization=is_optimization,
            search_params=search_params,
        )
        full_filename === nothing && continue

        e0, edensity_final, gs_overlap_final, E_list, GS_overlap_list = safe_read_keys(
            full_filename,
            "e₀",
            "Edensity_final",
            "GS_overlap_final",
            RESULT_ENERGY,
            RESULT_GROUND_STATE_OVERLAP,
        )
        e0 === nothing && continue

        e0 = _maybe_scalar(e0)

        edensity = if edensity_final !== nothing
            Float64(_maybe_scalar(edensity_final))
        elseif E_list !== nothing
            Float64(_maybe_scalar(E_list[end])) / N
        else
            nothing
        end

        overlap = if gs_overlap_final !== nothing
            Float64(_maybe_scalar(gs_overlap_final))
        elseif GS_overlap_list !== nothing
            Float64(_maybe_scalar(GS_overlap_list[end]))
        else
            nothing
        end

        (edensity === nothing || overlap === nothing) && continue

        push!(energy_densities, edensity)
        push!(final_overlaps, overlap)
        push!(valid_N_values, N)
        push!(e0_values, Float64(e0))
    end

    if isempty(valid_N_values)
        @error "No valid data points found. Skipping plot generation."
        return
    end

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(valid_N_values, energy_densities, marker="o", linestyle="-", label="Final energy density")
    ax1.plot(valid_N_values, e0_values ./ valid_N_values, linestyle="--", color="black", label=L"$E_0/N$")
    ax1.set_xlabel("System size (N)")
    ax1.set_ylabel("Energy density")
    ax1.legend()

    ax2.plot(valid_N_values, final_overlaps, marker="o", linestyle="-", label="Final overlap")
    ax2.set_xlabel("System size (N)")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    ham_params_first = _ham_params_with_N(ham_template, valid_N_values[1])
    filename_saveto = create_filename(ham_params_first, coupling_params, sim_params, backend)
    filename_saveto = prefix * filename_saveto * "_energy_density_and_overlap_vs_N.pdf"

    mkpath(joinpath(directory, "Figs"))
    fig.savefig(joinpath(directory, "Figs", filename_saveto), dpi=300)
end

function plot_cooling_curve_noise(
    ham_name::AbstractString,
    N::Int,
    coupling_params::CouplingParameters,
    sim_params::UnifiedSimulationParameters,
    peInt_range;
    backend=nothing,
    is_optimization::Bool=false,
    search_params=nothing,
)
    ham_template = parse_hamiltonian_name(ham_name)
    ham_params = _ham_params_with_N(ham_template, N)
    return plot_cooling_curve_noise(
        ham_params,
        coupling_params,
        sim_params,
        peInt_range;
        backend=backend,
        is_optimization=is_optimization,
        search_params=search_params,
    )
end

function plot_cooling_curve_noise(
    ham_params::HamiltonianParameters,
    coupling_params::CouplingParameters,
    sim_params::UnifiedSimulationParameters,
    peInt_range;
    backend=nothing,
    is_optimization::Bool=false,
    search_params=nothing,
)
    backend === nothing && throw(ArgumentError("backend must be provided for typed sim_params"))

    plt = get_pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    directory = is_optimization ? "ResultsOpt" : "Results"
    prefix = is_optimization ? "Optimize" : ""

    for peInt in peInt_range
        pe = round(peInt * 1e-3; digits=4)
        sim_pe = _with_pe(sim_params, pe)

        filename_prefix = prefix * create_filename(ham_params, coupling_params, sim_pe, backend)
        full_filename = _results_h5_path(
            directory,
            filename_prefix;
            is_optimization=is_optimization,
            search_params=search_params,
        )
        full_filename === nothing && continue

        e0, E_list, GS_overlap_list = safe_read_keys(
            full_filename, "e₀", RESULT_ENERGY, RESULT_GROUND_STATE_OVERLAP
        )
        (e0 === nothing || E_list === nothing || GS_overlap_list === nothing) && continue

        steps = length(E_list)
        ax1.plot(1:steps, E_list ./ ham_params.N, label="pe=$(pe)")
        ax2.plot(1:steps, GS_overlap_list, label="pe=$(pe)")
    end

    ax1.set_xlabel("Cooling steps")
    ax1.set_ylabel("Energy density")
    ax1.legend()

    ax2.set_xlabel("Cooling steps")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    filename_saveto = create_filename(ham_params, coupling_params, sim_params, backend)
    filename_saveto = prefix * filename_saveto * "_cooling_curve_noise.pdf"

    mkpath(joinpath(directory, "Figs"))
    fig.savefig(joinpath(directory, "Figs", filename_saveto), dpi=300)
end

function plot_vs_N_pe_range(
    ham_name::AbstractString,
    coupling_params::CouplingParameters,
    sim_params::UnifiedSimulationParameters,
    N_values,
    peInt_range;
    backend=nothing,
    is_optimization::Bool=false,
    search_params=nothing,
)
    backend === nothing && throw(ArgumentError("backend must be provided for typed sim_params"))

    return plot_vs_N_pe_range(
        parse_hamiltonian_name(ham_name),
        coupling_params,
        sim_params,
        backend,
        N_values,
        peInt_range;
        is_optimization=is_optimization,
        search_params=search_params,
    )
end

function plot_vs_N_pe_range(
    ham_template::HamiltonianParameters,
    coupling_params::CouplingParameters,
    sim_params::UnifiedSimulationParameters,
    backend::CoolingBackend,
    N_values,
    peInt_range;
    is_optimization::Bool=false,
    search_params=nothing,
)
    plt = get_pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    directory = is_optimization ? "ResultsOpt" : "Results"
    prefix = is_optimization ? "Optimize" : ""

    for peInt in peInt_range
        pe = round(peInt * 1e-3; digits=4)
        sim_pe = _with_pe(sim_params, pe)

        energy_errors = Float64[]
        final_overlaps = Float64[]
        valid_N_values = Int[]

        for N in N_values
            ham_params = _ham_params_with_N(ham_template, N)
            filename_prefix = prefix * create_filename(ham_params, coupling_params, sim_pe, backend)

            full_filename = _results_h5_path(
                directory,
                filename_prefix;
                is_optimization=is_optimization,
                search_params=search_params,
            )
            full_filename === nothing && continue

            e0, edensity_final, gs_overlap_final, E_list, GS_overlap_list = safe_read_keys(
                full_filename,
                "e₀",
                "Edensity_final",
                "GS_overlap_final",
                RESULT_ENERGY,
                RESULT_GROUND_STATE_OVERLAP,
            )
            e0 === nothing && continue

            e0 = Float64(_maybe_scalar(e0))

            edensity = if edensity_final !== nothing
                Float64(_maybe_scalar(edensity_final))
            elseif E_list !== nothing
                Float64(_maybe_scalar(E_list[end])) / N
            else
                nothing
            end

            overlap = if gs_overlap_final !== nothing
                Float64(_maybe_scalar(gs_overlap_final))
            elseif GS_overlap_list !== nothing
                Float64(_maybe_scalar(GS_overlap_list[end]))
            else
                nothing
            end

            (edensity === nothing || overlap === nothing) && continue

            push!(energy_errors, abs(edensity - e0 / N))
            push!(final_overlaps, overlap)
            push!(valid_N_values, N)
        end

        isempty(valid_N_values) && continue

        ax1.plot(valid_N_values, energy_errors, marker="o", linestyle="-", label="pe=$(pe)")
        ax2.plot(valid_N_values, final_overlaps, marker="o", linestyle="-", label="pe=$(pe)")
    end

    ax1.set_xlabel("System size (N)")
    ax1.set_ylabel("Energy density error")
    ax1.legend()

    ax2.set_xlabel("System size (N)")
    ax2.set_ylabel("Ground state overlap")
    ax2.legend()

    plt.tight_layout()

    ham_params_first = _ham_params_with_N(ham_template, N_values[1])
    filename_saveto = create_filename(ham_params_first, coupling_params, sim_params, backend)
    filename_saveto = prefix * filename_saveto * "_energy_error_and_overlap_vs_N_multiple_pe.pdf"

    mkpath(joinpath(directory, "Figs"))
    fig.savefig(joinpath(directory, "Figs", filename_saveto), dpi=300)
end


"""
    plot_momentum_distribution(filename; steps_to_plot=nothing, save_fig=true, show_fig=false)

Plot the momentum distribution n_k vs k at a subset of cooling steps.

If the file contains a scalar `delta`, resonant momenta are marked only when
the corresponding mode energies are known, either from stored mode energies or
from the Ising parameters `J,h`.
"""
function plot_momentum_distribution(filename; steps_to_plot=nothing, save_fig=true, show_fig=false)
    plt = get_pyplot()

    data = read_h5_data(filename)
    data === nothing && return

    if !haskey(data, RESULT_MOMENTUM_DISTRIBUTION) || !haskey(data, RESULT_K_VALUES)
        @warn "No k-space data found in file $filename"
        return
    end

    plot_data = kspace_evolution_plot_data(data)
    momentum_dist = plot_data.momentum_dist
    k_values = plot_data.k_values
    total_steps = plot_data.total_steps
    step_indices = select_evolution_steps(total_steps; steps_to_plot=steps_to_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_evolution_colors(plt, length(step_indices))

    for (i, step) in enumerate(step_indices)
        if step <= total_steps
            n_k = momentum_dist[step, :]
            label = step == 1 ? "Initial" : "Step $step"
            ax.plot(k_values, n_k, "o-", color=colors[i], label=label, markersize=4)
        end
    end

    mark_bath_resonance_from_data!(ax, data, k_values; momentum_scale=1)

    ax.set_xlabel(L"Momentum $k$")
    ax.set_ylabel(L"Occupation $n_k$")
    ax.set_title("Momentum Distribution Evolution")
    ax.legend()
    ax.grid(true, alpha=0.3)
    plt.tight_layout()

    if save_fig
        base_name = extract_filename_base(filename)
        save_figure(fig, dirname(filename), "momentum_dist_$(base_name).pdf")
    end

    show_fig && plt.show()
    return fig
end

"""
    plot_momentum_distribution_heatmap(filename; save_fig=true, show_fig=false)

Plot the momentum distribution as a heatmap showing n_k vs (k, step).
"""
function plot_momentum_distribution_heatmap(filename; save_fig=true, show_fig=false)
    plt = get_pyplot()

    data = read_h5_data(filename)
    data === nothing && return

    if !haskey(data, RESULT_MOMENTUM_DISTRIBUTION) || !haskey(data, RESULT_K_VALUES)
        @warn "No k-space data found in file $filename"
        return
    end

    plot_data = kspace_evolution_plot_data(data)
    momentum_dist = plot_data.momentum_dist
    k_values = plot_data.k_values
    total_steps = plot_data.total_steps

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        transpose(momentum_dist),
        aspect="auto",
        origin="lower",
        extent=[1, total_steps, k_values[1], k_values[end]],
        cmap="hot",
        interpolation="nearest",
    )

    ax.set_xlabel("Cooling Step")
    ax.set_ylabel(L"Momentum $k$")
    ax.set_title("Momentum Distribution Evolution Heatmap")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(L"Occupation $n_k$")

    if 0 in k_values
        ax.axhline(y=0, color="white", linestyle="--", alpha=0.5, linewidth=1)
    end

    plt.tight_layout()

    if save_fig
        base_name = extract_filename_base(filename)
        save_figure(fig, dirname(filename), "momentum_dist_heatmap_$(base_name).pdf")
    end

    show_fig && plt.show()
    return fig
end

"""
    plot_data(filename; moving_average=false, output_dir=nothing)

Convenience wrapper: load the energy and ground-state-overlap result keys,
`e₀`, and `N` from an HDF5 results file and generate the standard
energy/overlap plot.

The plot is saved next to the HDF5 file (same directory) unless `output_dir` is
provided.
"""
function plot_data(filename::AbstractString; moving_average::Bool=false, output_dir=nothing)
    data = read_h5_data(filename)
    data === nothing && return nothing

    for key in ("e₀", RESULT_ENERGY, RESULT_GROUND_STATE_OVERLAP, "N")
        if !haskey(data, key)
            @warn "Missing key \"$key\" in file $filename"
            return nothing
        end
    end

    e0 = _maybe_scalar(data["e₀"])
    E_list = data[RESULT_ENERGY]
    GS_overlap_list = data[RESULT_GROUND_STATE_OVERLAP]
    N = Int(_maybe_scalar(data["N"]))

    out_dir = output_dir === nothing ? dirname(filename) : output_dir
    base = extract_filename_base(filename)

    plot_energy_and_overlap(E_list, GS_overlap_list, e0, N, base; moving_average=moving_average, output_dir=out_dir)

    return nothing
end
