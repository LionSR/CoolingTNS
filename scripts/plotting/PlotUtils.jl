"""
Shared plotting utilities for k-space visualization.
Provides common helper functions to reduce duplication across plotting scripts.

Usage: `include("PlotUtils.jl")` from other scripts in this directory.
"""

using CoolingTNS
using HDF5
using PythonCall
using LaTeXStrings

# Import shared dispersion functions and result-key constants from CoolingTNS.
using CoolingTNS:
    generate_k_values,
    compute_energy_dispersion,
    compute_ground_state_occupation,
    RESULT_MOMENTUM_DISTRIBUTION,
    RESULT_K_VALUES,
    RESULT_MODE_ENERGIES

# Lazy pyplot access - imported once per session
const _pyplot = Ref{Py}()

"""
    get_pyplot()

Get the matplotlib.pyplot module, caching the import for performance.
"""
function get_pyplot()
    if !isassigned(_pyplot)
        _pyplot[] = pyimport("matplotlib.pyplot")
    end
    return _pyplot[]
end

"""
    read_h5_data(filename::String) -> Union{Dict{String, Any}, Nothing}

Read all top-level datasets from an HDF5 file into a dictionary.
Returns nothing if the file doesn't exist.
"""
function read_h5_data(filename::String)::Union{Dict{String, Any}, Nothing}
    if !isfile(filename)
        @warn "File not found: $filename"
        return nothing
    end

    data = Dict{String, Any}()
    h5open(filename, "r") do file
        for key in keys(file)
            object = file[key]
            object isa HDF5.Dataset || continue
            data[key] = read(object)
        end
    end
    return data
end

"""
    select_evolution_steps(total_steps::Int; steps_to_plot=nothing) -> Vector{Int}

Select which steps to plot from an evolution.
Default: initial, 25%, 50%, 75%, and final steps.
"""
function select_evolution_steps(total_steps::Int; steps_to_plot=nothing)::Vector{Int}
    total_steps < 1 && return Int[]

    if steps_to_plot !== nothing
        return unique([Int(step) for step in steps_to_plot if 1 <= Int(step) <= total_steps])
    end

    n_steps = min(total_steps, 5)
    return unique(round.(Int, range(1, total_steps; length=n_steps)))
end

"""
    save_figure(fig, base_dir::String, fig_name::String)

Save a figure to the Figs subdirectory, creating it if needed.
"""
function save_figure(fig, base_dir::String, fig_name::String)
    fig_dir = joinpath(base_dir, "Figs")
    mkpath(fig_dir)
    fig_path = joinpath(fig_dir, fig_name)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    println("Figure saved to $fig_path")
end

"""
    extract_filename_base(filepath::String) -> String

Extract the base filename without extension from a file path.
"""
function extract_filename_base(filepath::String)::String
    base = splitpath(filepath)[end]
    return replace(base, ".h5" => "")
end

"""
    setup_kspace_axis(ax, bc::Symbol)

Configure common axis settings for k-space plots.
"""
function setup_kspace_axis(ax, bc::Symbol)
    ax.set_xlim(0, 2)
    ax.grid(true, alpha=0.3)
end

"""
    get_evolution_colors(plt, n_steps::Int)

Generate a color array for evolution plots using viridis colormap.
"""
function get_evolution_colors(plt, n_steps::Int)
    n_steps <= 0 && return Any[]
    color_values = n_steps == 1 ? [0.0] : collect(range(0, 1; length=n_steps))
    return [plt.cm.viridis(value) for value in color_values]
end

"""
    mark_bath_detuning_energy!(ax, delta; reference_energies=nothing, kwargs...)

Mark a bath detuning on an energy axis.

In a dispersion plot, `delta` is an energy/frequency, not a momentum.  The
resonant modes are therefore the intersections of `epsilon_k` with the
horizontal line at the detuning energy. When `reference_energies` are supplied,
the marker uses the sign convention of the plotted dispersion.
"""
function mark_bath_detuning_energy!(
    ax,
    delta;
    reference_energies=nothing,
    color="red",
    linestyle="--",
    linewidth=2,
    label="signed |delta|",
    alpha=0.7,
)
    delta_energy = _bath_detuning_energy(delta; reference_energies=reference_energies)
    if delta_energy === nothing
        return nothing
    end

    return ax.axhline(
        y=delta_energy,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        alpha=alpha,
    )
end

"""
    _maybe_scalar(x)

If `x` is a 0-d or length-1 array (a common HDF5 scalar encoding), return its only
entry. Otherwise return `x` unchanged.
"""
_maybe_scalar(x) = (x isa AbstractArray && length(x) == 1) ? only(x) : x

function _bath_detuning_energy(delta; reference_energies=nothing)
    value = _maybe_scalar(delta)
    if value === nothing || value == 0
        return nothing
    end
    value isa Number || return nothing
    magnitude = abs(Float64(value))
    reference_energies === nothing && return magnitude

    energies = filter(isfinite, Float64.(vec(reference_energies)))
    isempty(energies) && return magnitude

    distance_to_positive = minimum(abs.(energies .- magnitude))
    distance_to_negative = minimum(abs.(energies .+ magnitude))
    return distance_to_negative < distance_to_positive ? -magnitude : magnitude
end

function _scalar_float_from_data(data::AbstractDict, key::AbstractString)
    haskey(data, key) || return nothing
    value = _maybe_scalar(data[key])
    value isa Number || return nothing
    return Float64(value)
end

"""
    nearest_bath_resonance_indices(mode_energies, delta; atol=1e-12)

Return every mode index whose quasiparticle energy is closest to the bath
detuning energy in the sign convention of `mode_energies`.
"""
function nearest_bath_resonance_indices(mode_energies, delta; atol=1e-12)
    isempty(mode_energies) && return Int[]
    delta_energy = _bath_detuning_energy(delta; reference_energies=mode_energies)
    delta_energy === nothing && return Int[]

    distances = abs.(Float64.(mode_energies) .- delta_energy)
    dmin = minimum(distances)
    return findall(d -> isapprox(d, dmin; atol=atol, rtol=sqrt(eps(Float64))), distances)
end

"""
    momentum_plot_mode_energies(data, k_values)

Return the mode energies associated with a momentum plot.

Stored `RESULT_MODE_ENERGIES` are preferred, since they are the energies saved
with the simulation. If they are absent, the helper falls back to the canonical
Ising dispersion when scalar `J` and `h` metadata are available. If neither
source is available, it returns `nothing` rather than inventing a resonance
coordinate.
"""
function momentum_plot_mode_energies(data::AbstractDict, k_values)
    if haskey(data, RESULT_MODE_ENERGIES)
        mode_energies = vec(Float64.(data[RESULT_MODE_ENERGIES]))
        if length(mode_energies) == length(k_values)
            return mode_energies
        end
        @warn "Skipping stored mode energies whose length does not match k_values" *
              " (got $(length(mode_energies)), expected $(length(k_values)))." *
              " Falling back to J,h if available."
    end

    J = _scalar_float_from_data(data, "J")
    h = _scalar_float_from_data(data, "h")
    if J !== nothing && h !== nothing
        return compute_energy_dispersion(k_values, J, h)
    end

    return nothing
end

"""
    mark_bath_resonance_momentum!(ax, k_values, mode_energies, delta; kwargs...)

Mark resonant momenta on a momentum-axis plot. The marker positions are computed
from the resonance condition `epsilon_k ~= |delta|`; the bath detuning itself is
not treated as a momentum coordinate.
"""
function mark_bath_resonance_momentum!(
    ax,
    k_values,
    mode_energies,
    delta;
    momentum_scale=pi,
    color="red",
    linestyle=":",
    linewidth=2,
    alpha=0.7,
    label="nearest epsilon_k ~= |delta|",
)
    indices = nearest_bath_resonance_indices(mode_energies, delta)
    isempty(indices) && return nothing

    handles = Any[]
    for (j, idx) in enumerate(indices)
        push!(
            handles,
            ax.axvline(
                x=k_values[idx] / momentum_scale,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=(j == 1 ? label : "_nolegend_"),
            ),
        )
    end
    return handles
end

function mark_bath_resonance_from_data!(ax, data::AbstractDict, k_values; momentum_scale=1)
    if !haskey(data, "delta") || _bath_detuning_energy(data["delta"]) === nothing
        return nothing
    end

    mode_energies = momentum_plot_mode_energies(data, k_values)
    if mode_energies === nothing
        @warn "Cannot mark bath resonance in momentum plot without mode energies or Ising parameters J,h."
        return nothing
    end

    return mark_bath_resonance_momentum!(
        ax,
        k_values,
        mode_energies,
        data["delta"];
        momentum_scale=momentum_scale,
    )
end

"""
    _normalize_momentum_distribution_by_step(momentum_dist, k_values)

Return the momentum distribution in the canonical plotting orientation
`(cooling step, momentum mode)`.

Current result files write `RESULT_MOMENTUM_DISTRIBUTION` with one row per
cooling step and one column per momentum mode.  Some older plotting scripts also
accepted the transpose; that legacy orientation is still accepted here.  If both
axes have length `length(k_values)`, the canonical result-file orientation is
preferred.
"""
function _normalize_momentum_distribution_by_step(momentum_dist, k_values)
    matrix = Float64.(momentum_dist)
    ndims(matrix) == 2 || throw(ArgumentError("Expected a matrix for $RESULT_MOMENTUM_DISTRIBUTION"))

    n_modes = length(k_values)
    if size(matrix, 2) == n_modes
        return matrix
    elseif size(matrix, 1) == n_modes
        return permutedims(matrix)
    end

    throw(
        DimensionMismatch(
            "$RESULT_MOMENTUM_DISTRIBUTION has size $(size(matrix)), " *
            "but $RESULT_K_VALUES has length $n_modes",
        ),
    )
end

"""
    kspace_evolution_plot_data(data)

Extract k-space evolution data from an HDF5 data dictionary using the canonical
result-key constants.  The returned `momentum_dist` has shape `(steps, modes)`.
"""
function kspace_evolution_plot_data(data::AbstractDict)
    if !haskey(data, RESULT_MOMENTUM_DISTRIBUTION) || !haskey(data, RESULT_K_VALUES)
        error("Expected HDF5 datasets \"$RESULT_MOMENTUM_DISTRIBUTION\" and \"$RESULT_K_VALUES\"")
    end

    k_values = vec(Float64.(data[RESULT_K_VALUES]))
    momentum_dist = _normalize_momentum_distribution_by_step(
        data[RESULT_MOMENTUM_DISTRIBUTION],
        k_values,
    )

    return (
        momentum_dist=momentum_dist,
        k_values=k_values,
        total_steps=size(momentum_dist, 1),
        N=Int(_maybe_scalar(get(data, "N", length(k_values)))),
        J=Float64(_maybe_scalar(get(data, "J", 1.0))),
        h=Float64(_maybe_scalar(get(data, "h", 1.0))),
        bc=Symbol(string(_maybe_scalar(get(data, "bc", "open")))),
    )
end

"""
    safe_read_keys(filename, keys...) -> Tuple

Read selected datasets from an HDF5 results file.
"""
function safe_read_keys(filename::AbstractString, dsets::AbstractString...)
    if !isfile(filename)
        @warn "File not found: $filename"
        return ntuple(_ -> nothing, length(dsets))
    end

    try
        h5open(filename, "r") do file
            available = keys(file)
            return ntuple(i -> (dsets[i] in available ? read(file, dsets[i]) : nothing), length(dsets))
        end
    catch e
        msg = if isa(e, HDF5.HDF5Error)
            "HDF5 error: $(e.msg)"
        else
            "Error: $e"
        end
        @warn "Failed to read $filename: $msg"
        return ntuple(_ -> nothing, length(dsets))
    end
end

"""
    safe_read_data(filename)

Backward-compatible helper returning `(e₀, E_list, GS_overlap_list, Edensity_final)`.
"""
safe_read_data(filename::AbstractString) =
    safe_read_keys(filename, "e₀", RESULT_ENERGY, RESULT_GROUND_STATE_OVERLAP, "Edensity_final")
