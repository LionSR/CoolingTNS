"""
Shared plotting utilities for k-space visualization.
Provides common helper functions to reduce duplication across plotting scripts.

Usage: `include("PlotUtils.jl")` from other scripts in this directory.
"""

using CoolingTNS
using HDF5
using PythonCall
using LaTeXStrings

# Re-export dispersion functions from CoolingTNS for convenience
using CoolingTNS: generate_k_values, compute_energy_dispersion, compute_ground_state_occupation

const _MODE_ENERGIES_DATASET = "mode_ek_values"

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

Read all data from an HDF5 file into a dictionary.
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
            data[key] = read(file, key)
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
    if steps_to_plot !== nothing
        return steps_to_plot
    end
    return unique([1,
                   div(total_steps, 4),
                   div(total_steps, 2),
                   div(3*total_steps, 4),
                   total_steps])
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
    return plt.cm.viridis(range(0, 1, length=n_steps))
end

"""
    _maybe_scalar(x)

If `x` is a 0-d or length-1 array (a common HDF5 scalar encoding), return its only
entry. Otherwise return `x` unchanged.
"""
_maybe_scalar(x) = (x isa AbstractArray && length(x) == 1) ? only(x) : x

function _bath_detuning_energy(delta)
    δ = _maybe_scalar(delta)
    if δ === nothing || δ == 0
        return nothing
    end
    return abs(δ)
end

"""
    mark_bath_detuning_energy!(ax, delta; kwargs...)

Draw the bath detuning on an energy axis.  The detuning `delta` has the same
units as the quasiparticle energy `epsilon_k`, so dispersion plots should mark
it horizontally rather than as a momentum coordinate.
"""
function mark_bath_detuning_energy!(ax, delta;
                                    color="red", linestyle="--", linewidth=2,
                                    alpha=0.7, label=L"bath $|\Delta|$")
    δ_abs = _bath_detuning_energy(delta)
    if δ_abs === nothing
        return nothing
    end
    return ax.axhline(y=δ_abs, color=color, linestyle=linestyle,
                      linewidth=linewidth, alpha=alpha, label=label)
end

"""
    nearest_bath_resonance_indices(εk_values, delta; atol=1e-12)

Return all mode indices whose quasiparticle energy is closest to `|delta|`.
This is the correct way to place a resonance marker on plots whose horizontal
axis is momentum.
"""
function nearest_bath_resonance_indices(εk_values, delta; atol=1e-12)
    δ_abs = _bath_detuning_energy(delta)
    if δ_abs === nothing
        return Int[]
    end
    isempty(εk_values) && return Int[]
    distances = abs.(εk_values .- δ_abs)
    dmin = minimum(distances)
    return findall(d -> isapprox(d, dmin; atol=atol, rtol=sqrt(eps(Float64))), distances)
end

"""
    mark_bath_resonance_momentum!(ax, k_values, εk_values, delta; kwargs...)

Draw vertical markers at the momentum values of the modes closest to resonance,
`ε_k ≈ |delta|`.  This should be used on occupation-vs-momentum plots; it does
not identify the bath detuning itself with a momentum.
"""
function mark_bath_resonance_momentum!(ax, k_values, εk_values, delta;
                                       momentum_scale=pi, color="red",
                                       linestyle=":", linewidth=2, alpha=0.7,
                                       label=L"nearest $\varepsilon_k \approx |\Delta|$")
    indices = nearest_bath_resonance_indices(εk_values, delta)
    isempty(indices) && return nothing

    handles = Any[]
    for (j, idx) in enumerate(indices)
        push!(handles, ax.axvline(x=k_values[idx] / momentum_scale,
                                  color=color, linestyle=linestyle,
                                  linewidth=linewidth, alpha=alpha,
                                  label=(j == 1 ? label : "_nolegend_")))
    end
    return handles
end

function _scalar_float_from_data(data::AbstractDict, key::AbstractString)
    haskey(data, key) || return nothing
    value = _maybe_scalar(data[key])
    value isa Number || return nothing
    return Float64(value)
end

"""
    momentum_plot_mode_energies(data, k_values)

Return the quasiparticle energies that correspond to `k_values` in an HDF5 plot
data dictionary.  Stored mode energies are preferred because they are the
energies actually measured during the simulation.  If those are unavailable,
the Ising parameters `J` and `h` are used with the canonical dispersion helper.
If neither source is present, return `nothing`.
"""
function momentum_plot_mode_energies(data::AbstractDict, k_values)
    if haskey(data, _MODE_ENERGIES_DATASET)
        εk_values = vec(Float64.(data[_MODE_ENERGIES_DATASET]))
        if length(εk_values) == length(k_values)
            return εk_values
        end
        @warn "Skipping stored mode energies whose length does not match k_values" *
              " (got $(length(εk_values)), expected $(length(k_values)))." *
              " Falling back to J,h if available."
    end

    J = _scalar_float_from_data(data, "J")
    h = _scalar_float_from_data(data, "h")
    if J !== nothing && h !== nothing
        return compute_energy_dispersion(k_values, J, h)
    end

    return nothing
end

function mark_bath_resonance_from_data!(ax, data::AbstractDict, k_values; momentum_scale=1)
    if !haskey(data, "delta") || data["delta"] === nothing
        return nothing
    end

    εk_values = momentum_plot_mode_energies(data, k_values)
    if εk_values === nothing
        @warn "Cannot mark bath resonance in momentum plot without mode energies or Ising parameters J,h."
        return nothing
    end

    return mark_bath_resonance_momentum!(
        ax, k_values, εk_values, data["delta"]; momentum_scale=momentum_scale
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
    safe_read_keys(filename, "e₀", "E_list", "GS_overlap_list", "Edensity_final")
