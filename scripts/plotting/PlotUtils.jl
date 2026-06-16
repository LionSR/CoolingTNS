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
    n_steps == 1 && return [plt.cm.viridis(0.0)]
    return plt.cm.viridis(range(0, 1, length=n_steps))
end

"""
    _maybe_scalar(x)

If `x` is a 0-d or length-1 array (a common HDF5 scalar encoding), return its only
entry. Otherwise return `x` unchanged.
"""
_maybe_scalar(x) = (x isa AbstractArray && length(x) == 1) ? only(x) : x

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
