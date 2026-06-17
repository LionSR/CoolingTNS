"""
Shared plotting utilities for k-space visualization.
Provides common helper functions to reduce duplication across plotting scripts.

Usage: `include("PlotUtils.jl")` from other scripts in this directory.
"""

using CoolingTNS
using HDF5
using PythonCall
using LaTeXStrings
using Printf

# Re-export dispersion functions from CoolingTNS for convenience
using CoolingTNS: bath_detuning_energy,
                  generate_k_values,
                  compute_energy_dispersion,
                  compute_ground_state_occupation

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
            object = file[key]
            try
                object isa HDF5.Group && continue
                data[key] = read(object)
            finally
                close(object)
            end
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

_detuning_label_value(δ_abs::Real) = @sprintf("%.6g", δ_abs)
_detuning_label_value(δ_abs) = string(δ_abs)

"""
    add_detuning_energy_marker!(ax, delta; color="red", linestyle="--", linewidth=2, alpha=0.7,
                                label=nothing)

Draw a bath detuning as a horizontal energy marker on a dispersion axis.

The transverse-field Ising dispersion is plotted as `epsilon_k` versus `k/pi`,
so the resonant bath line belongs on the energy axis at `|delta|`, not on the
momentum axis at `delta/pi`. Returns the plotted energy, or `nothing` when
`delta` does not specify a single nonzero detuning. The default legend label
includes the plotted value of `|delta|`; pass `label` to override it.
"""
function add_detuning_energy_marker!(ax, delta;
                                     color="red", linestyle="--", linewidth=2, alpha=0.7,
                                     label=nothing)
    δ_abs = bath_detuning_energy(delta)
    δ_abs === nothing && return nothing

    line_label = label === nothing ? L"|\Delta| = %$(_detuning_label_value(δ_abs))" : label
    ax.axhline(y=δ_abs, color=color, linestyle=linestyle, linewidth=linewidth,
               alpha=alpha, label=line_label)
    return δ_abs
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
