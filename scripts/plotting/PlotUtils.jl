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

if !isdefined(@__MODULE__, :_COOLINGTNS_PLOTUTILS_INCLUDED)
const _COOLINGTNS_PLOTUTILS_INCLUDED = true

# Lazy pyplot access - imported once per session
const _pyplot = Ref{Py}()

const RAW_FOURIER_OCCUPATION_LABEL = L"Raw Fourier occupation $\tilde n_k$"
const RAW_FOURIER_GS_OCCUPATION_LABEL = L"$\tilde n_k^{\mathrm{GS}}$"
const BOGOLIUBOV_OCCUPATION_LABEL = L"Bogoliubov occupation $n_k^{\mathrm{Bog}}$"
const BOGOLIUBOV_GS_OCCUPATION_LABEL = L"$n_k^{\mathrm{Bog}} = 0 \; (\mathrm{GS})$"
const BOGOLIUBOV_HALF_OCCUPATION_LABEL = L"$n_k^{\mathrm{Bog}} = 1/2$"

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
Default: initial, 25%, 50%, 75%, and final steps, clamped to the
available interval `1:total_steps`. Explicit `steps_to_plot` values must also
lie in this interval; impossible requests throw `ArgumentError`.
"""
function select_evolution_steps(total_steps::Int; steps_to_plot=nothing)::Vector{Int}
    total_steps >= 1 || throw(ArgumentError("total_steps must be positive; got $total_steps"))
    if steps_to_plot !== nothing
        steps = Int.(collect(steps_to_plot))
        all(step -> 1 <= step <= total_steps, steps) || throw(ArgumentError(
            "steps_to_plot must lie in 1:$total_steps; got $steps",
        ))
        return steps
    end
    return unique(clamp.([1,
                          div(total_steps, 4),
                          div(total_steps, 2),
                          div(3*total_steps, 4),
                          total_steps],
                         1, total_steps))
end

"""
    _mode_measurement_cycle_rows(n_rows, measurement_cycles=nothing)

Return the measured zero-based cooling cycles and their one-based matrix rows.
Mode-observable result arrays keep the full cycle-by-mode shape; when strided
measurements are used, the unmeasured rows are intentionally left as `NaN` and
the physically measured cycles are stored in
`CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES`.
"""
function _mode_measurement_cycle_rows(n_rows::Integer, measurement_cycles=nothing)
    n_rows >= 1 || throw(ArgumentError("mode-observable arrays must have at least one row"))

    if measurement_cycles === nothing
        cycles = collect(0:(n_rows - 1))
    else
        cycles = Int.(vec(measurement_cycles))
        isempty(cycles) && throw(ArgumentError("mode measurement cycle list is empty"))
        issorted(cycles) || throw(ArgumentError(
            "mode measurement cycles must be sorted; got $cycles",
        ))
        length(unique(cycles)) == length(cycles) || throw(ArgumentError(
            "mode measurement cycles must be unique; got $cycles",
        ))
        all(cycle -> 0 <= cycle < n_rows, cycles) || throw(ArgumentError(
            "mode measurement cycles must lie in 0:$(n_rows - 1); got $cycles",
        ))
    end

    return (cycles=cycles, rows=cycles .+ 1)
end

function _mode_measurement_cycle_rows(data::AbstractDict, n_rows::Integer)
    cycles = get(data, CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES, nothing)
    return _mode_measurement_cycle_rows(n_rows, cycles)
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
    _maybe_show_figure(plt, show_fig::Bool) -> Bool

Display matplotlib figures only when explicitly requested. Returns whether
display was requested.
"""
function _maybe_show_figure(plt, show_fig::Bool)::Bool
    show_fig || return false
    plt.show()
    return true
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

Generate a Julia-indexed color vector for evolution plots using the viridis colormap.
"""
function get_evolution_colors(plt, n_steps::Int)
    return pyconvert(Vector, plt.cm.viridis(range(0, 1, length=n_steps)))
end

"""
    _momentum_distribution_modes_by_steps(values, n_modes::Int; name=RESULT_MOMENTUM_DISTRIBUTION)

Return a copy of a momentum-distribution history in modes-by-steps orientation.

The plotting orientation is one row per momentum point and one column per saved
cooling step. Current result writers store the transpose, with one row per
cooling step and one column per momentum point. This helper accepts both layouts
and rejects arrays whose shape cannot be reconciled with the plotted momentum
grid. If both axes match `n_modes`, the stored-array convention is ambiguous, so
the helper prefers the current writer contract and transposes.
"""
function _momentum_distribution_modes_by_steps(values::AbstractMatrix, n_modes::Int;
                                               name::AbstractString=RESULT_MOMENTUM_DISTRIBUTION)
    if size(values, 2) == n_modes
        return Float64.(transpose(values))
    elseif size(values, 1) == n_modes
        return Float64.(values)
    end
    throw(DimensionMismatch(
        "$name has size $(size(values)), but expected one dimension to match $n_modes momentum points",
    ))
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

end
