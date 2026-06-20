"""
    plot_mode_cooling.jl

Plot mode-resolved cooling: Bogoliubov quasiparticle occupation
``n_k^{Bog}`` vs cooling step for each mode k, colored by the mode energy ε_k.

Can be called standalone (loads data from HDF5) or from the diagnostic script
via `plot_mode_occupation_from_data(...)`.

Usage (standalone):
    julia --project=. scripts/plotting/plot_mode_cooling.jl path/to/results.h5

Usage (from another script):
    include("scripts/plotting/plot_mode_cooling.jl")
    plot_mode_occupation_from_data(mode_nk, k_indices, εk_values; delta=Δ, savepath="fig.pdf")
"""

# Only include PlotUtils if not already loaded
if !isdefined(@__MODULE__, :get_pyplot)
    Base.include(@__MODULE__, joinpath(@__DIR__, "PlotUtils.jl"))
end

if !isdefined(@__MODULE__, :_COOLINGTNS_PLOT_MODE_COOLING_INCLUDED)
const _COOLINGTNS_PLOT_MODE_COOLING_INCLUDED = true

using CoolingTNS:
    RESULT_MODE_HK,
    RESULT_MODE_NK,
    RESULT_MODE_K_INDICES,
    RESULT_MODE_ENERGIES,
    RESULT_MODE_MEASUREMENT_CYCLES,
    mode_occupation_from_hk,
    bath_detuning_energy,
    nearest_bath_resonance_indices

"""
    _mode_occupation_from_plot_data(data)

Return the Bogoliubov quasiparticle occupation matrix for mode-cooling plots.

New result files store `RESULT_MODE_NK` directly. Older files stored only `RESULT_MODE_HK`,
with `h_k = 2n_k - 1`; those files are converted through
`CoolingTNS.mode_occupation_from_hk`, which is the single source of truth for
this convention.
"""
function _mode_occupation_from_plot_data(data::AbstractDict)
    if haskey(data, RESULT_MODE_NK)
        return Float64.(data[RESULT_MODE_NK])
    elseif haskey(data, RESULT_MODE_HK)
        return Float64.(mode_occupation_from_hk(data[RESULT_MODE_HK]))
    end
    error("Expected HDF5 dataset \"$RESULT_MODE_NK\" or legacy dataset \"$RESULT_MODE_HK\"")
end

function _occupation_ylim(mode_nk)
    finite_values = filter(isfinite, vec(Float64.(mode_nk)))
    isempty(finite_values) && return (-0.05, 1.05)
    return (
        min(-0.05, minimum(finite_values) - 0.05),
        max(1.05, maximum(finite_values) + 0.05),
    )
end

_mode_index_label(k) = k isa Rational ? "$(numerator(k))/$(denominator(k))" : "$(k)"

"""
    plot_mode_occupation_from_data(mode_nk, k_indices, εk_values;
                                   delta=nothing, savepath=nothing, title=nothing,
                                   measurement_cycles=nothing)

Plot the Bogoliubov quasiparticle occupation ``n_k^{Bog}`` vs cooling step for
each mode.

# Arguments
- `mode_nk`: Matrix of size (n_steps, n_modes) with occupation values
  ``n_k^{Bog}``
- `k_indices`: Vector of mode indices (integer or half-integer)
- `εk_values`: Vector of mode energies ε_k in code units
- `delta`: Bath detuning Δ (optional, used to highlight resonant mode)
- `savepath`: Path to save figure (optional)
- `title`: Figure title (optional)
- `measurement_cycles`: Optional zero-based physical cooling cycles for measured
  rows.  When omitted, every row is treated as measured.
"""
function plot_mode_occupation_from_data(mode_nk::AbstractMatrix, k_indices, εk_values;
                                        delta=nothing, savepath=nothing, title=nothing,
                                        measurement_cycles=nothing)
    plt = get_pyplot()
    n_steps, n_modes = size(mode_nk)
    length(k_indices) == n_modes || throw(ArgumentError("k_indices length must match mode_nk columns"))
    length(εk_values) == n_modes || throw(ArgumentError("εk_values length must match mode_nk columns"))
    measured = _mode_measurement_cycle_rows(n_steps, measurement_cycles)
    cycles = measured.cycles
    measured_mode_nk = Float64.(mode_nk[measured.rows, :])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Normalize ε_k for colormap
    ε_min, ε_max = extrema(εk_values)
    norm = plt.matplotlib.colors.Normalize(vmin=ε_min, vmax=ε_max)
    cmap = plt.cm.coolwarm

    # --- Left panel: Bogoliubov occupation vs step ---
    for i in 1:n_modes
        color = cmap(norm(εk_values[i]))
        k_label = _mode_index_label(k_indices[i])
        ax1.plot(cycles, measured_mode_nk[:, i], color=color, linewidth=1.5,
                 label=L"k = %$k_label" * L", \varepsilon_k = %$(round(εk_values[i], digits=3))")
    end

    # Reference lines
    ax1.axhline(y=0.0, color="black", linestyle="--", alpha=0.5,
                label=BOGOLIUBOV_GS_OCCUPATION_LABEL)
    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3,
                label=BOGOLIUBOV_HALF_OCCUPATION_LABEL)
    ax1.axhline(y=1.0, color="black", linestyle=":", alpha=0.25)

    # Highlight all resonant modes closest to the bath detuning.
    δ_abs = bath_detuning_energy(delta)
    if δ_abs !== nothing
        for res_idx in nearest_bath_resonance_indices(εk_values, delta)
            k_label = _mode_index_label(k_indices[res_idx])
            ax1.plot(cycles, measured_mode_nk[:, res_idx], color="red", linewidth=3, alpha=0.4,
                     label=L"resonant: k=%$k_label" * L", \Delta=%$(round(δ_abs, digits=3))")
        end
    end

    ax1.set_xlabel("Cooling cycle")
    ax1.set_ylabel(BOGOLIUBOV_OCCUPATION_LABEL)
    ax1.set_title(title !== nothing ? title : "Bogoliubov Mode Occupation Evolution")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(true, alpha=0.3)
    ymin, ymax = _occupation_ylim(measured_mode_nk)
    ax1.set_ylim(ymin, ymax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(pylist([]))
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.8)
    cbar.set_label(L"\varepsilon_k \; \mathrm{(code\;units)}")

    # --- Right panel: final Bogoliubov occupation vs ε_k ---
    initial_nk = measured_mode_nk[1, :]
    final_nk = measured_mode_nk[end, :]
    initial_label = first(cycles) == 0 ? "Initial" : "Cycle $(first(cycles))"
    final_label = last(cycles) == n_steps - 1 ? "Final" : "Cycle $(last(cycles))"

    ax2.scatter(εk_values, initial_nk, marker="o", s=80, color="blue", alpha=0.6,
                label=initial_label, zorder=3)
    ax2.scatter(εk_values, final_nk, marker="s", s=80, color="red", alpha=0.6,
                label=final_label, zorder=3)
    ax2.axhline(y=0.0, color="black", linestyle="--", alpha=0.5,
                label=BOGOLIUBOV_GS_OCCUPATION_LABEL)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3,
                label=BOGOLIUBOV_HALF_OCCUPATION_LABEL)
    ax2.axhline(y=1.0, color="black", linestyle=":", alpha=0.25)

    if δ_abs !== nothing
        ax2.axvline(x=δ_abs, color="green", linestyle="--", alpha=0.7,
                    label=L"\Delta = %$(round(δ_abs, digits=3))")
    end

    ax2.set_xlabel(L"\varepsilon_k \; \mathrm{(code\;units)}")
    ax2.set_ylabel(BOGOLIUBOV_OCCUPATION_LABEL)
    ax2.set_title("Initial and Final Bogoliubov Occupation")
    ax2.legend(fontsize=9)
    ax2.grid(true, alpha=0.3)
    ax2.set_ylim(ymin, ymax)

    fig.tight_layout()

    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        println("Figure saved to $savepath")
    end

    return fig
end

"""
    plot_mode_cooling_from_data(mode_hk, k_indices, εk_values;
                                delta=nothing, savepath=nothing, title=nothing,
                                measurement_cycles=nothing)

Legacy entry point accepting the Bogoliubov observable `h_k`. The plotted
quantity is still the Bogoliubov occupation `n_k^Bog`, obtained from
`CoolingTNS.mode_occupation_from_hk`.  The optional `measurement_cycles`
keyword has the same convention as in `plot_mode_occupation_from_data`.
"""
function plot_mode_cooling_from_data(mode_hk::AbstractMatrix, k_indices, εk_values;
                                     delta=nothing, savepath=nothing, title=nothing,
                                     measurement_cycles=nothing)
    return plot_mode_occupation_from_data(mode_occupation_from_hk(mode_hk), k_indices, εk_values;
                                          delta=delta, savepath=savepath, title=title,
                                          measurement_cycles=measurement_cycles)
end

"""
    plot_mode_cooling_from_h5(filepath::String; savepath=nothing)

Load mode cooling data from an HDF5 file and plot.
Expected datasets are named by `CoolingTNS.RESULT_MODE_NK` or legacy
`CoolingTNS.RESULT_MODE_HK`, together with `CoolingTNS.RESULT_MODE_K_INDICES`
and `CoolingTNS.RESULT_MODE_ENERGIES`; the optional detuning dataset is
`"delta"`.
"""
function plot_mode_cooling_from_h5(filepath::String; savepath=nothing)
    data = read_h5_data(filepath)
    data === nothing && error("Could not read $filepath")

    mode_nk = _mode_occupation_from_plot_data(data)
    k_indices = data[RESULT_MODE_K_INDICES]
    εk_values = data[RESULT_MODE_ENERGIES]
    delta = get(data, "delta", nothing)
    measurement_cycles = get(data, RESULT_MODE_MEASUREMENT_CYCLES, nothing)

    return plot_mode_occupation_from_data(mode_nk, k_indices, εk_values;
                                          delta=delta, savepath=savepath,
                                          title=extract_filename_base(filepath),
                                          measurement_cycles=measurement_cycles)
end

end

# Standalone execution
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia --project=. scripts/plotting/plot_mode_cooling.jl <results.h5> [output.pdf]")
        exit(1)
    end
    filepath = ARGS[1]
    savepath = length(ARGS) >= 2 ? ARGS[2] : replace(filepath, ".h5" => "_mode_cooling.pdf")
    fig = plot_mode_cooling_from_h5(filepath; savepath=savepath)
    get_pyplot().show()
end
