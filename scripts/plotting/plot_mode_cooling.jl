"""
    plot_mode_cooling.jl

Plot mode-resolved cooling: ⟨h_k⟩ vs cooling step for each quasiparticle mode k,
colored by the mode energy ε_k.

Can be called standalone (loads data from HDF5) or from the diagnostic script
via `plot_mode_cooling_from_data(...)`.

Usage (standalone):
    julia --project=. scripts/plotting/plot_mode_cooling.jl path/to/results.h5

Usage (from another script):
    include("scripts/plotting/plot_mode_cooling.jl")
    plot_mode_cooling_from_data(mode_hk, k_indices, εk_values; delta=Δ, savepath="fig.pdf")
"""

# Only include PlotUtils if not already loaded
if !@isdefined(get_pyplot)
    include(joinpath(@__DIR__, "PlotUtils.jl"))
end

"""
    plot_mode_cooling_from_data(mode_hk, k_indices, εk_values;
                                delta=nothing, savepath=nothing, title=nothing)

Plot ⟨h_k⟩ vs cooling step for each mode.

# Arguments
- `mode_hk`: Matrix of size (n_steps, n_modes) with ⟨h_k⟩ values
- `k_indices`: Vector of mode indices (integer or half-integer)
- `εk_values`: Vector of mode energies ε_k in code units
- `delta`: Bath detuning Δ (optional, used to highlight resonant mode)
- `savepath`: Path to save figure (optional)
- `title`: Figure title (optional)
"""
function plot_mode_cooling_from_data(mode_hk::AbstractMatrix, k_indices, εk_values;
                                     delta=nothing, savepath=nothing, title=nothing)
    plt = get_pyplot()
    n_steps, n_modes = size(mode_hk)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Normalize ε_k for colormap
    ε_min, ε_max = extrema(εk_values)
    norm = plt.matplotlib.colors.Normalize(vmin=ε_min, vmax=ε_max)
    cmap = plt.cm.coolwarm

    steps = 0:(n_steps - 1)

    # --- Left panel: ⟨h_k⟩ vs step ---
    for i in 1:n_modes
        color = cmap(norm(εk_values[i]))
        k_label = k_indices[i] isa Rational ? "$(numerator(k_indices[i]))/$(denominator(k_indices[i]))" : "$(k_indices[i])"
        ax1.plot(steps, mode_hk[:, i], color=color, linewidth=1.5,
                 label=L"k = %$k_label" * L", \varepsilon_k = %$(round(εk_values[i], digits=3))")
    end

    # Reference lines
    ax1.axhline(y=-1.0, color="black", linestyle="--", alpha=0.5, label=L"\langle h_k \rangle = -1 \; (\mathrm{GS})")
    ax1.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3)

    # Highlight resonant mode
    if delta !== nothing
        res_idx = argmin(abs.(εk_values .- abs(delta)))
        res_k = k_indices[res_idx]
        k_label = res_k isa Rational ? "$(numerator(res_k))/$(denominator(res_k))" : "$(res_k)"
        ax1.plot(steps, mode_hk[:, res_idx], color="red", linewidth=3, alpha=0.4,
                 label=L"resonant: k=%$k_label" * L", \Delta=%$(round(abs(delta), digits=3))")
    end

    ax1.set_xlabel("Cooling step")
    ax1.set_ylabel(L"\langle h_k \rangle")
    ax1.set_title(title !== nothing ? title : "Mode cooling evolution")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(true, alpha=0.3)
    ax1.set_ylim(-1.15, 1.15)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(pylist([]))
    cbar = fig.colorbar(sm, ax=ax1, shrink=0.8)
    cbar.set_label(L"\varepsilon_k \; \mathrm{(code\;units)}")

    # --- Right panel: final ⟨h_k⟩ vs ε_k ---
    initial_hk = mode_hk[1, :]
    final_hk = mode_hk[end, :]

    ax2.scatter(εk_values, initial_hk, marker="o", s=80, color="blue", alpha=0.6,
                label="Initial", zorder=3)
    ax2.scatter(εk_values, final_hk, marker="s", s=80, color="red", alpha=0.6,
                label="Final", zorder=3)
    ax2.axhline(y=-1.0, color="black", linestyle="--", alpha=0.5, label=L"h_k = -1 \; (\mathrm{GS})")

    if delta !== nothing
        ax2.axvline(x=abs(delta), color="green", linestyle="--", alpha=0.7,
                    label=L"\Delta = %$(round(abs(delta), digits=3))")
    end

    ax2.set_xlabel(L"\varepsilon_k \; \mathrm{(code\;units)}")
    ax2.set_ylabel(L"\langle h_k \rangle")
    ax2.set_title("Initial vs final mode occupation")
    ax2.legend(fontsize=9)
    ax2.grid(true, alpha=0.3)
    ax2.set_ylim(-1.15, 1.15)

    fig.tight_layout()

    if savepath !== nothing
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
        println("Figure saved to $savepath")
    end

    return fig
end

"""
    plot_mode_cooling_from_h5(filepath::String; savepath=nothing)

Load mode cooling data from an HDF5 file and plot.
Expected datasets: "mode_hk", "mode_k_indices", "mode_ek_values", optionally "delta".
"""
function plot_mode_cooling_from_h5(filepath::String; savepath=nothing)
    data = read_h5_data(filepath)
    data === nothing && error("Could not read $filepath")

    mode_hk = data["mode_hk"]
    k_indices = data["mode_k_indices"]
    εk_values = data["mode_ek_values"]
    delta = get(data, "delta", nothing)

    return plot_mode_cooling_from_data(mode_hk, k_indices, εk_values;
                                       delta=delta, savepath=savepath,
                                       title=extract_filename_base(filepath))
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
