"""
Plot evolution of raw Fourier occupations during cooling.

Standalone plotting script. Usage:
    julia --project=. scripts/plotting/plot_nk_evolution.jl <filename.h5>
"""

if !isdefined(@__MODULE__, :get_pyplot)
    Base.include(@__MODULE__, joinpath(@__DIR__, "PlotUtils.jl"))
end

if !isdefined(@__MODULE__, :_COOLINGTNS_PLOT_NK_EVOLUTION_INCLUDED)
const _COOLINGTNS_PLOT_NK_EVOLUTION_INCLUDED = true

"""
    plot_nk_evolution(filename; steps_to_plot=nothing, save_fig=true)

Plot the evolution of the raw Fourier occupation
``\\tilde n_k = \\langle \\tilde a_k^\\dagger \\tilde a_k\\rangle`` during the
cooling process. This is not the Bogoliubov quasiparticle occupation
``n_k^{\\mathrm{Bog}}``; use `plot_mode_cooling.jl` for the latter.  The
dashed reference curve is the parity-unconstrained BdG raw Fourier reference on
the stored fermionic grid, not necessarily the fixed-parity sector ground-state
occupation.
"""
function plot_nk_evolution(filename; steps_to_plot=nothing, save_fig=true)
    plt = get_pyplot()

    data = read_h5_data(filename)
    data === nothing && return

    if !haskey(data, CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION) ||
       !haskey(data, CoolingTNS.RESULT_K_VALUES)
        @warn "No k-space data found in file $filename"
        return
    end

    momentum_dist = data[CoolingTNS.RESULT_MOMENTUM_DISTRIBUTION]
    k_values = data[CoolingTNS.RESULT_K_VALUES]
    N = data["N"]
    J = get(data, "J", 1.0)
    h = get(data, "h", 1.0)
    bc = Symbol(get(data, "bc", "open"))

    momentum_dist = _momentum_distribution_modes_by_steps(momentum_dist, length(k_values))
    total_steps = size(momentum_dist, 2)

    step_indices = select_evolution_steps(total_steps; steps_to_plot=steps_to_plot)
    n_k_ref = compute_bdg_reference_occupation(k_values, J, h)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_evolution_colors(plt, length(step_indices))

    for (idx, step_idx) in enumerate(step_indices)
        if step_idx <= total_steps
            label = step_idx == 1 ? "Initial" : "Step $(step_idx-1)"
            ax.plot(k_values/pi, momentum_dist[:, step_idx], "o-",
                   color=colors[idx], linewidth=2, markersize=6, label=label)
        end
    end

    ax.plot(k_values/pi, n_k_ref, "k--", linewidth=2.5,
            label=RAW_FOURIER_BDG_REFERENCE_OCCUPATION_LABEL)


    ax.set_xlabel("k/pi", fontsize=14)
    ax.set_ylabel(RAW_FOURIER_OCCUPATION_LABEL, fontsize=14)
    ax.set_title("Raw Fourier Occupation Evolution\n(N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best", fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(minimum(k_values) / pi, maximum(k_values) / pi)
    plt.tight_layout()

    if save_fig
        base_filename = extract_filename_base(filename)
        save_figure(fig, dirname(filename), "nk_evolution_$(base_filename).pdf")
    end

    return fig
end

end
