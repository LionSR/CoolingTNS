"""
Plot evolution of momentum distribution n_k during cooling.

Standalone plotting script. Usage:
    julia --project=. scripts/plotting/plot_nk_evolution.jl <filename.h5>
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

"""
    plot_nk_evolution(filename; steps_to_plot=nothing, save_fig=true)

Plot the evolution of momentum distribution n_k during cooling process.
Shows how n_k approaches the ground state distribution.
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

    # Handle both possible orientations of momentum_dist
    if size(momentum_dist, 1) == length(k_values)
        total_steps = size(momentum_dist, 2)
    else
        momentum_dist = transpose(momentum_dist)
        total_steps = size(momentum_dist, 2)
    end

    step_indices = select_evolution_steps(total_steps; steps_to_plot=steps_to_plot)
    n_k_gs = compute_ground_state_occupation(k_values, J, h)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_evolution_colors(plt, length(step_indices))

    for (idx, step_idx) in enumerate(step_indices)
        if step_idx <= total_steps
            label = step_idx == 1 ? "Initial" : "Step $(step_idx-1)"
            ax.plot(k_values/pi, momentum_dist[:, step_idx], "o-",
                   color=colors[idx], linewidth=2, markersize=6, label=label)
        end
    end

    ax.plot(k_values/pi, n_k_gs, "k--", linewidth=2.5, label="Ground state")


    ax.set_xlabel("k/pi", fontsize=14)
    ax.set_ylabel("n_k", fontsize=14)
    ax.set_title("Momentum Distribution Evolution\n(N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
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
