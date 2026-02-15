"""
Plot evolution of energy e_k = ε_k * n_k during cooling.

Standalone plotting script. Usage:
    julia --project=. scripts/plotting/plot_ek_evolution.jl <filename.h5>
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

"""
    plot_ek_evolution(filename; steps_to_plot=nothing, save_fig=true)

Plot the evolution of energy e_k = epsilon_k * n_k during cooling process.
Shows how the energy distribution in k-space evolves.
"""
function plot_ek_evolution(filename; steps_to_plot=nothing, save_fig=true)
    plt = get_pyplot()

    data = read_h5_data(filename)
    data === nothing && return

    if !haskey(data, "momentum_dist") || !haskey(data, "k_values")
        @warn "No k-space data found in file $filename"
        return
    end

    momentum_dist = data["momentum_dist"]
    k_values = data["k_values"]
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
    epsilon_k = compute_energy_dispersion(k_values, J, h)
    n_k_gs = compute_ground_state_occupation(k_values, J, h)
    e_k_gs = epsilon_k .* n_k_gs

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = get_evolution_colors(plt, length(step_indices))

    for (idx, step_idx) in enumerate(step_indices)
        if step_idx <= total_steps
            e_k = epsilon_k .* momentum_dist[:, step_idx]
            label = step_idx == 1 ? "Initial" : "Step $(step_idx-1)"
            ax.plot(k_values/pi, e_k, "o-",
                   color=colors[idx], linewidth=2, markersize=6, label=label)
        end
    end

    ax.plot(k_values/pi, e_k_gs, "k--", linewidth=2.5, label="Ground state")
    ax.plot(k_values/pi, epsilon_k, ":", color="gray", linewidth=1.5, label="epsilon_k", alpha=0.7)


    ax.set_xlabel("k/pi", fontsize=14)
    ax.set_ylabel("e_k = epsilon_k n_k", fontsize=14)
    ax.set_title("Energy Distribution Evolution\n(N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best", fontsize=12)
    ax.set_xlim(minimum(k_values) / pi, maximum(k_values) / pi)
    plt.tight_layout()

    if save_fig
        base_filename = extract_filename_base(filename)
        save_figure(fig, dirname(filename), "ek_evolution_$(base_filename).pdf")
    end

    return fig
end
