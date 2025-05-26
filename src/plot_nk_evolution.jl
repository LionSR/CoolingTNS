"""
    plot_nk_evolution(filename; steps_to_plot=nothing, save_fig=true)

Plot the evolution of momentum distribution n_k during cooling process.
Shows how n_k approaches the ground state distribution.
"""
function plot_nk_evolution(filename; steps_to_plot=nothing, save_fig=true)
    plt = pyimport("matplotlib.pyplot")
    
    # Read data
    if !isfile(filename)
        @warn "File not found: $filename"
        return
    end
    
    data = Dict{String, Any}()
    h5open(filename, "r") do file
        for key in keys(file)
            data[key] = read(file, key)
        end
    end
    
    # Check if k-space data exists
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
    delta = get(data, "delta", 0.0)
    total_steps = size(momentum_dist, 2)
    
    # Determine which steps to plot
    if steps_to_plot === nothing
        # Default: plot initial, 25%, 50%, 75%, and final
        step_indices = unique([1, 
                              div(total_steps, 4), 
                              div(total_steps, 2), 
                              div(3*total_steps, 4), 
                              total_steps])
    else
        step_indices = steps_to_plot
    end
    
    # Compute ground state occupation
    # For TF Ising ground state: all k modes have n_k = 1/2 in paramagnetic phase
    # More generally: n_k^(GS) = (1/2)(1 - (J cos(k) + h)/√(J² + h² + 2Jh cos(k)))
    n_k_gs = [0.5 * (1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k))) for k in k_values]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color map for different steps
    colors = plt.cm.viridis(range(0, 1, length=length(step_indices)))
    
    # Plot momentum distribution at different steps
    for (idx, (step_idx, color)) in enumerate(zip(step_indices, colors))
        if step_idx <= total_steps
            label = step_idx == 1 ? "Initial" : "Step $(step_idx-1)"
            ax.plot(k_values/π, momentum_dist[:, step_idx], "o-", 
                   color=color, linewidth=2, markersize=6, label=label)
        end
    end
    
    # Plot ground state distribution
    ax.plot(k_values/π, n_k_gs, "k--", linewidth=2.5, label="Ground state")
    
    # Add vertical line for delta if provided
    if delta != 0
        ax.axvline(x=delta/π, color="red", linestyle=":", linewidth=2, label="δ/π", alpha=0.7)
    end
    
    # Formatting
    ax.set_xlabel("k/π", fontsize=14)
    ax.set_ylabel("n_k", fontsize=14)
    ax.set_title("Momentum Distribution Evolution\n(N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
    ax.grid(true, alpha=0.3)
    ax.legend(loc="best", fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    
    # Set x-axis limits based on BC
    if bc == :periodic
        ax.set_xlim(0, 2)
    else  # antiperiodic
        ax.set_xlim(0, 2)
    end
    
    plt.tight_layout()
    
    # Save figure
    if save_fig
        base_filename = splitpath(filename)[end]
        base_filename = replace(base_filename, ".h5" => "")
        fig_filename = joinpath("Results", "Figs", "nk_evolution_$(base_filename).pdf")
        mkpath(dirname(fig_filename))
        fig.savefig(fig_filename, dpi=300, bbox_inches="tight")
        println("n_k evolution plot saved to $fig_filename")
    end
    
    return fig
end

# Export the function
export plot_nk_evolution