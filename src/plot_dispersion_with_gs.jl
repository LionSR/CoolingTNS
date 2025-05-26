"""
    plot_dispersion_with_ground_state(N, J, h, bc; delta=nothing, save_fig=true, filename=nothing)

Plot the energy dispersion e_k vs k and ground state occupation n_k^(GS) for the transverse field Ising model.
"""
function plot_dispersion_with_ground_state(N::Int, J::Real, h::Real, bc::Symbol; 
                                         delta=nothing, save_fig=true, filename=nothing)
    plt = pyimport("matplotlib.pyplot")
    
    # Generate k values based on boundary conditions
    if bc == :periodic
        k_values = [2π * n / N for n in 0:N-1]
    elseif bc == :antiperiodic
        k_values = [π * (2n + 1) / N for n in 0:N-1]
    else
        error("Unsupported boundary condition: $bc")
    end
    
    # Sort k values for plotting
    k_sorted = sort(k_values)
    
    # Compute energy dispersion
    # ε_k = -2√(J² + h² + 2Jh cos(k))
    e_k = [-2 * sqrt(J^2 + h^2 + 2*J*h*cos(k)) for k in k_sorted]
    
    # Compute ground state occupation
    # For ground state: n_k = 0 if ε_k > 0, n_k = 1 if ε_k < 0
    # Since all ε_k < 0 for this model, we need the actual formula:
    # n_k^(GS) = (1/2)(1 - (J cos(k) + h)/√(J² + h² + 2Jh cos(k)))
    n_k_gs = [0.5 * (1 - (J*cos(k) + h)/sqrt(J^2 + h^2 + 2*J*h*cos(k))) for k in k_sorted]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot energy dispersion on left y-axis
    ax1.plot(k_sorted / π, e_k, "b-", linewidth=2, label="ε_k")
    ax1.set_xlabel("k/π", fontsize=14)
    ax1.set_ylabel("ε_k", fontsize=14, color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    
    # Plot ground state occupation on right y-axis
    ax2.plot(k_sorted / π, n_k_gs, "g--", linewidth=2, label="n_k^(GS)")
    ax2.set_ylabel("n_k^(GS)", fontsize=14, color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.set_ylim(-0.1, 1.1)
    
    # Add vertical line for delta if provided
    if delta !== nothing && delta != 0
        ax1.axvline(x=delta/π, color="red", linestyle="--", linewidth=2, label="δ/π", alpha=0.7)
    end
    
    # Formatting
    ax1.set_title("Energy Dispersion and Ground State Occupation\n(N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
    ax1.grid(true, alpha=0.3)
    
    # Set x-axis limits
    if bc == :periodic
        ax1.set_xlim(0, 2)
    else  # antiperiodic
        ax1.set_xlim(0, 2)
    end
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    if save_fig && filename !== nothing
        fig_filename = joinpath("Results", "Figs", "dispersion_with_gs_$(filename).pdf")
        mkpath(dirname(fig_filename))
        fig.savefig(fig_filename, dpi=300, bbox_inches="tight")
        println("Dispersion plot with ground state saved to $fig_filename")
    end
    
    return fig
end

# Export the function
export plot_dispersion_with_ground_state