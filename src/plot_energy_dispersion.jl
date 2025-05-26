"""
    plot_energy_dispersion(N, J, h, bc; delta=nothing, save_fig=true, filename=nothing)

Plot the energy dispersion e_k vs k for the transverse field Ising model.
For periodic BC: k = 2πn/N where n = 0, 1, ..., N-1
For antiperiodic BC: k = π(2n+1)/N where n = 0, 1, ..., N-1
"""
function plot_energy_dispersion(N::Int, J::Real, h::Real, bc::Symbol; 
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
    # After Jordan-Wigner: H = Σ_k ε_k (a†_k a_k - 1/2)
    # where ε_k = -2√(J² + h² + 2Jh cos(k)) for the Ising model
    e_k = [-2 * sqrt(J^2 + h^2 + 2*J*h*cos(k)) for k in k_sorted]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot dispersion
    ax.plot(k_sorted / π, e_k, "b-", linewidth=2, label="ε_k")
    
    # Add vertical line for delta if provided
    if delta !== nothing && delta != 0
        ax.axvline(x=delta/π, color="red", linestyle="--", linewidth=2, label="δ/π")
    end
    
    # Formatting
    ax.set_xlabel("k/π", fontsize=14)
    ax.set_ylabel("ε_k", fontsize=14)
    ax.set_title("Energy Dispersion (N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
    ax.grid(true, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Set x-axis limits
    if bc == :periodic
        ax.set_xlim(0, 2)
    else  # antiperiodic
        ax.set_xlim(0, 2)
    end
    
    plt.tight_layout()
    
    # Save figure
    if save_fig && filename !== nothing
        fig_filename = joinpath("Results", "Figs", "energy_dispersion_$(filename).pdf")
        mkpath(dirname(fig_filename))
        fig.savefig(fig_filename, dpi=300, bbox_inches="tight")
        println("Energy dispersion plot saved to $fig_filename")
    end
    
    return fig
end

# Export the function
export plot_energy_dispersion