"""
Plot energy dispersion ε_k vs k for the transverse field Ising model.

Standalone plotting script. Usage:
    julia --project=. scripts/plotting/plot_energy_dispersion.jl
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

"""
    plot_energy_dispersion(N, J, h, bc; delta=nothing, save_fig=true, filename=nothing)

Plot the energy dispersion epsilon_k vs k for the transverse field Ising model.
For periodic BC: k = 2*pi*n/N where n = 0, 1, ..., N-1
For antiperiodic BC: k = pi*(2n+1)/N where n = 0, 1, ..., N-1
"""
function plot_energy_dispersion(N::Int, J::Real, h::Real, bc::Symbol;
                               delta=nothing, save_fig=true, filename=nothing)
    plt = get_pyplot()

    k_values = generate_k_values(N, bc)
    k_sorted = sort(k_values)
    e_k = compute_energy_dispersion(k_sorted, J, h; N=N)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(k_sorted / pi, e_k, "b-", linewidth=2, label="epsilon_k")

    mark_bath_detuning_energy!(ax, delta; reference_energies=e_k, linewidth=2)

    ax.set_xlabel("k/pi", fontsize=14)
    ax.set_ylabel("epsilon_k", fontsize=14)
    ax.set_title("Energy Dispersion (N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
    ax.grid(true, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 2)
    plt.tight_layout()

    if save_fig && filename !== nothing
        save_figure(fig, "Results", "energy_dispersion_$(filename).pdf")
    end

    return fig
end
