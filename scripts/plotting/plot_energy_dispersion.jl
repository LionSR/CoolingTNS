"""
Plot energy dispersion ε_k vs k for the transverse field Ising model.

Standalone plotting script. Usage:
    julia --project=. scripts/plotting/plot_energy_dispersion.jl
"""

# Allow tests and notebooks to include several standalone plot scripts in one session.
if !@isdefined(get_pyplot)
    include(joinpath(@__DIR__, "PlotUtils.jl"))
end

"""
    plot_energy_dispersion(N, J, h, bc; delta=nothing, save_fig=true, filename=nothing)

Plot the energy dispersion epsilon_k vs k for the transverse field Ising model.
Here `bc` denotes the fermionic boundary condition used for the plotted
momentum grid.
"""
function plot_energy_dispersion(N::Int, J::Real, h::Real, bc::Symbol;
                               delta=nothing, save_fig=true, filename=nothing)
    plt = get_pyplot()

    k_values = generate_k_values(N, bc)
    k_sorted = sort(k_values)
    e_k = compute_energy_dispersion(k_sorted, J, h)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(k_sorted / pi, e_k, "b-", linewidth=2, label="epsilon_k")

    add_detuning_energy_marker!(ax, delta; alpha=1.0)

    ax.set_xlabel("k/pi", fontsize=14)
    ax.set_ylabel("epsilon_k", fontsize=14)
    ax.set_title("Energy Dispersion (N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
    ax.grid(true, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(minimum(k_sorted) / pi, maximum(k_sorted) / pi)
    plt.tight_layout()

    if save_fig && filename !== nothing
        save_figure(fig, "Results", "energy_dispersion_$(filename).pdf")
    end

    return fig
end
