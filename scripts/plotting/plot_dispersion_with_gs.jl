"""
Plot energy dispersion ε_k and ground state occupation n_k^(GS).

Standalone plotting script. Usage:
    julia --project=. scripts/plotting/plot_dispersion_with_gs.jl
"""

# Allow tests and notebooks to include several standalone plot scripts in one session.
if !isdefined(@__MODULE__, :get_pyplot)
    Base.include(@__MODULE__, joinpath(@__DIR__, "PlotUtils.jl"))
end

if !isdefined(@__MODULE__, :_COOLINGTNS_PLOT_DISPERSION_WITH_GS_INCLUDED)
const _COOLINGTNS_PLOT_DISPERSION_WITH_GS_INCLUDED = true

"""
    plot_dispersion_with_ground_state(N, J, h, bc; delta=nothing, save_fig=true, filename=nothing)

Plot the energy dispersion epsilon_k vs k and ground state occupation n_k^(GS) for the transverse field Ising model.
Here `bc` denotes the fermionic boundary condition used for the plotted
momentum grid.
"""
function plot_dispersion_with_ground_state(N::Int, J::Real, h::Real, bc::Symbol;
                                         delta=nothing, save_fig=true, filename=nothing)
    plt = get_pyplot()

    k_values = generate_k_values(N, bc)
    k_sorted = sort(k_values)
    e_k = compute_energy_dispersion(k_sorted, J, h)
    n_k_gs = compute_ground_state_occupation(k_sorted, J, h)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(k_sorted / pi, e_k, "b-", linewidth=2, label="epsilon_k")
    ax1.set_xlabel("k/pi", fontsize=14)
    ax1.set_ylabel("epsilon_k", fontsize=14, color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2.plot(k_sorted / pi, n_k_gs, "g--", linewidth=2, label="n_k^(GS)")
    ax2.set_ylabel("n_k^(GS)", fontsize=14, color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax2.set_ylim(-0.1, 1.1)

    add_detuning_energy_marker!(ax1, delta; alpha=0.7)

    ax1.set_title("Energy Dispersion and Ground State Occupation\n(N=$N, J=$J, h=$h, BC=$bc)", fontsize=16)
    ax1.grid(true, alpha=0.3)
    ax1.set_xlim(minimum(k_sorted) / pi, maximum(k_sorted) / pi)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=12)

    plt.tight_layout()

    if save_fig && filename !== nothing
        save_figure(fig, "Results", "dispersion_with_gs_$(filename).pdf")
    end

    return fig
end

end
