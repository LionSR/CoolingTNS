"""
    plot_consistency_tn_vs_ed_dm_trotter.jl

Generate a seminar-style consistency plot comparing:

- EDBackend density-matrix evolution (Trotter)
- TNBackend density-matrix evolution (MPO + Trotter)

Figure: energy density E/N vs cooling step and the absolute discrepancy.

Usage:
  julia --project=. --startup-file=no scripts/plotting/plot_consistency_tn_vs_ed_dm_trotter.jl

Output:
  scripts/plotting/Figs/consistency_tn_vs_ed_dm_trotter.pdf
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

using CoolingTNS
using Random
using Statistics
using Printf

Random.seed!(1)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

ham_params = IsingParameters(4, 1.0, 1.0)  # choose small N where ED is exact
N = ham_params.N

coupling = "XX"
g = 0.2
te = 1.0
steps = 80

τ = 0.05

init_type = "product"
theta = 0.0

# Use the same Δ for both backends (compute from ED once)
cp_ref = BasicCouplingParameters(coupling, g, 1, te, nothing)
prob_ref = setup_problem(EDBackend(), ham_params, cp_ref, UnifiedSimulationParameters(DensityMatrix(), TrotterEvolution(); tau=τ, pe=0.0))
Δ = prob_ref.extra.coupling_params.delta
Δ === nothing && error("Failed to compute Δ")

cp = BasicCouplingParameters(coupling, g, steps, te, Δ)

# ----------------------------------------------------------------------------
# ED run
# ----------------------------------------------------------------------------

sim_ed = UnifiedSimulationParameters(DensityMatrix(), TrotterEvolution(); tau=τ, pe=0.0)
prob_ed = setup_problem(EDBackend(), ham_params, cp, sim_ed)
state0_ed = setup_initial_state(prob_ed, sim_ed, init_type, theta)
res_ed = redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        run_cooling(prob_ed, state0_ed, cp, sim_ed, ham_params)
    end
end
E_ed = Float64.(res_ed[CoolingTNS.RESULT_ENERGY])
E0 = Float64(prob_ed.e₀)

# ----------------------------------------------------------------------------
# TN run
# ----------------------------------------------------------------------------

sim_tn = UnifiedSimulationParameters(
    DensityMatrix(),
    TrotterEvolution();
    tau=τ,
    pe=0.0,
    Dmax=120,
    cutoff=1e-10,
)

prob_tn = setup_problem(TNBackend(), ham_params, cp, sim_tn)
state0_tn = setup_initial_state(prob_tn, sim_tn, init_type, theta)
res_tn = redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        run_cooling(prob_tn, state0_tn, cp, sim_tn, ham_params)
    end
end
E_tn = Float64.(res_tn[CoolingTNS.RESULT_ENERGY])

ΔE = abs.(E_ed .- E_tn)

@printf("TN vs ED (DM+Trotter): N=%d, steps=%d, tau=%.3f\n", N, steps, τ)
@printf("  max |E_ED - E_TN|/N = %.3e\n", maximum(ΔE) / N)

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

plt = get_pyplot()
plt.rcParams.update(Dict(
    "font.size" => 9,
    "axes.labelsize" => 9,
    "axes.titlesize" => 9,
    "legend.fontsize" => 8,
    "xtick.labelsize" => 8,
    "ytick.labelsize" => 8,
    "lines.linewidth" => 1.8,
    "pdf.fonttype" => 42,
    "ps.fonttype" => 42,
))

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
steps_ax = collect(0:steps)

ax = axs[0]
ax.plot(steps_ax, E_ed ./ N, color="C0", label="ED")
ax.plot(steps_ax, E_tn ./ N, color="C1", linestyle="--", label="TN")
ax.axhline(E0 / N, color="black", linestyle="--", linewidth=1.2, label=L"$E_0/N$")
ax.set_xlabel("cooling step")
ax.set_ylabel(L"energy density $E/N$")
ax.set_title(@sprintf("DM+Trotter (τ=%.2f)", τ))
ax.grid(true, alpha=0.25)
ax.legend(frameon=false, loc="best")

ax = axs[1]
ax.semilogy(steps_ax, ΔE ./ N .+ 1e-18, color="C3")
ax.set_xlabel("cooling step")
ax.set_ylabel(L"$|E_{\mathrm{ED}}-E_{\mathrm{TN}}|/N$")
ax.set_title("Discrepancy")
ax.grid(true, alpha=0.25, which="both")

fig.suptitle(@sprintf("Ising (open): N=%d, %s, g=%.2f, t_e=%.1f", N, coupling, g, te), y=1.02)
fig.tight_layout()

save_figure(fig, @__DIR__, "consistency_tn_vs_ed_dm_trotter.pdf")
