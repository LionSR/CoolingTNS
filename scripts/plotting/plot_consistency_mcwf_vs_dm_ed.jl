"""
    plot_consistency_mcwf_vs_dm_ed.jl

Generate a seminar-style consistency plot comparing:

- ED density matrix evolution (deterministic)
- ED MCWF ensemble average (stochastic unraveling)

Figure: energy density E/N vs cooling step.

Usage:
  julia --project=. --startup-file=no scripts/plotting/plot_consistency_mcwf_vs_dm_ed.jl

Output:
  scripts/plotting/Figs/consistency_mcwf_vs_dm_ed.pdf
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

backend = EDBackend()

ham_params = IsingParameters(4, 1.0, 1.0)  # small N for fast ED
N = ham_params.N

coupling = "XX"
g = 0.2
te = 1.0
steps = 120

init_type = "product"
theta = 0.0

n_traj = 60
seed_list = collect(1:n_traj)

# ----------------------------------------------------------------------------
# Density matrix reference
# ----------------------------------------------------------------------------

sim_dm = UnifiedSimulationParameters(DensityMatrix(), ContinuousEvolution(); pe=0.0)
cp = BasicCouplingParameters(coupling, g, steps, te, nothing)

prob_dm = setup_problem(backend, ham_params, cp, sim_dm)
state0_dm = setup_initial_state(prob_dm, sim_dm, init_type, theta)

res_dm = redirect_stdout(devnull) do
    redirect_stderr(devnull) do
        run_cooling(prob_dm, state0_dm, cp, sim_dm, ham_params)
    end
end

E_dm = Float64.(res_dm[CoolingTNS.RESULT_ENERGY])
E0 = Float64(prob_dm.e₀)

# ----------------------------------------------------------------------------
# MCWF ensemble
# ----------------------------------------------------------------------------

sim_mc = UnifiedSimulationParameters(MonteCarloWavefunction(), ContinuousEvolution(); pe=0.0)
E_mc_mat = zeros(Float64, steps + 1, n_traj)

for (j, seed) in enumerate(seed_list)
    Random.seed!(seed)
    prob_mc = setup_problem(backend, ham_params, cp, sim_mc)
    state0_mc = setup_initial_state(prob_mc, sim_mc, init_type, theta)

    res_mc = redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            run_cooling(prob_mc, state0_mc, cp, sim_mc, ham_params)
        end
    end

    E_mc_mat[:, j] .= Float64.(res_mc[CoolingTNS.RESULT_ENERGY])
end

E_mc_mean = vec(mean(E_mc_mat; dims=2))
E_mc_stderr = vec(std(E_mc_mat; dims=2)) ./ sqrt(n_traj)

@printf("MCWF vs DM (ED): N=%d, steps=%d, n_traj=%d\n", N, steps, n_traj)
@printf("  max |E_mc_mean - E_dm|/N = %.3e\n", maximum(abs.(E_mc_mean .- E_dm)) / N)

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

fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.1))
steps_ax = collect(0:steps)

ax.plot(steps_ax, E_dm ./ N, color="C0", label="DM")
ax.plot(steps_ax, E_mc_mean ./ N, color="C1", label="MCWF mean")
ax.fill_between(steps_ax,
                (E_mc_mean .- E_mc_stderr) ./ N,
                (E_mc_mean .+ E_mc_stderr) ./ N,
                color="C1", alpha=0.25, linewidth=0)
ax.axhline(E0 / N, color="black", linestyle="--", linewidth=1.2, label=L"$E_0/N$")

ax.set_xlabel("cooling step")
ax.set_ylabel(L"energy density $E/N$")
ax.set_title(@sprintf("ED: MCWF ensemble vs density matrix (%s, g=%.2f, t_e=%.1f)", coupling, g, te))
ax.grid(true, alpha=0.25)
ax.legend(frameon=false, loc="best")

fig.tight_layout()

save_figure(fig, @__DIR__, "consistency_mcwf_vs_dm_ed.pdf")
