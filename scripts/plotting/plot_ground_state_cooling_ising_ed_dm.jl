"""
    plot_ground_state_cooling_ising_ed_dm.jl

Generate an ED (density-matrix) cooling curve for the *integrable* transverse-field
Ising chain and compare:

  (A) single-frequency cooling (Δ = many-body gap, computed automatically)
  (B) multi-frequency cooling (cycle through R detunings Δ_r on a uniform grid)

The plot uses the **relative energy** figure of merit from the paper:

    e = |(E - E_0)/E_0|,

where E_0 is the ground-state energy of H_S.

Usage:
    julia --project=. scripts/plotting/plot_ground_state_cooling_ising_ed_dm.jl

Output:
    scripts/plotting/Figs/ground_state_cooling_ising_ed_dm_single_vs_multiD.pdf
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

using CoolingTNS
using Random
using Printf


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

Random.seed!(1)

backend = EDBackend()

# Integrable transverse-field Ising: H = J Σ Z_i Z_{i+1} + h Σ X_i
ham_params = IsingParameters(4, 1.0, -1.05)
N = ham_params.N

sim_params = UnifiedSimulationParameters(
    DensityMatrix(),
    ContinuousEvolution();
    pe=0.0,
)

coupling = "ZZ"
# Good parameters found by a small ED scan (integrable Ising, N=4):
# larger cycle time strongly improves the steady-state energy.
g = 0.02
te = 24.0
steps = 8000
window = 2000

# Multi-frequency settings
R = 15
Δ_max_factor = 6.0
schedule = :round_robin
randomize_times = false

init_type = "product"
theta = 0.0


# ----------------------------------------------------------------------------
# Single-frequency run (Δ = gap)
# ----------------------------------------------------------------------------

cp_single = BasicCouplingParameters(coupling, g, steps, te, nothing)
prob_single = setup_problem(backend, ham_params, cp_single, sim_params)
Δ_gap = prob_single.extra.coupling_params.delta
Δ_gap === nothing && error("setup_problem did not populate Δ (gap)")

E0 = prob_single.e₀
E0_over_N = E0 / N

@printf("ED DM (Ising): N=%d, steps=%d, te=%.3f, g=%.3f\n", N, steps, te, g)
@printf("  gap Δ=%.6f,  E0/N=%.8f\n", Δ_gap, E0_over_N)

st_single = setup_initial_state(prob_single, sim_params, init_type, theta)
res_single = run_cooling(prob_single, st_single, prob_single.extra.coupling_params, sim_params, ham_params)
E_single = res_single[CoolingTNS.RESULT_ENERGY]
E_single_over_N = E_single ./ N
rel_single = relative_energy.(E_single, Ref(E0))

E_single_ss = mean_last_window(E_single, window)
e_single_ss = relative_energy(E_single_ss, E0)
@printf("  single-Δ steady e(last %d) = %.6f\n", window, e_single_ss)


# ----------------------------------------------------------------------------
# Multi-frequency run
# ----------------------------------------------------------------------------

Δ_values = uniform_delta_grid(Δ_gap, Δ_max_factor * Δ_gap, R)

cp_multi = MultiFrequencyCouplingParameters(
    coupling,
    g,
    steps,
    te,
    Δ_values;
    randomize_times=randomize_times,
    schedule=schedule,
)

prob_multi = setup_problem(backend, ham_params, cp_multi, sim_params)

st_multi = setup_initial_state(prob_multi, sim_params, init_type, theta)
res_multi = run_cooling(prob_multi, st_multi, cp_multi, sim_params, ham_params)
E_multi = res_multi[CoolingTNS.RESULT_ENERGY]
E_multi_over_N = E_multi ./ N
rel_multi = relative_energy.(E_multi, Ref(E0))

E_multi_ss = mean_last_window(E_multi, window)
e_multi_ss = relative_energy(E_multi_ss, E0)
@printf("  multi-Δ  steady e(last %d) = %.6f   (R=%d, Δ∈[%.3f, %.3f])\n",
        window, e_multi_ss, R, minimum(Δ_values), maximum(Δ_values))


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
steps_axis = collect(0:steps)

ax = axs[0]
ax.plot(steps_axis, E_single_over_N, color="C0", label="single-Δ (gap)")
ax.plot(steps_axis, E_multi_over_N, color="C1", label=@sprintf("multi-Δ (R=%d)", R))
ax.axhline(E0_over_N, color="black", linestyle="--", linewidth=1.2, label=L"$E_0/N$")
ax.set_xlabel("cooling step")
ax.set_ylabel(L"energy density $E/N$")
ax.set_title(@sprintf("ED density matrix (%s, t_e=%.1f, g=%.2f)", coupling, te, g))
ax.grid(true, alpha=0.25)
ax.legend(frameon=false, loc="best")

ax = axs[1]
ax.plot(steps_axis, rel_single, color="C0", label="single-Δ")
ax.plot(steps_axis, rel_multi, color="C1", label="multi-Δ")
ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
ax.set_xlabel("cooling step")
ax.set_ylabel(L"relative energy $e$")
ax.set_title(L"$e = |(E-E_0)/E_0|$")
ax.grid(true, alpha=0.25)
ax.legend(frameon=false, loc="best")

fig.tight_layout()
save_figure(fig, @__DIR__, "ground_state_cooling_ising_ed_dm_ZZ_t24_g0.02_R15.pdf")
