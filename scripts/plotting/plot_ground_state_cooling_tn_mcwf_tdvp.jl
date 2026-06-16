"""
    plot_ground_state_cooling_tn_mcwf_tdvp.jl

Generate a TN (MPS) cooling curve that reaches deep into the negative-energy
regime using the recommended method:

  TNBackend() + MonteCarloWavefunction() + ContinuousEvolution()  (MPS + TDVP)

The script plots energy density E/N vs cooling step and compares:
  1) single-frequency cooling (Δ = many-body gap, computed automatically)
  2) multi-frequency cooling (cycle through R detunings Δ_r on a uniform grid)

This is a *local demo* configured for small N. For production runs, increase N,
steps, bond dimension, and consider averaging over multiple trajectories.

Usage:
    julia --project=. scripts/plotting/plot_ground_state_cooling_tn_mcwf_tdvp.jl

Output:
    scripts/plotting/Figs/ground_state_cooling_tn_mcwf_tdvp_single_vs_multiD.pdf
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

using CoolingTNS
using Random
using Printf


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

Random.seed!(1)

backend = TNBackend()

# Interacting model used in the TN draft
ham_params = NiIsingParameters(6, 1.0, -1.05, 0.5)
N = ham_params.N

# Recommended TN method
sim_params = UnifiedSimulationParameters(
    MonteCarloWavefunction(),
    ContinuousEvolution();
    Dmax=40,
    cutoff=1e-6,
    tau=0.2,
    pe=0.0,
)

coupling = "XX"
g = 0.3
te = 2.0
steps = 80  # keep moderate so the script runs locally; increase for production

# Multi-frequency settings
R = 5
Δ_max_factor = 6.0
schedule = :round_robin
randomize_times = false

# Initial state
init_type = "product"
theta = 0.0


# -----------------------------------------------------------------------------
# Single-frequency run (Δ = gap)
# -----------------------------------------------------------------------------

cp_single = BasicCouplingParameters(coupling, g, steps, te, nothing)
prob_single = setup_problem(backend, ham_params, cp_single, sim_params)
Δ_gap = prob_single.extra.coupling_params.delta
Δ_gap === nothing && error("setup_problem did not populate Δ (gap)")

E0_over_N = prob_single.e₀ / N

@printf("TN MCWF+TDVP: N=%d, steps=%d, te=%.3f, g=%.3f\n", N, steps, te, g)
@printf("  gap Δ=%.6f,  E0/N=%.6f\n", Δ_gap, E0_over_N)

st_single = setup_initial_state(prob_single, sim_params, init_type, theta)
res_single = run_cooling(prob_single, st_single, prob_single.extra.coupling_params, sim_params, ham_params)
E_single = res_single[CoolingTNS.RESULT_ENERGY] ./ N


# -----------------------------------------------------------------------------
# Multi-frequency run (reuse H_sys / ϕ0 / e0 from single-frequency setup)
# -----------------------------------------------------------------------------

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

# Avoid recomputing DMRG: reuse from prob_single
extra_multi = (
    coupling_params=cp_multi,
    coupling=coupling,
    g=g,
    sites=prob_single.extra.sites,
    gap=Δ_gap,
    ham_params=ham_params,
    H_cache=Dict{Float64, Any}(),
)
prob_multi = CoolingProblem(backend, prob_single.H_sys, nothing, prob_single.ϕ₀, prob_single.e₀, extra_multi)

st_multi = setup_initial_state(prob_multi, sim_params, init_type, theta)
res_multi = run_cooling(prob_multi, st_multi, cp_multi, sim_params, ham_params)
E_multi = res_multi[CoolingTNS.RESULT_ENERGY] ./ N

@printf("  single-Δ final E/N = %.6f\n", E_single[end])
@printf("  multi-Δ  final E/N = %.6f   (R=%d, Δ∈[%.3f, %.3f])\n", E_multi[end], R, minimum(Δ_values), maximum(Δ_values))


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

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

fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2))
steps_axis = collect(0:steps)
ax.plot(steps_axis, E_single, color="C0", label="single-Δ (gap)")
ax.plot(steps_axis, E_multi, color="C1", label=@sprintf("multi-Δ (R=%d)", R))
ax.axhline(E0_over_N, color="black", linestyle="--", linewidth=1.2, label=L"$E_0/N$")
ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.8, label=L"$E/N=0$")
ax.set_xlabel("cooling step")
ax.set_ylabel(L"energy density $E/N$")
ax.set_title(L"TN MCWF+TDVP ($t_e=2.0$, $g=0.3$)")
ax.grid(true, alpha=0.25)
ax.legend(frameon=false, loc="best")
fig.tight_layout()

save_figure(fig, @__DIR__, "ground_state_cooling_tn_mcwf_tdvp_single_vs_multiD.pdf")
