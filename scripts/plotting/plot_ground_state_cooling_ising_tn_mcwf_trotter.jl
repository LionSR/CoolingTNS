"""
    plot_ground_state_cooling_ising_tn_mcwf_trotter.jl

Ground-state cooling demo for the *integrable* transverse-field Ising chain using
Tensor Networks (MPS trajectories) and the paper metric

    e = |(E - E0) / E0|.

We compare:
  (A) single-frequency cooling (Δ = many-body gap)
  (B) multi-frequency cooling (cycle detunings Δ_r on a uniform grid)

Backend/method:
  TNBackend() + MonteCarloWavefunction() + TrotterEvolution()  (MPS + Trotter)

This script is configured for N=20 and parameters that reliably cool into the
negative-energy regime within a few minutes on a laptop.

Usage:
  julia --project=. scripts/plotting/plot_ground_state_cooling_ising_tn_mcwf_trotter.jl

Output:
  scripts/plotting/Figs/ground_state_cooling_ising_tn_mcwf_trotter_ZZ_N20.pdf
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

# Integrable transverse-field Ising: H = J Σ Z_i Z_{i+1} + h_x Σ X_i
ham_params = IsingParameters(20, 1.0, -1.05)
N = ham_params.N

sim_params = UnifiedSimulationParameters(
    MonteCarloWavefunction(),
    TrotterEvolution();
    Dmax=30,
    cutoff=1e-6,
    tau=0.2,
    pe=0.0,
)

coupling = "ZZ"

# Empirically good parameters (TN, N=20)
# NOTE: Smaller g reduces reheating and reaches closer to E0.
g = 0.2
te = 3.2
steps = 80
window = 20

# Multi-frequency settings
R = 15
Δ_max_factor = 6.0
schedule = :round_robin
randomize_times = false

init_type = "product"
theta = 0.0


# -----------------------------------------------------------------------------
# Single-frequency run (Δ = gap)
# -----------------------------------------------------------------------------

cp_single = BasicCouplingParameters(coupling, g, steps, te, nothing)
prob_single = setup_problem(backend, ham_params, cp_single, sim_params)
Δ_gap = prob_single.extra.coupling_params.delta
Δ_gap === nothing && error("setup_problem did not populate Δ (gap)")

E0 = prob_single.e₀
E0_over_N = E0 / N

@printf("TN MCWF+Trotter (Ising): N=%d, steps=%d, te=%.3f, g=%.3f\n", N, steps, te, g)
@printf("  gap Δ=%.6f,  E0/N=%.8f\n", Δ_gap, E0_over_N)

st_single = setup_initial_state(prob_single, sim_params, init_type, theta)
res_single = run_cooling(prob_single, st_single, prob_single.extra.coupling_params, sim_params, ham_params)
E_single = res_single[CoolingTNS.RESULT_ENERGY]
E_single_over_N = E_single ./ N
rel_single = relative_energy.(E_single, Ref(E0))

E_single_ss = mean_last_window(E_single, window)
e_single_ss = relative_energy(E_single_ss, E0)
@printf("  single-Δ steady e(last %d) = %.6f\n", window, e_single_ss)


# -----------------------------------------------------------------------------
# Multi-frequency run
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

# Reuse H_sys / ϕ0 / e0 / sites from prob_single to avoid re-running DMRG
extra_multi = (
    coupling_params=cp_multi,
    coupling=coupling,
    g=g,
    sites=prob_single.extra.sites,
    gap=Δ_gap,
    ham_params=ham_params,
    gates_cache=Dict{Float64, Any}(),
)
prob_multi = CoolingProblem(backend, prob_single.H_sys, nothing, prob_single.ϕ₀, prob_single.e₀, extra_multi)

st_multi = setup_initial_state(prob_multi, sim_params, init_type, theta)
res_multi = run_cooling(prob_multi, st_multi, cp_multi, sim_params, ham_params)
E_multi = res_multi[CoolingTNS.RESULT_ENERGY]
E_multi_over_N = E_multi ./ N
rel_multi = relative_energy.(E_multi, Ref(E0))

E_multi_ss = mean_last_window(E_multi, window)
e_multi_ss = relative_energy(E_multi_ss, E0)
@printf("  multi-Δ  steady e(last %d) = %.6f   (R=%d, Δ∈[%.3f, %.3f])\n",
        window, e_multi_ss, R, minimum(Δ_values), maximum(Δ_values))


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

fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
steps_axis = collect(0:steps)

ax = axs[0]
ax.plot(steps_axis, E_single_over_N, color="C0", label="single-Δ (gap)")
ax.plot(steps_axis, E_multi_over_N, color="C1", label=@sprintf("multi-Δ (R=%d)", R))
ax.axhline(E0_over_N, color="black", linestyle="--", linewidth=1.2, label=L"$E_0/N$")
ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
ax.set_xlabel("cooling step")
ax.set_ylabel(L"energy density $E/N$")
ax.set_title(@sprintf("TN MCWF+Trotter (%s, t_e=%.1f, g=%.1f)", coupling, te, g))
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
save_figure(fig, @__DIR__, "ground_state_cooling_ising_tn_mcwf_trotter_ZZ_N20_g0.2_te3.2_steps80_R15.pdf")
