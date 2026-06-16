"""
    plot_randomized_time_cooling_curve_tn_dm_trotter.jl

Generate a clearer TN (deterministic) demonstration that randomized cycle times
suppress accidental heating by comparing cooling curves at a representative
mean time t_bad.

Workflow:
1. Compute resonant Δ from the system gap.
2. Coarsely scan a small grid of mean times `t` with a *short* run to find a
   t_bad where fixed-time cooling performs poorly.
3. Run a longer simulation at t_bad for:
   - fixed-time protocol
   - randomized-time protocol (averaged over a few random seeds)
4. Plot energy density E/N vs cooling step (with mean ± std band).

This script uses:
- TNBackend()
- DensityMatrix() + TrotterEvolution() (MPO + Trotter)

Output:
    scripts/plotting/Figs/randomized_times_cooling_curve_tn_dm_trotter.pdf

Note: This is intended as a *local* (moderate-cost) demonstration. For
publication-grade parameter scans over many t-values, run on cluster resources.
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

using CoolingTNS
using Random
using Statistics
using Printf


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

function _silence(f)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            return f()
        end
    end
end

_mean_last(xs::AbstractVector, window::Int) = mean(xs[max(1, length(xs) - window + 1):end])


# -----------------------------------------------------------------------------
# Config (keep moderate so it runs locally)
# -----------------------------------------------------------------------------

backend = TNBackend()

# Small-ish interacting chain for local demo
ham_params = NiIsingParameters(6, 1.0, -1.05, 0.5)
N = ham_params.N

coupling = "XX"
g = 0.3

# Evolution settings
sim_params = UnifiedSimulationParameters(
    DensityMatrix(),
    TrotterEvolution();
    Dmax=20,
    cutoff=1e-6,
    tau=0.2,
    pe=0.0,
)

steps_scan = 10
steps_long = 50
window_size = 5

# Candidate mean times to scan (coarse)
t_values = collect(range(0.5, 4.0; length=8))

# Seeds for randomized-time averaging at the chosen t_bad
seed_list = collect(1:5)

init_type = "product"
theta = 0.0


# -----------------------------------------------------------------------------
# Compute Δ from the gap (same as resonant single-frequency default)
# -----------------------------------------------------------------------------

cp_gap = BasicCouplingParameters(coupling, g, 1, 1.0, nothing)
prob_gap = setup_problem(backend, ham_params, cp_gap, sim_params)
Δ = prob_gap.extra.coupling_params.delta
Δ === nothing && error("Failed to compute gap Δ")
E0 = prob_gap.e₀

@printf("TN DM+Trotter demo: N=%d, g=%.3f, Δ(gap)=%.6f, E0/N=%.6f\n", N, g, Δ, E0 / N)


# -----------------------------------------------------------------------------
# Coarse scan for a bad fixed-time mean time t_bad
# -----------------------------------------------------------------------------

E_fixed_end = Float64[]
for t̄ in t_values
    cp = BasicCouplingParameters(coupling, g, steps_scan, t̄, Δ)
    prob = setup_problem(backend, ham_params, cp, sim_params)
    st = setup_initial_state(prob, sim_params, init_type, theta)

    res = _silence() do
        run_cooling(prob, st, cp, sim_params, ham_params)
    end

    eend = _mean_last(res[CoolingTNS.RESULT_ENERGY], window_size) / N
    push!(E_fixed_end, eend)
    @printf("scan t=%.3f  fixed mean-last(E/N)=%.6f\n", t̄, eend)
end

idx_bad = argmax(E_fixed_end)
t_bad = t_values[idx_bad]
@printf("\nSelected t_bad=%.3f (worst fixed-time among scan grid)\n", t_bad)


# -----------------------------------------------------------------------------
# Long run: fixed time
# -----------------------------------------------------------------------------

cp_fixed = BasicCouplingParameters(coupling, g, steps_long, t_bad, Δ)
prob_fixed = setup_problem(backend, ham_params, cp_fixed, sim_params)
st_fixed = setup_initial_state(prob_fixed, sim_params, init_type, theta)

res_fixed = _silence() do
    run_cooling(prob_fixed, st_fixed, cp_fixed, sim_params, ham_params)
end
E_fixed = res_fixed[CoolingTNS.RESULT_ENERGY] ./ N


# -----------------------------------------------------------------------------
# Long run: randomized times (mean ± std over seeds)
# -----------------------------------------------------------------------------

E_rand = zeros(Float64, steps_long + 1, length(seed_list))
for (j, seed) in enumerate(seed_list)
    Random.seed!(seed)

    cp_rand = MultiFrequencyCouplingParameters(
        coupling,
        g,
        steps_long,
        t_bad,
        [Δ];
        randomize_times=true,
        schedule=:round_robin,
    )

    prob_rand = setup_problem(backend, ham_params, cp_rand, sim_params)
    st_rand = setup_initial_state(prob_rand, sim_params, init_type, theta)

    res_rand = _silence() do
        run_cooling(prob_rand, st_rand, cp_rand, sim_params, ham_params)
    end

    E_rand[:, j] .= res_rand[CoolingTNS.RESULT_ENERGY] ./ N
end

E_rand_mean = vec(mean(E_rand; dims=2))
E_rand_std = vec(std(E_rand; dims=2))

@printf("\nLong-run results at t_bad=%.3f:\n", t_bad)
@printf("  fixed:      E_init/N=%.6f  E_final/N=%.6f\n", E_fixed[1], E_fixed[end])
@printf("  randomized: E_init/N=%.6f  E_final/N=%.6f ± %.6f\n", E_rand_mean[1], E_rand_mean[end], E_rand_std[end])


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

fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.1))

steps_axis = collect(0:steps_long)
ax.plot(steps_axis, E_fixed, color="C0", label=@sprintf("fixed (t=%.2f)", t_bad))
ax.plot(steps_axis, E_rand_mean, color="C1", label=@sprintf("randomized (mean t=%.2f)", t_bad))
ax.fill_between(
    steps_axis,
    E_rand_mean .- E_rand_std,
    E_rand_mean .+ E_rand_std,
    color="C1",
    alpha=0.25,
    linewidth=0,
)

ax.axhline(E0 / N, color="black", linestyle="--", linewidth=1.2, label=L"$E_0/N$")
ax.axhline(E_fixed[1], color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="initial")

ax.set_xlabel("cooling step")
ax.set_ylabel(L"energy density $E/N$")
ax.set_title("TN (MPO+Trotter): randomized times")
ax.grid(true, alpha=0.25)
ax.legend(frameon=false, loc="best")

fig.tight_layout()

save_figure(fig, @__DIR__, "randomized_times_cooling_curve_tn_dm_trotter.pdf")
