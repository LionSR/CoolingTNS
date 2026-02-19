"""
    scan_randomized_times_resonances_tn_ising_mcwf_trotter_N20.jl

Compute-only quick scan (no plotting) over mean cycle times `t̄` to locate a
representative accidental-resonance point for TN simulations of the integrable
Ising model.

It compares fixed-time vs randomized-time protocols using the paper metric

  e = |(E - E0)/E0|,

computed as a mean over the last `window` cooling steps.

Usage:
  julia --project=. scripts/plotting/scan_randomized_times_resonances_tn_ising_mcwf_trotter_N20.jl
"""

using CoolingTNS
using Random
using Statistics
using Printf

"""Run `f()` with stdout/stderr redirected to /dev/null."""
function _silence(f)
    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            return f()
        end
    end
end

_mean_last(xs::AbstractVector, window::Int) = mean(xs[max(1, length(xs) - window + 1):end])

backend = TNBackend()
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
g = 0.2

# Scan settings
steps = 20
window = 5
seed = 1

# Candidate mean times to probe (keep small to limit runtime)
t_values = [2.8, 3.2, 3.6, 4.0]

# Multi-frequency settings
R = 15
Δ_max_factor = 6.0
schedule = :round_robin

init_type = "product"
theta = 0.0

# Reference setup (computes Δ_gap, E0, and provides sites/H_sys/ϕ0)
cp_ref = BasicCouplingParameters(coupling, g, 1, t_values[1], nothing)
prob_ref = setup_problem(backend, ham_params, cp_ref, sim_params)
Δ_gap = prob_ref.extra.coupling_params.delta
Δ_gap === nothing && error("setup_problem did not populate Δ (gap)")
E0 = Float64(prob_ref.e₀)

Δ_values = uniform_delta_grid(Δ_gap, Δ_max_factor * Δ_gap, R)

@printf("TN scan (Ising): N=%d, g=%.3f, steps=%d, window=%d\n", N, g, steps, window)
@printf("  gap Δ=%.6f, E0/N=%.8f, R=%d\n", Δ_gap, E0 / N, R)
@printf("  t_values=%s\n\n", string(t_values))

shared_gates_cache = Dict{Float64, Any}()

function _run_once(; t_mean::Float64, randomize_times::Bool)
    Random.seed!(seed)

    cp = MultiFrequencyCouplingParameters(
        coupling,
        g,
        steps,
        t_mean,
        Δ_values;
        randomize_times=randomize_times,
        schedule=schedule,
    )

    extra = (
        coupling_params=cp,
        coupling=coupling,
        g=g,
        sites=prob_ref.extra.sites,
        gap=Δ_gap,
        ham_params=ham_params,
        gates_cache=shared_gates_cache,
    )
    prob = CoolingProblem(backend, prob_ref.H_sys, nothing, prob_ref.ϕ₀, prob_ref.e₀, extra)

    st = setup_initial_state(prob, sim_params, init_type, theta)
    res = _silence(() -> run_cooling(prob, st, cp, sim_params, ham_params))

    E_ss = _mean_last(res["E_list"], window)
    return relative_energy(Float64(E_ss), E0)
end

fixed_e = Float64[]
rand_e = Float64[]

for t̄ in t_values
    ef = _run_once(t_mean=t̄, randomize_times=false)
    er = _run_once(t_mean=t̄, randomize_times=true)
    push!(fixed_e, ef)
    push!(rand_e, er)
    @printf("t=%.3f  fixed(e)=%.6f  randomized(e)=%.6f  Δe=%.6f\n", t̄, ef, er, ef - er)
end

improvement = fixed_e .- rand_e
idx = argmax(improvement)
@printf("\nBest candidate (max Δe): t_bad=%.3f  Δe=%.6f\n", t_values[idx], improvement[idx])
