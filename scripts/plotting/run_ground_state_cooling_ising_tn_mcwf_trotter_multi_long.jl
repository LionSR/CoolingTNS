"""
    run_ground_state_cooling_ising_tn_mcwf_trotter_multi_long.jl

Compute-only script (no Python plotting) for a long multi-frequency TN cooling run
of the *integrable* transverse-field Ising model.

This writes an HDF5 cache file under `scripts/plotting/Data/` so plotting can be
run separately without re-running the expensive TN simulation.

Usage:
  julia --project=. scripts/plotting/run_ground_state_cooling_ising_tn_mcwf_trotter_multi_long.jl

Output:
  scripts/plotting/Data/ground_state_cooling_ising_tn_mcwf_trotter_multi_ZZ_N20_g0.2_te3.2_steps120_R15_seed1.h5
"""

using CoolingTNS
using Random
using Statistics
using Printf
using HDF5

Random.seed!(1)

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
te = 3.2
steps = 120
window = 30

R = 15
Δ_max_factor = 6.0
schedule = :round_robin
randomize_times = false

init_type = "product"
theta = 0.0

# Reference setup to compute Δ_gap and reuse DMRG objects
cp_ref = BasicCouplingParameters(coupling, g, 1, te, nothing)
prob_ref = setup_problem(backend, ham_params, cp_ref, sim_params)
Δ_gap = prob_ref.extra.coupling_params.delta
Δ_gap === nothing && error("setup_problem did not populate Δ (gap)")

E0 = Float64(prob_ref.e₀)
E0_over_N = E0 / N

@printf("TN MCWF+Trotter (Ising, multi-Δ long run): N=%d, steps=%d, te=%.3f, g=%.3f\n", N, steps, te, g)
@printf("  gap Δ=%.6f,  E0/N=%.8f\n", Δ_gap, E0_over_N)

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

extra_multi = (
    coupling_params=cp_multi,
    coupling=coupling,
    g=g,
    sites=prob_ref.extra.sites,
    gap=Δ_gap,
    ham_params=ham_params,
    gates_cache=Dict{Float64, Any}(),
)
prob_multi = CoolingProblem(backend, prob_ref.H_sys, nothing, prob_ref.ϕ₀, prob_ref.e₀, extra_multi)

st_multi = setup_initial_state(prob_multi, sim_params, init_type, theta)
res_multi = run_cooling(prob_multi, st_multi, cp_multi, sim_params, ham_params)

E_list = Float64.(res_multi[RESULT_ENERGY])
rel_list = relative_energy.(E_list, Ref(E0))

E_ss = mean_last_window(E_list, window)
e_ss = relative_energy(E_ss, E0)

E_min = minimum(E_list)
e_min = relative_energy(E_min, E0)

@printf("  multi-Δ steady e(last %d) = %.6f   (R=%d, Δ∈[%.3f, %.3f])\n",
        window, e_ss, R, minimum(Δ_values), maximum(Δ_values))
@printf("  multi-Δ E_end/N = %.8f  (e_end=%.6f)\n", E_list[end] / N, rel_list[end])
@printf("  multi-Δ E_min/N = %.8f  (e_min=%.6f)\n", E_min / N, e_min)

outdir = joinpath(@__DIR__, "Data")
mkpath(outdir)

outfile = joinpath(
    outdir,
    "ground_state_cooling_ising_tn_mcwf_trotter_multi_ZZ_N20_g0.2_te3.2_steps120_R15_seed1.h5",
)

h5open(outfile, "w") do f
    # primary curves
    write(f, RESULT_ENERGY, E_list)
    write(f, "rel_list", rel_list)

    # metadata
    write(f, "E0", E0)
    write(f, "N", N)
    write(f, "coupling", coupling)
    write(f, "g", g)
    write(f, "te", te)
    write(f, "steps", steps)
    write(f, "window", window)
    write(f, "Dmax", sim_params.Dmax)
    write(f, "cutoff", sim_params.cutoff)
    write(f, "tau", sim_params.tau)
    write(f, "R", R)
    write(f, "Δ_gap", Float64(Δ_gap))
    write(f, RESULT_DELTA_VALUES, Float64.(Δ_values))
    write(f, "schedule", String(schedule))
    write(f, "randomize_times", randomize_times)

    if haskey(res_multi, RESULT_DELTA_LIST)
        write(f, RESULT_DELTA_LIST, Float64.(res_multi[RESULT_DELTA_LIST]))
    end
    if haskey(res_multi, RESULT_TE_LIST)
        write(f, RESULT_TE_LIST, Float64.(res_multi[RESULT_TE_LIST]))
    end
end

println("Wrote data to $outfile")
