# Multi-frequency cooling comparison script
#
# This script runs single-frequency (Δ = gap) cooling and compares it to a
# multi-frequency protocol that cycles Δ through a list of values.
#
# See docs/multi_frequency_cooling_plan.md for the motivation.

using CoolingTNS
using Random
using Printf


"""
    run_multi_frequency_comparison(; kwargs...) -> (results_single, results_multi)

Run single-Δ vs multi-Δ cooling for the interacting niIsing model and print a
summary comparing energies and (optionally) steady-state averages.

Notes:
- Defaults run a fast `EDBackend()` demo (N=4, steps=50). For TN benchmarks, pass
  `backend=TNBackend()` and increase `N`/`steps` (expect longer runtimes).
- For TN runs, the recommended method is `MonteCarloWavefunction + ContinuousEvolution`
  (MPS + TDVP). This is the default for `TNBackend()`.
- Multi-frequency cooling may require longer runs (hundreds of steps) to see a
  clear separation from single-frequency protocols.
"""
function run_multi_frequency_comparison(; 
    backend::CoolingBackend=EDBackend(),
    N::Int=4,
    steps::Int=50,
    coupling::String="XX",
    g::Float64=0.3,
    te::Float64=2.0,
    # Multi-frequency protocol
    R::Int=5,
    delta_strategy::Symbol=:uniform,        # :uniform or :spectral
    delta_max_factor::Float64=6.0,          # only used for :uniform
    randomize_times::Bool=true,
    schedule::Symbol=:round_robin,
    # Simulation parameters
    Dmax::Int=40,
    cutoff::Float64=1e-6,
    tau::Float64=0.1,
    pe::Float64=0.0,
    n_trajectories::Int=1,
    # Initial state
    init_type::String="product",
    theta::Float64=0.0,
    # Analysis
    window_size::Int=10,
)
    ham_params = NiIsingParameters(N, 1.0, -1.05, 0.5)

    sim_params = create_sim_params(
        backend;
        evolution_method=ContinuousEvolution(),
        Dmax=Dmax,
        cutoff=cutoff,
        tau=tau,
        pe=pe,
        n_trajectories=n_trajectories,
    )

    # ------------------------------------------------------------------
    # Baseline: single-frequency cooling (Δ = gap)
    # ------------------------------------------------------------------

    single_params = BasicCouplingParameters(coupling, g, steps, te, nothing)
    problem_single = setup_problem(backend, ham_params, single_params, sim_params)
    state_single = setup_initial_state(problem_single, sim_params, init_type, theta)
    results_single = run_cooling(problem_single, state_single, single_params, sim_params, ham_params)

    gap = problem_single.extra.coupling_params.delta
    gap === nothing && error("Single-frequency setup did not populate delta (gap).")

    # ------------------------------------------------------------------
    # Multi-frequency cooling
    # ------------------------------------------------------------------

    delta_values = if delta_strategy == :uniform
        uniform_delta_grid(gap, delta_max_factor * gap, R)
    elseif delta_strategy == :spectral
        spectral_delta_values(ham_params, backend; R=R)
    else
        error("Unknown delta_strategy=$delta_strategy (use :uniform or :spectral)")
    end

    mf_params = MultiFrequencyCouplingParameters(
        coupling,
        g,
        steps,
        te,
        delta_values;
        randomize_times=randomize_times,
        schedule=schedule,
    )

    problem_mf = setup_problem(backend, ham_params, mf_params, sim_params)
    state_mf = setup_initial_state(problem_mf, sim_params, init_type, theta)
    results_mf = run_cooling(problem_mf, state_mf, mf_params, sim_params, ham_params)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    E_single = results_single[RESULT_ENERGY]
    E_multi = results_mf[RESULT_ENERGY]

    E0 = problem_single.e₀

    E_single_ss = mean_last_window(E_single, window_size)
    E_multi_ss = mean_last_window(E_multi, window_size)

    @printf("\n--- Multi-frequency comparison ---\n")
    @printf("backend        = %s\n", string(typeof(backend)))
    @printf("N              = %d\n", N)
    @printf("steps          = %d\n", steps)
    @printf("coupling       = %s\n", coupling)
    @printf("g              = %.3f\n", g)
    @printf("te (mean)      = %.3f\n", te)
    @printf("gap Δ          = %.6f\n", gap)
    @printf("\n")

    @printf("E0/N                         = %.8f\n", E0 / N)
    @printf("single-Δ final E/N           = %.8f\n", E_single[end] / N)
    @printf("single-Δ steady(last %d)/N   = %.8f\n", window_size, E_single_ss / N)
    @printf("\n")

    @printf("multi-Δ R                    = %d\n", R)
    @printf("multi-Δ strategy             = %s\n", string(delta_strategy))
    @printf("multi-Δ schedule             = %s\n", string(schedule))
    @printf("multi-Δ randomize_times      = %s\n", string(randomize_times))
    @printf("multi-Δ Δ-range              = [%.6f, %.6f]\n", minimum(delta_values), maximum(delta_values))
    @printf("multi-Δ final E/N            = %.8f\n", E_multi[end] / N)
    @printf("multi-Δ steady(last %d)/N    = %.8f\n", window_size, E_multi_ss / N)

    return results_single, results_mf
end


if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(1)
    run_multi_frequency_comparison()
end
