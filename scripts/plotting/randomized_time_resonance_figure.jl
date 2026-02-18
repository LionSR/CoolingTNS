"""
    randomized_time_resonance_figure.jl

Shared utilities to generate a publication-style figure demonstrating how
randomizing cycle times suppresses accidental resonances/heating.

This is inspired by the randomized-time analysis in arXiv:2503.24330.

The figure typically has two panels:

1. A scan of the (steady-state) energy density versus the *mean* cycle time `t`.
2. A representative time point `t_bad` where fixed-time cooling performs worse,
   showing the energy density trajectory versus cooling step.

The same code can be used with:
- EDBackend (fast parameter scans)
- TNBackend + DensityMatrix + TrotterEvolution (deterministic MPO evolution)

The randomized-time protocol is implemented as `t_m ~ Uniform(0, 2t)` per step.
"""

include(joinpath(@__DIR__, "PlotUtils.jl"))

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

"""Return mean over the last `window` entries of `xs` (expects `window ≥ 1`)."""
_mean_last(xs::AbstractVector, window::Int) = mean(xs[max(1, length(xs) - window + 1):end])

function _maybe_clear_ed_cache!(backend)
    backend isa EDBackend && empty!(CoolingTNS.EVOLUTION_OP_CACHE)
    return nothing
end

function _compute_gap_and_e0(backend::CoolingBackend, ham_params::HamiltonianParameters, sim_params;
                             coupling::String, g::Float64)
    cp_gap = BasicCouplingParameters(coupling, g, 1, 1.0, nothing)
    prob_gap = setup_problem(backend, ham_params, cp_gap, sim_params)
    Δ = prob_gap.extra.coupling_params.delta
    Δ === nothing && error("Failed to compute gap Δ")
    return Float64(Δ), Float64(prob_gap.e₀)
end

function _run_fixed_time(
    backend::CoolingBackend,
    ham_params::HamiltonianParameters,
    sim_params::UnifiedSimulationParameters,
    Δ::Float64;
    coupling::String,
    g::Float64,
    steps::Int,
    t_mean::Float64,
    init_type::String,
    theta::Float64,
    silence::Bool=true,
)
    cp_fixed = BasicCouplingParameters(coupling, g, steps, t_mean, Δ)
    prob_fixed = setup_problem(backend, ham_params, cp_fixed, sim_params)
    st_fixed = setup_initial_state(prob_fixed, sim_params, init_type, theta)

    runner = () -> run_cooling(prob_fixed, st_fixed, cp_fixed, sim_params, ham_params)
    res = silence ? _silence(runner) : runner()
    return res
end

function _run_randomized_time_once(
    backend::CoolingBackend,
    ham_params::HamiltonianParameters,
    sim_params::UnifiedSimulationParameters,
    Δ::Float64;
    coupling::String,
    g::Float64,
    steps::Int,
    t_mean::Float64,
    init_type::String,
    theta::Float64,
    seed::Int,
    silence::Bool=true,
)
    Random.seed!(seed)
    _maybe_clear_ed_cache!(backend)

    cp_rand = MultiFrequencyCouplingParameters(
        coupling,
        g,
        steps,
        t_mean,
        [Δ];
        randomize_times=true,
        schedule=:round_robin,
    )

    prob_rand = setup_problem(backend, ham_params, cp_rand, sim_params)
    st_rand = setup_initial_state(prob_rand, sim_params, init_type, theta)

    runner = () -> run_cooling(prob_rand, st_rand, cp_rand, sim_params, ham_params)
    res = silence ? _silence(runner) : runner()
    return res
end


"""
    plot_time_randomization_resonances(; kwargs...)

Compute and plot fixed-time vs randomized-time cooling performance as a function
of the mean cycle time.

Returns `(fig, data)` where `data` is a named tuple containing the scan arrays.

Key kwargs:
- `backend`: `EDBackend()` or `TNBackend()`.
- `sim_params`: for TN deterministic runs, pass `DensityMatrix(), TrotterEvolution()`.
- `t_values`: vector of mean times to scan.
- `seed_list`: seeds used for randomized-time averaging.
"""
function plot_time_randomization_resonances(; 
    backend::CoolingBackend=EDBackend(),
    ham_params::HamiltonianParameters=NiIsingParameters(4, 1.0, -1.05, 0.5),
    sim_params::UnifiedSimulationParameters=UnifiedSimulationParameters(DensityMatrix(), ContinuousEvolution(); pe=0.0),
    coupling::String="XX",
    g::Float64=0.3,
    steps::Int=50,
    window_size::Int=10,
    t_values::Vector{Float64}=collect(range(0.2, 10.0; length=21)),
    seed_list::Vector{Int}=collect(1:10),
    init_type::String="product",
    theta::Float64=0.0,
    output_name::String="randomized_times_remove_accidental_heating.pdf",
    silence::Bool=true,
)
    N = ham_params.N

    Δ, E0 = _compute_gap_and_e0(backend, ham_params, sim_params; coupling=coupling, g=g)
    @printf("Using backend=%s, N=%d, g=%.3f, Δ(gap)=%.6f\n", string(typeof(backend)), N, g, Δ)

    nt = length(t_values)
    E_fixed = fill(NaN, nt)
    E_rand_mean = fill(NaN, nt)
    E_rand_std = fill(NaN, nt)

    for (i, t̄) in enumerate(t_values)
        res_fixed = _run_fixed_time(
            backend,
            ham_params,
            sim_params,
            Δ;
            coupling=coupling,
            g=g,
            steps=steps,
            t_mean=t̄,
            init_type=init_type,
            theta=theta,
            silence=silence,
        )
        E_fixed[i] = _mean_last(res_fixed["E_list"], window_size) / N

        Es = Float64[]
        for seed in seed_list
            res_rand = _run_randomized_time_once(
                backend,
                ham_params,
                sim_params,
                Δ;
                coupling=coupling,
                g=g,
                steps=steps,
                t_mean=t̄,
                init_type=init_type,
                theta=theta,
                seed=seed,
                silence=silence,
            )
            push!(Es, _mean_last(res_rand["E_list"], window_size) / N)
        end

        E_rand_mean[i] = mean(Es)
        E_rand_std[i] = std(Es)

        @printf("t=%.3f  fixed(E/N)=%.6f  randomized(E/N)=%.6f ± %.6f\n", t̄, E_fixed[i], E_rand_mean[i], E_rand_std[i])
    end

    improvement = E_fixed .- E_rand_mean
    idx_bad = argmax(improvement)
    t_bad = t_values[idx_bad]
    @printf("\nRepresentative accidental-resonance point: t_bad=%.3f (ΔE/N=%.6f)\n", t_bad, improvement[idx_bad])

    # Time series at t_bad
    res_fixed_bad = _run_fixed_time(
        backend,
        ham_params,
        sim_params,
        Δ;
        coupling=coupling,
        g=g,
        steps=steps,
        t_mean=t_bad,
        init_type=init_type,
        theta=theta,
        silence=silence,
    )
    E_series_fixed = res_fixed_bad["E_list"] ./ N

    E_series_rand = zeros(Float64, steps + 1, length(seed_list))
    for (j, seed) in enumerate(seed_list)
        res_rand_bad = _run_randomized_time_once(
            backend,
            ham_params,
            sim_params,
            Δ;
            coupling=coupling,
            g=g,
            steps=steps,
            t_mean=t_bad,
            init_type=init_type,
            theta=theta,
            seed=seed,
            silence=silence,
        )
        E_series_rand[:, j] .= res_rand_bad["E_list"] ./ N
    end

    E_series_rand_mean = vec(mean(E_series_rand; dims=2))
    E_series_rand_std = vec(std(E_series_rand; dims=2))

    # Plot
    plt = get_pyplot()

    plt.rcParams.update(Dict(
        "font.size" => 9,
        "axes.labelsize" => 9,
        "axes.titlesize" => 9,
        "legend.fontsize" => 8,
        "xtick.labelsize" => 8,
        "ytick.labelsize" => 8,
        "lines.linewidth" => 1.6,
        "lines.markersize" => 3.5,
        "pdf.fonttype" => 42,
        "ps.fonttype" => 42,
    ))

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))

    ax = axs[0]
    ax.plot(t_values, E_fixed, color="C0", marker="o", label="fixed time")
    ax.plot(t_values, E_rand_mean, color="C1", marker="s", label="randomized time")
    ax.fill_between(t_values, E_rand_mean .- E_rand_std, E_rand_mean .+ E_rand_std, color="C1", alpha=0.25, linewidth=0)
    ax.axhline(E0 / N, color="black", linestyle="--", linewidth=1.2, label=L"$E_0/N$")
    ax.axvline(t_bad, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(L"mean cycle time $t$")
    ax.set_ylabel(L"steady-state energy density $\bar E/N$")
    ax.set_title("Accidental resonances vs cycle time")
    ax.grid(true, alpha=0.25)
    ax.legend(frameon=false, loc="best")

    ax = axs[1]
    steps_axis = collect(0:steps)
    ax.plot(steps_axis, E_series_fixed, color="C0", label=@sprintf("fixed (t=%.2f)", t_bad))
    ax.plot(steps_axis, E_series_rand_mean, color="C1", label=@sprintf("randomized (mean t=%.2f)", t_bad))
    ax.fill_between(steps_axis, E_series_rand_mean .- E_series_rand_std, E_series_rand_mean .+ E_series_rand_std, color="C1", alpha=0.25, linewidth=0)
    ax.axhline(E0 / N, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("cooling step")
    ax.set_ylabel(L"energy density $E/N$")
    ax.set_title("Example: resonance suppressed")
    ax.grid(true, alpha=0.25)
    ax.legend(frameon=false, loc="best")

    fig.tight_layout()

    save_figure(fig, @__DIR__, output_name)

    data = (
        Δ=Δ,
        E0=E0,
        t_values=t_values,
        E_fixed=E_fixed,
        E_rand_mean=E_rand_mean,
        E_rand_std=E_rand_std,
        t_bad=t_bad,
        E_series_fixed=E_series_fixed,
        E_series_rand_mean=E_series_rand_mean,
        E_series_rand_std=E_series_rand_std,
    )

    return fig, data
end
