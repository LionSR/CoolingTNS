"""
    randomized_time_resonance_figure.jl

Shared utilities to generate a publication-style figure demonstrating how
randomizing cycle times suppresses accidental resonances/heating.

This is inspired by the randomized-time analysis in arXiv:2503.24330.

The figure typically has two panels:

1. A scan of the finite-window late-time energy density versus the *mean* cycle
   time `t`.
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

"""
    _energy_metric(E, E0, N, metric) -> Float64

Convert a total energy `E` to the plotting metric.

- `metric = :energy_density`: returns `E/N`.
- `metric = :energy_density_error`: returns the intensive energy error
  `\\lvert E/N - E_0/N \\rvert`.
- `metric = :relative_energy`: returns the dimensionless relative energy
  `e = \\lvert (E-E_0)/E_0 \\rvert`.

Here `E_0` is the ground-state energy.
"""
function _energy_metric(E::Real, E0::Real, N::Int, metric::Symbol)::Float64
    metric == :energy_density && return Float64(E) / N
    metric == :energy_density_error && return abs(Float64(E) / N - Float64(E0) / N)
    metric == :relative_energy && return relative_energy(Float64(E), Float64(E0))
    throw(ArgumentError(
        "Unknown metric=$metric (use :energy_density, :energy_density_error, or :relative_energy)",
    ))
end

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
    delta_values::Vector{Float64};
    coupling::String,
    g::Float64,
    steps::Int,
    t_mean::Float64,
    init_type::String,
    theta::Float64,
    schedule::Symbol=:round_robin,
    silence::Bool=true,
)
    cp_fixed = if length(delta_values) == 1
        BasicCouplingParameters(coupling, g, steps, t_mean, delta_values[1])
    else
        MultiFrequencyCouplingParameters(
            coupling,
            g,
            steps,
            t_mean,
            delta_values;
            randomize_times=false,
            schedule=schedule,
        )
    end

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
    delta_values::Vector{Float64};
    coupling::String,
    g::Float64,
    steps::Int,
    t_mean::Float64,
    init_type::String,
    theta::Float64,
    seed::Int,
    schedule::Symbol=:round_robin,
    silence::Bool=true,
)
    Random.seed!(seed)
    _maybe_clear_ed_cache!(backend)

    cp_rand = MultiFrequencyCouplingParameters(
        coupling,
        g,
        steps,
        t_mean,
        delta_values;
        randomize_times=true,
        schedule=schedule,
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
    # Plotting metric (paper default): relative energy e = |(E-E0)/E0|.
    metric::Symbol=:relative_energy,
    R::Int=1,
    delta_max_factor::Float64=6.0,
    delta_strategy::Symbol=:uniform,
    delta_values::Union{Nothing, Vector{Float64}}=nothing,
    schedule::Symbol=:round_robin,
    output_name::String="randomized_times_remove_accidental_heating.pdf",
    silence::Bool=true,
)
    N = ham_params.N

    Δ, E0 = _compute_gap_and_e0(backend, ham_params, sim_params; coupling=coupling, g=g)

    delta_values_used = if delta_values !== nothing
        Float64.(delta_values)
    elseif R <= 1
        [Δ]
    elseif delta_strategy == :uniform
        uniform_delta_grid(Δ, delta_max_factor * Δ, R)
    elseif delta_strategy == :spectral
        spectral_delta_values(ham_params, backend; R=R)
    else
        throw(ArgumentError("Unknown delta_strategy=$delta_strategy (use :uniform or :spectral)"))
    end

    @printf(
        "Using backend=%s, N=%d, g=%.3f, Δ(gap)=%.6f, R=%d\n",
        string(typeof(backend)),
        N,
        g,
        Δ,
        length(delta_values_used),
    )
    if length(delta_values_used) > 1
        @printf("  Δ grid: [%.6f, %.6f]\n", minimum(delta_values_used), maximum(delta_values_used))
    end

    metric in (:energy_density, :energy_density_error, :relative_energy) ||
        throw(ArgumentError("Unknown metric=$metric (use :energy_density, :energy_density_error, or :relative_energy)"))

    metric_str = if metric == :energy_density
        "E/N"
    elseif metric == :energy_density_error
        "|E/N-E0/N|"
    else
        "e"
    end

    nt = length(t_values)
    E_fixed_mean = fill(NaN, nt)
    E_fixed_std = fill(0.0, nt)
    E_rand_mean = fill(NaN, nt)
    E_rand_std = fill(NaN, nt)

    is_mc = sim_params.sim_method isa MonteCarloWavefunction

    for (i, t̄) in enumerate(t_values)
        # ------------------------------------------------------------
        # Fixed-time protocol
        # ------------------------------------------------------------
        if is_mc
            Ef = Float64[]
            for seed in seed_list
                Random.seed!(seed)
                res_fixed = _run_fixed_time(
                    backend,
                    ham_params,
                    sim_params,
                    delta_values_used;
                    coupling=coupling,
                    g=g,
                    steps=steps,
                    t_mean=t̄,
                    init_type=init_type,
                    theta=theta,
                    schedule=schedule,
                    silence=silence,
                )
                push!(Ef, _energy_metric(_mean_last(res_fixed[CoolingTNS.RESULT_ENERGY], window_size), E0, N, metric))
            end
            E_fixed_mean[i] = mean(Ef)
            E_fixed_std[i] = std(Ef)
        else
            res_fixed = _run_fixed_time(
                backend,
                ham_params,
                sim_params,
                delta_values_used;
                coupling=coupling,
                g=g,
                steps=steps,
                t_mean=t̄,
                init_type=init_type,
                theta=theta,
                schedule=schedule,
                silence=silence,
            )
            E_fixed_mean[i] = _energy_metric(_mean_last(res_fixed[CoolingTNS.RESULT_ENERGY], window_size), E0, N, metric)
            E_fixed_std[i] = 0.0
        end

        # ------------------------------------------------------------
        # Randomized-time protocol (always averaged over seeds)
        # ------------------------------------------------------------
        Es = Float64[]
        for seed in seed_list
            res_rand = _run_randomized_time_once(
                backend,
                ham_params,
                sim_params,
                delta_values_used;
                coupling=coupling,
                g=g,
                steps=steps,
                t_mean=t̄,
                init_type=init_type,
                theta=theta,
                seed=seed,
                schedule=schedule,
                silence=silence,
            )
            push!(Es, _energy_metric(_mean_last(res_rand[CoolingTNS.RESULT_ENERGY], window_size), E0, N, metric))
        end

        E_rand_mean[i] = mean(Es)
        E_rand_std[i] = std(Es)

        if is_mc
            @printf("t=%.3f  fixed(%s)=%.6f ± %.6f  randomized(%s)=%.6f ± %.6f\n",
                    t̄, metric_str, E_fixed_mean[i], E_fixed_std[i], metric_str, E_rand_mean[i], E_rand_std[i])
        else
            @printf("t=%.3f  fixed(%s)=%.6f  randomized(%s)=%.6f ± %.6f\n",
                    t̄, metric_str, E_fixed_mean[i], metric_str, E_rand_mean[i], E_rand_std[i])
        end
    end

    improvement = E_fixed_mean .- E_rand_mean
    idx_bad = argmax(improvement)
    t_bad = t_values[idx_bad]
    @printf("\nRepresentative accidental-resonance point: t_bad=%.3f (Δ%s=%.6f)\n", t_bad, metric_str, improvement[idx_bad])

    # Time series at t_bad
    if is_mc
        E_series_fixed = zeros(Float64, steps + 1, length(seed_list))
        for (j, seed) in enumerate(seed_list)
            Random.seed!(seed)
            res_fixed_bad = _run_fixed_time(
                backend,
                ham_params,
                sim_params,
                delta_values_used;
                coupling=coupling,
                g=g,
                steps=steps,
                t_mean=t_bad,
                init_type=init_type,
                theta=theta,
                schedule=schedule,
                silence=silence,
            )
            E_series_fixed[:, j] .= [_energy_metric(E, E0, N, metric) for E in res_fixed_bad[CoolingTNS.RESULT_ENERGY]]
        end
        E_series_fixed_mean = vec(mean(E_series_fixed; dims=2))
        E_series_fixed_std = vec(std(E_series_fixed; dims=2))
    else
        res_fixed_bad = _run_fixed_time(
            backend,
            ham_params,
            sim_params,
            delta_values_used;
            coupling=coupling,
            g=g,
            steps=steps,
            t_mean=t_bad,
            init_type=init_type,
            theta=theta,
            schedule=schedule,
            silence=silence,
        )
        E_series_fixed_mean = [_energy_metric(E, E0, N, metric) for E in res_fixed_bad[CoolingTNS.RESULT_ENERGY]]
        E_series_fixed_std = zeros(Float64, steps + 1)
    end

    E_series_rand = zeros(Float64, steps + 1, length(seed_list))
    for (j, seed) in enumerate(seed_list)
        res_rand_bad = _run_randomized_time_once(
            backend,
            ham_params,
            sim_params,
            delta_values_used;
            coupling=coupling,
            g=g,
            steps=steps,
            t_mean=t_bad,
            init_type=init_type,
            theta=theta,
            seed=seed,
            schedule=schedule,
            silence=silence,
        )
        E_series_rand[:, j] .= [_energy_metric(E, E0, N, metric) for E in res_rand_bad[CoolingTNS.RESULT_ENERGY]]
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

    y_gs = metric == :energy_density ? (E0 / N) : 0.0

    gs_label = if metric == :energy_density
        L"$E_0/N$"
    elseif metric == :energy_density_error
        L"$0$"
    else
        L"$e=0$"
    end

    ylab_scan = if metric == :energy_density
        L"late-time mean energy density $\bar E/N$"
    elseif metric == :energy_density_error
        L"late-time mean energy density error $|\bar E/N - E_0/N|$"
    else
        L"late-time mean relative energy $\bar{e}$"
    end

    ylab = if metric == :energy_density
        L"energy density $E/N$"
    elseif metric == :energy_density_error
        L"energy density error $|E/N - E_0/N|$"
    else
        L"relative energy $e$"
    end

    ax = axs[0]
    ax.plot(t_values, E_fixed_mean, color="C0", marker="o", label="fixed time")
    ax.fill_between(t_values, E_fixed_mean .- E_fixed_std, E_fixed_mean .+ E_fixed_std, color="C0", alpha=0.20, linewidth=0)
    ax.plot(t_values, E_rand_mean, color="C1", marker="s", label="randomized time")
    ax.fill_between(t_values, E_rand_mean .- E_rand_std, E_rand_mean .+ E_rand_std, color="C1", alpha=0.25, linewidth=0)
    ax.axhline(y_gs, color="black", linestyle="--", linewidth=1.2, label=gs_label)
    ax.axvline(t_bad, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.set_xlabel(L"mean cycle time $t$")
    ax.set_ylabel(ylab_scan)
    ax.set_title("Accidental resonances vs cycle time")
    ax.grid(true, alpha=0.25)
    ax.legend(frameon=false, loc="best")

    ax = axs[1]
    steps_axis = collect(0:steps)
    ax.plot(steps_axis, E_series_fixed_mean, color="C0", label=@sprintf("fixed (t=%.2f)", t_bad))
    ax.fill_between(steps_axis, E_series_fixed_mean .- E_series_fixed_std, E_series_fixed_mean .+ E_series_fixed_std, color="C0", alpha=0.20, linewidth=0)
    ax.plot(steps_axis, E_series_rand_mean, color="C1", label=@sprintf("randomized (mean t=%.2f)", t_bad))
    ax.fill_between(steps_axis, E_series_rand_mean .- E_series_rand_std, E_series_rand_mean .+ E_series_rand_std, color="C1", alpha=0.25, linewidth=0)
    ax.axhline(y_gs, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("cooling step")
    ax.set_ylabel(ylab)
    ax.set_title("Example: resonance suppressed")
    ax.grid(true, alpha=0.25)
    ax.legend(frameon=false, loc="best")

    fig.tight_layout()

    save_figure(fig, @__DIR__, output_name)

    data = (
        Δ=Δ,
        delta_values=delta_values_used,
        E0=E0,
        metric=string(metric),
        schedule=string(schedule),
        t_values=t_values,
        E_fixed_mean=E_fixed_mean,
        E_fixed_std=E_fixed_std,
        E_rand_mean=E_rand_mean,
        E_rand_std=E_rand_std,
        t_bad=t_bad,
        E_series_fixed_mean=E_series_fixed_mean,
        E_series_fixed_std=E_series_fixed_std,
        E_series_rand_mean=E_series_rand_mean,
        E_series_rand_std=E_series_rand_std,
    )

    return fig, data
end
