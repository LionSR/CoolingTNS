#!/usr/bin/env julia
"""
Scaling study: MC+Trotter cooling for increasing system sizes N.

Verifies that E/N vs cooling step is roughly system-size independent,
scales N up until too slow, profiles bottlenecks.

Usage:
    julia --project=. scripts/scaling_study.jl                    # default
    julia --project=. scripts/scaling_study.jl --N_max 64         # limit max N
    julia --project=. scripts/scaling_study.jl --n_trajectories 100
    julia --project=. scripts/scaling_study.jl --profile          # per-step profiling
"""

if Sys.islinux()
    using MKL
end

using CoolingTNS
using Statistics
using Printf
using HDF5
using PythonCall
using LaTeXStrings
using TimerOutputs

# ============================================================================
# Configuration
# ============================================================================

function parse_scaling_args()
    args = Dict{String,Any}(
        "N_values" => [4, 8, 16, 32, 64, 128],
        "n_trajectories" => 50,
        "profile" => false,
    )

    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--N_max"
            N_max = parse(Int, ARGS[i+1])
            args["N_values"] = filter(n -> n <= N_max, args["N_values"])
            i += 2
        elseif ARGS[i] == "--n_trajectories"
            args["n_trajectories"] = parse(Int, ARGS[i+1])
            i += 2
        elseif ARGS[i] == "--profile"
            args["profile"] = true
            i += 1
        else
            i += 1
        end
    end
    return args
end

# Adaptive trajectory count: use fewer trajectories for larger N to keep runtime bounded
function adaptive_n_traj(N, base_n_traj)
    if N <= 8
        return base_n_traj
    elseif N <= 16
        return max(10, base_n_traj ÷ 3)
    elseif N <= 32
        return max(5, base_n_traj ÷ 10)
    elseif N <= 64
        return max(3, base_n_traj ÷ 20)
    else
        return 2
    end
end

# ============================================================================
# Main scaling study
# ============================================================================

function run_scaling_study()
    config = parse_scaling_args()
    N_values = config["N_values"]
    base_n_traj = config["n_trajectories"]
    do_profile = config["profile"]

    # Fixed physical parameters (resonant cooling)
    g = 1.0
    te = 1.5        # ≈ π/2 for resonant coupling
    tau = 0.1        # 15 Trotter steps per cooling iteration
    steps = 30
    Dmax = 50
    coupling = "XX"
    J = 1.0
    h = 1.0

    println("=" ^ 70)
    println("SCALING STUDY: MC+Trotter Cooling")
    println("=" ^ 70)
    println("N values:       $N_values")
    println("Base traj:      $base_n_traj (adaptive for large N)")
    println("g=$g, te=$te, tau=$tau, steps=$steps, Dmax=$Dmax, coupling=$coupling")
    println("Profiling:      $do_profile")
    println("=" ^ 70)

    # Storage
    all_E_density = Dict{Int, Vector{Float64}}()
    all_timings = Dict{Int, Float64}()
    all_per_traj = Dict{Int, Float64}()
    all_e0_density = Dict{Int, Float64}()
    all_n_traj = Dict{Int, Int}()

    timer = TimerOutput()

    for N in N_values
        n_traj = adaptive_n_traj(N, base_n_traj)
        all_n_traj[N] = n_traj

        println("\n" * "─" ^ 50)
        @printf("N = %d  (trajectories: %d)\n", N, n_traj)
        println("─" ^ 50)

        ham_params = IsingParameters(N, J, h)
        coupling_params = BasicCouplingParameters(coupling, g, steps, te, nothing)
        sim_params = UnifiedSimulationParameters(
            MonteCarloWavefunction(), TrotterEvolution();
            Dmax=Dmax, cutoff=1e-10, tau=tau, n_trajectories=1
        )
        backend = TNBackend()

        # Setup
        t_setup = @elapsed begin
            @timeit timer "setup N=$N" begin
                cooling_problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
            end
        end
        e0 = cooling_problem.e₀
        all_e0_density[N] = e0 / N
        @printf("  Ground state: e₀/N = %.6f (setup: %.1fs)\n", e0/N, t_setup)

        # Single-trajectory timing for profiling (first trajectory)
        if do_profile
            @printf("  Profiling first trajectory...\n")
            initial_state = CoolingTNS.setup_initial_state(cooling_problem, sim_params, "product", 0.0)
            @timeit timer "trajectory N=$N" begin
                CoolingTNS.run_cooling(cooling_problem, initial_state, coupling_params, sim_params, ham_params)
            end
        end

        # Run all trajectories
        E_trajectories = zeros(Float64, steps + 1, n_traj)

        t_cooling = @elapsed begin
            @timeit timer "cooling N=$N" begin
                for traj in 1:n_traj
                    initial_state = CoolingTNS.setup_initial_state(
                        cooling_problem, sim_params, "product", 0.0
                    )
                    results = CoolingTNS.run_cooling(
                        cooling_problem, initial_state,
                        coupling_params, sim_params, ham_params
                    )
                    E_trajectories[:, traj] = results["E_list"]

                    if n_traj >= 10 && traj % max(1, n_traj ÷ 5) == 0
                        @printf("    Trajectory %d/%d done\n", traj, n_traj)
                    end
                end
            end
        end

        E_mean = vec(mean(E_trajectories, dims=2))
        all_E_density[N] = E_mean / N
        all_timings[N] = t_cooling
        all_per_traj[N] = t_cooling / n_traj

        @printf("  Cooling done: %.1fs total, %.2fs/trajectory\n", t_cooling, t_cooling/n_traj)
        @printf("  Final E/N = %.6f (GS: %.6f)\n", E_mean[end]/N, e0/N)

        save_N_results(N, E_mean, e0, coupling_params, sim_params, n_traj)

        # Abort if a single trajectory takes too long
        if t_cooling / n_traj > 600
            @printf("\n  ⚠ Per-trajectory time exceeds 10 minutes. Stopping at N=%d.\n", N)
            N_values = N_values[1:findfirst(==(N), N_values)]
            break
        end
    end

    # Timing summary
    println("\n" * "=" ^ 70)
    println("TIMING SUMMARY")
    println("=" ^ 70)
    @printf("%-6s  %-6s  %-10s  %-12s  %-10s  %-10s\n",
            "N", "Traj", "Total(s)", "Per traj(s)", "Final E/N", "GS E/N")
    println("-" ^ 62)
    for N in N_values
        haskey(all_timings, N) || continue
        @printf("%-6d  %-6d  %-10.1f  %-12.2f  %-10.6f  %-10.6f\n",
                N, all_n_traj[N], all_timings[N], all_per_traj[N],
                all_E_density[N][end], all_e0_density[N])
    end

    # Scaling exponent
    completed_N = filter(N -> haskey(all_per_traj, N), N_values)
    if length(completed_N) >= 2
        println("\nScaling exponents (per-trajectory time):")
        for i in 2:length(completed_N)
            N1, N2 = completed_N[i-1], completed_N[i]
            t1, t2 = all_per_traj[N1], all_per_traj[N2]
            α = log(t2/t1) / log(N2/N1)
            @printf("  N=%d→%d: α = %.2f  (t ∝ N^α)\n", N1, N2, α)
        end
    end

    if do_profile
        println("\n" * "=" ^ 70)
        println("PROFILING (TimerOutputs)")
        println("=" ^ 70)
        show(timer)
        println()
    end

    plot_scaling_results(N_values, all_E_density, all_e0_density, all_per_traj, steps)

    println("\nScaling study complete.")
end

# ============================================================================
# Save results
# ============================================================================

function save_N_results(N, E_mean, e0, coupling_params, sim_params, n_traj)
    mkpath("Results")
    filename = "Results/scaling_N$(N).h5"
    h5open(filename, "w") do file
        write(file, "N", N)
        write(file, "e0", e0)
        write(file, "E_list", E_mean)
        write(file, "E_density", E_mean / N)
        write(file, "n_trajectories", n_traj)
        write(file, "g", coupling_params.g)
        write(file, "te", coupling_params.te)
        write(file, "tau", sim_params.tau)
        write(file, "Dmax", sim_params.Dmax)
        write(file, "steps", coupling_params.steps)
    end
    println("  Saved: $filename")
end

# ============================================================================
# Plotting
# ============================================================================

function plot_scaling_results(N_values, all_E_density, all_e0_density, all_per_traj, steps)
    plt = CoolingTNS.get_pyplot()
    mkpath("Results/Figs")

    completed = filter(N -> haskey(all_E_density, N), N_values)

    # --- Figure 1: E/N vs cooling step ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    n_curves = length(completed)
    colors = plt.cm.viridis(range(0.1, 0.9, length=n_curves))

    for (i, N) in enumerate(completed)
        E_dens = all_E_density[N]
        ax.plot(0:steps, E_dens, color=colors[i-1], linewidth=1.5,
                marker="o", markersize=3, label="N=$N")
    end

    N_ref = completed[end]
    ax.axhline(y=all_e0_density[N_ref], color="black", linewidth=1.5,
               linestyle="--", alpha=0.7, label=L"$E_0/N$ (N=%$N_ref)")

    ax.set_xlabel("Cooling step", fontsize=12)
    ax.set_ylabel(L"Energy density $E/N$", fontsize=12)
    ax.set_title("MC+Trotter Cooling: System Size Scaling", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(true, alpha=0.3)
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.12, top=0.92)

    figpath = "Results/Figs/scaling_energy_density.pdf"
    fig.savefig(figpath, dpi=300)
    println("Plot saved: $figpath")
    plt.close(fig)

    # --- Figure 2: Per-trajectory timing vs N (log-log) ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    Ns = Float64.(collect(completed))
    times = [all_per_traj[N] for N in completed]

    ax.loglog(Ns, times, "o-", color="steelblue", linewidth=2, markersize=6, label="Per trajectory")

    # Reference scaling lines
    t_ref = times[1]
    N_ref_val = Ns[1]
    ax.loglog(Ns, t_ref .* (Ns ./ N_ref_val) .^ 1, "--", color="gray",
              alpha=0.5, label=L"$\propto N$")
    ax.loglog(Ns, t_ref .* (Ns ./ N_ref_val) .^ 2, ":", color="gray",
              alpha=0.5, label=L"$\propto N^2$")
    ax.loglog(Ns, t_ref .* (Ns ./ N_ref_val) .^ 3, "-.", color="gray",
              alpha=0.4, label=L"$\propto N^3$")

    ax.set_xlabel("System size N", fontsize=12)
    ax.set_ylabel("Wall time per trajectory (s)", fontsize=12)
    ax.set_title("Timing Scaling (MC+Trotter)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(true, alpha=0.3, which="both")
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.14, top=0.92)

    figpath = "Results/Figs/scaling_timing.pdf"
    fig.savefig(figpath, dpi=300)
    println("Plot saved: $figpath")
    plt.close(fig)
end

# ============================================================================
# Entry point
# ============================================================================

run_scaling_study()
