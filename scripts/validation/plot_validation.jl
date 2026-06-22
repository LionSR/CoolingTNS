"""
    plot_validation.jl

Publication-quality validation plots for CoolingTNS.
Generates 5 PDF figures demonstrating consistency across backends (ED/TN),
simulation methods (DM/MC), and evolution methods (Continuous/Trotter).

Run: julia --project=. scripts/validation/plot_validation.jl
"""

using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra
using Statistics
using Random
using PythonCall
using LaTeXStrings

Random.seed!(42)

# ============================================================================
# Helper: run a cooling simulation and return results + problem
# ============================================================================
function run_sim(; backend_str, sim_method_str, evolution_method_str,
                   ham_params, coupling_params,
                   Dmax=50, cutoff=1e-10, tau=0.1, pe=0.0,
                   n_trajectories=1, init_type="product", theta=0.0)
    backend = CoolingTNS.get_backend(backend_str)
    sim_method = CoolingTNS.get_sim_method(sim_method_str)
    evolution_method = CoolingTNS.get_evolution_method(evolution_method_str)

    sim_params = CoolingTNS.UnifiedSimulationParameters(
        sim_method, evolution_method;
        Dmax=Dmax, cutoff=cutoff, tau=tau, pe=pe,
        n_trajectories=n_trajectories
    )

    problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
    state0  = CoolingTNS.setup_initial_state(problem, sim_params, init_type, theta)
    results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)
    return results, problem
end

# ============================================================================
# Plotting setup
# ============================================================================
const plt = pyimport("matplotlib.pyplot")
const mpl = pyimport("matplotlib")
mpl.rcParams["font.size"] = 12

const FIGDIR = joinpath(@__DIR__, "..", "Results", "Figs")
mkpath(FIGDIR)

# ============================================================================
# Common parameters
# ============================================================================
const N_SYS = 4
const STEPS = 200
const TE = 1.0
const G = 0.2
const TAU_DEFAULT = 0.05

const HAM_PARAMS = CoolingTNS.IsingParameters(N_SYS, 1.0, 1.0)

function make_coupling(steps)
    CoolingTNS.BasicCouplingParameters("XX", G, steps, TE, nothing)
end

# ============================================================================
# Figure 1: Method Consistency — Energy & GS Overlap (ED, N=4)
# ============================================================================
function figure1_method_consistency()
    println("\n" * "="^60)
    println("Figure 1: Method Consistency (ED, N=$N_SYS, $STEPS steps)")
    println("="^60)

    cp = make_coupling(STEPS)

    # DM + Continuous (gold standard)
    println("  Running ED DM+Continuous...")
    res_dm_cont, prob_dm = run_sim(backend_str="ED", sim_method_str="density_matrix",
                                   evolution_method_str="continuous",
                                   ham_params=HAM_PARAMS, coupling_params=cp)

    # DM + Trotter
    println("  Running ED DM+Trotter (tau=$TAU_DEFAULT)...")
    res_dm_trot, _ = run_sim(backend_str="ED", sim_method_str="density_matrix",
                             evolution_method_str="trotter", tau=TAU_DEFAULT,
                             ham_params=HAM_PARAMS, coupling_params=cp)

    # MC + Continuous (50 trajectories)
    n_traj = 50
    println("  Running ED MC+Continuous ($n_traj trajectories)...")
    mc_E_lists = Vector{Vector{Float64}}()
    mc_ov_lists = Vector{Vector{Float64}}()
    for t in 1:n_traj
        res_mc, _ = run_sim(backend_str="ED", sim_method_str="monte_carlo",
                            evolution_method_str="continuous",
                            ham_params=HAM_PARAMS, coupling_params=cp)
        push!(mc_E_lists, res_mc[CoolingTNS.RESULT_ENERGY])
        push!(mc_ov_lists, res_mc[CoolingTNS.RESULT_GROUND_STATE_OVERLAP])
    end
    mc_E_avg = mean(mc_E_lists)
    mc_E_std = std(mc_E_lists)
    mc_ov_avg = mean(mc_ov_lists)
    mc_ov_std = std(mc_ov_lists)

    steps_ax = 0:STEPS

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Energy density
    ax = axs[0]
    ax.plot(steps_ax, res_dm_cont[CoolingTNS.RESULT_ENERGY] ./ N_SYS,
            linewidth=2, label="DM + Continuous", color="C0")
    ax.plot(steps_ax, res_dm_trot[CoolingTNS.RESULT_ENERGY] ./ N_SYS,
            linewidth=2, linestyle="--", label="DM + Trotter", color="C1")
    ax.errorbar(collect(steps_ax), mc_E_avg ./ N_SYS, yerr=mc_E_std ./ (N_SYS * sqrt(n_traj)),
                linewidth=1.5, linestyle=":", capsize=2, label="MC + Continuous", color="C2",
                errorevery=5, markevery=5, marker="o", markersize=4)
    ax.axhline(y=prob_dm.e₀ / N_SYS, linewidth=1.5, color="black", linestyle="--",
               alpha=0.5, label=L"$E_0/N$")
    ax.set_xlabel("Cooling step", fontsize=14)
    ax.set_ylabel(L"$E/N$", fontsize=14)
    ax.set_title("Energy density", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(true, alpha=0.3)

    # Right: GS overlap
    ax = axs[1]
    ax.plot(steps_ax, res_dm_cont[CoolingTNS.RESULT_GROUND_STATE_OVERLAP],
            linewidth=2, label="DM + Continuous", color="C0")
    ax.plot(steps_ax, res_dm_trot[CoolingTNS.RESULT_GROUND_STATE_OVERLAP],
            linewidth=2, linestyle="--", label="DM + Trotter", color="C1")
    ax.errorbar(collect(steps_ax), mc_ov_avg, yerr=mc_ov_std ./ sqrt(n_traj),
                linewidth=1.5, linestyle=":", capsize=2, label="MC + Continuous", color="C2",
                errorevery=5, markevery=5, marker="o", markersize=4)
    ax.set_xlabel("Cooling step", fontsize=14)
    ax.set_ylabel("GS overlap", fontsize=14)
    ax.set_title("Ground state overlap", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(true, alpha=0.3)

    fig.tight_layout()
    path = joinpath(FIGDIR, "validation_method_consistency.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    println("  Saved: $path")
end

# ============================================================================
# Figure 2: TN Gate Trotter Convergence — Error vs tau
# ============================================================================
function figure2_tn_trotter_convergence()
    N_trotter = 3
    steps_trotter = 60
    ham_trotter = CoolingTNS.IsingParameters(N_trotter, 1.0, 1.0)
    cp_auto = CoolingTNS.BasicCouplingParameters("XX", G, steps_trotter, TE, nothing)
    tau_values = [0.5, 0.2, 0.1, 0.05]

    println("\n" * "="^60)
    println("Figure 2: TN Gate Trotter Convergence (N=$N_trotter)")
    println("="^60)

    println("  Running ED continuous reference...")
    res_ref, prob_ref = run_sim(backend_str="ED", sim_method_str="density_matrix",
                                evolution_method_str="continuous",
                                ham_params=ham_trotter, coupling_params=cp_auto)
    delta_ref = prob_ref.extra.coupling_params.delta
    cp = CoolingTNS.BasicCouplingParameters("XX", G, steps_trotter, TE, delta_ref)
    println("  Using shared bath detuning Delta=$delta_ref for TN runs.")
    E_ref = res_ref[CoolingTNS.RESULT_ENERGY][end] / N_trotter

    E_tn = Float64[]
    for tau in tau_values
        println("  Running TN DM+Trotter tau=$tau...")
        res, _ = run_sim(backend_str="TN", sim_method_str="density_matrix",
                         evolution_method_str="trotter", tau=tau,
                         ham_params=ham_trotter, coupling_params=cp,
                         Dmax=200, cutoff=1e-12)
        push!(E_tn, res[CoolingTNS.RESULT_ENERGY][end] / N_trotter)
    end

    errors = abs.(E_tn .- E_ref)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Final E/N vs tau
    ax = axs[0]
    ax.plot(tau_values, E_tn, "o-", linewidth=2, markersize=6, color="C0", label="TN Trotter")
    ax.axhline(y=E_ref, linewidth=1.5, color="black", linestyle="--",
               label="ED continuous reference")
    ax.set_xlabel(L"$\tau$", fontsize=14)
    ax.set_ylabel(L"Final $E/N$", fontsize=14)
    ax.set_title("Final energy vs gate step", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(true, alpha=0.3)

    # Right: log-log error vs tau
    ax = axs[1]
    # Filter out zero errors for log-log
    mask = errors .> 0
    if any(mask)
        ax.loglog(tau_values[mask], errors[mask], "o-", linewidth=2, markersize=6,
                  color="C0", label=L"|E_{\mathrm{TN}}(\tau)-E_{\mathrm{ED}}|")
        if count(mask) >= 2
            tau_ref = tau_values[mask]
            slope2 = errors[mask][end] .* (tau_ref ./ tau_ref[end]).^2
            ax.loglog(tau_ref, slope2, "--", linewidth=1.5, color="gray",
                      label=L"$\mathcal{O}(\tau^2)$ guide")
        end
    end
    ax.set_xlabel(L"$\tau$", fontsize=14)
    ax.set_ylabel("Energy error", fontsize=14)
    ax.set_title("TN gate approximation error", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(true, alpha=0.3, which="both")

    fig.tight_layout()
    path = joinpath(FIGDIR, "validation_tn_trotter_convergence.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    println("  Saved: $path")
end

# ============================================================================
# Figure 3: Cross-Backend — ED vs TN DM+Trotter (N=3)
# ============================================================================
function figure3_cross_backend()
    N_cross = 3
    println("\n" * "="^60)
    println("Figure 3: Cross-Backend ED vs TN (DM+Trotter, N=$N_cross)")
    println("="^60)

    ham_cross = CoolingTNS.IsingParameters(N_cross, 1.0, 1.0)
    cp_cross_auto = CoolingTNS.BasicCouplingParameters("XX", G, STEPS, TE, nothing)
    tau_cross = 0.05

    println("  Running ED DM+Trotter...")
    res_ed, prob_cross = run_sim(backend_str="ED", sim_method_str="density_matrix",
                                evolution_method_str="trotter", tau=tau_cross,
                                ham_params=ham_cross, coupling_params=cp_cross_auto)
    delta_cross = prob_cross.extra.coupling_params.delta
    cp_cross = CoolingTNS.BasicCouplingParameters("XX", G, STEPS, TE, delta_cross)
    println("  Using shared cross-backend bath detuning Delta=$delta_cross.")

    println("  Running TN DM+Trotter...")
    res_tn, _ = run_sim(backend_str="TN", sim_method_str="density_matrix",
                        evolution_method_str="trotter", tau=tau_cross,
                        ham_params=ham_cross, coupling_params=cp_cross, Dmax=100)

    steps_ax = 0:STEPS
    E_ed = res_ed[CoolingTNS.RESULT_ENERGY] ./ N_cross
    E_tn = res_tn[CoolingTNS.RESULT_ENERGY] ./ N_cross
    diff = abs.(E_ed .- E_tn)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Left: E/N vs step
    ax = axs[0]
    ax.plot(steps_ax, E_ed, linewidth=2, label="ED", color="C0")
    ax.plot(steps_ax, E_tn, linewidth=2, linestyle="--", label="TN", color="C1")
    ax.axhline(y=prob_cross.e₀ / N_cross, linewidth=1.5, color="black", linestyle="--",
               alpha=0.5, label=L"$E_0/N$")
    ax.set_xlabel("Cooling step", fontsize=14)
    ax.set_ylabel(L"$E/N$", fontsize=14)
    ax.set_title("ED vs TN energy density", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(true, alpha=0.3)

    # Right: |E_ED - E_TN| per step
    ax = axs[1]
    ax.semilogy(steps_ax, diff .+ 1e-16, linewidth=2, color="C3")
    ax.set_xlabel("Cooling step", fontsize=14)
    ax.set_ylabel(L"|E_{\mathrm{ED}} - E_{\mathrm{TN}}|/N", fontsize=14)
    ax.set_title("Cross-backend discrepancy", fontsize=16)
    ax.grid(true, alpha=0.3, which="both")

    fig.tight_layout()
    path = joinpath(FIGDIR, "validation_cross_backend.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    println("  Saved: $path")
end

# ============================================================================
# Figure 4: Physical Invariants — Purity & Phase diagram
# ============================================================================
function figure4_physical_invariants()
    println("\n" * "="^60)
    println("Figure 4: Physical Invariants (ED DM+Continuous, N=$N_SYS)")
    println("="^60)

    cp = make_coupling(STEPS)

    println("  Running ED DM+Continuous...")
    res, prob_inv = run_sim(backend_str="ED", sim_method_str="density_matrix",
                            evolution_method_str="continuous",
                            ham_params=HAM_PARAMS, coupling_params=cp)

    steps_ax = 0:STEPS
    purity = res[CoolingTNS.RESULT_PURITY]
    E_density = res[CoolingTNS.RESULT_ENERGY] ./ N_SYS
    E0_density = prob_inv.e₀ / N_SYS

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Purity vs step
    ax = axs[0]
    ax.plot(steps_ax, purity, linewidth=2, color="C4")
    ax.axhline(y=1.0 / 2^N_SYS, linewidth=1.5, color="black", linestyle="--",
               label=L"$1/2^N$ (max. mixed)")
    ax.set_xlabel("Cooling step", fontsize=14)
    ax.set_ylabel("Purity " * L"$\mathrm{Tr}(\rho^2)$", fontsize=14)
    ax.set_title("Purity evolution", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(true, alpha=0.3)

    # Right: E/N vs purity phase diagram
    ax = axs[1]
    sc = ax.scatter(pylist(purity), pylist(E_density), c=pylist(collect(steps_ax)),
                    cmap="viridis", s=30, edgecolors="none")
    ax.plot(purity, E_density, linewidth=0.8, alpha=0.5, color="gray")
    # Mark start and end
    ax.plot(purity[1], E_density[1], "o", markersize=10, color="C3", label="Start", zorder=5)
    ax.plot(purity[end], E_density[end], "s", markersize=10, color="C0", label="End", zorder=5)
    ax.axhline(y=E0_density, linewidth=1.5, color="black", linestyle="--",
               alpha=0.5, label=L"$E_0/N$")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Cooling step", fontsize=12)
    ax.set_xlabel("Purity " * L"$\mathrm{Tr}(\rho^2)$", fontsize=14)
    ax.set_ylabel(L"$E/N$", fontsize=14)
    ax.set_title("Energy–purity trajectory", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(true, alpha=0.3)

    fig.tight_layout()
    path = joinpath(FIGDIR, "validation_physical_invariants.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    println("  Saved: $path")
end

# ============================================================================
# Figure 5: Ising vs niIsing — Parity-preserving cooling
# ============================================================================
function figure5_ising_vs_niising()
    println("\n" * "="^60)
    println("Figure 5: Ising vs niIsing (ED DM+Continuous, N=$N_SYS)")
    println("="^60)

    cp = make_coupling(STEPS)
    ham_ising = CoolingTNS.IsingParameters(N_SYS, 1.0, 1.0)
    ham_niising = CoolingTNS.NiIsingParameters(N_SYS, 1.0, -1.05, 0.5)

    println("  Running Ising (parity-preserving)...")
    res_ising, prob_ising = run_sim(backend_str="ED", sim_method_str="density_matrix",
                                    evolution_method_str="continuous",
                                    ham_params=ham_ising, coupling_params=cp)

    println("  Running niIsing (parity-breaking)...")
    res_niising, prob_niising = run_sim(backend_str="ED", sim_method_str="density_matrix",
                                        evolution_method_str="continuous",
                                        ham_params=ham_niising, coupling_params=cp)

    steps_ax = 0:STEPS

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Energy density
    ax = axs[0]
    ax.plot(steps_ax, res_ising[CoolingTNS.RESULT_ENERGY] ./ N_SYS,
            linewidth=2, label="Ising (parity ✓)", color="C0")
    ax.plot(steps_ax, res_niising[CoolingTNS.RESULT_ENERGY] ./ N_SYS,
            linewidth=2, linestyle="--", label="niIsing (parity ✗)", color="C3")
    ax.axhline(y=prob_ising.e₀ / N_SYS, linewidth=1, color="C0", linestyle=":",
               alpha=0.5, label="Ising GS")
    ax.axhline(y=prob_niising.e₀ / N_SYS, linewidth=1, color="C3", linestyle=":",
               alpha=0.5, label="niIsing GS")
    ax.set_xlabel("Cooling step", fontsize=14)
    ax.set_ylabel(L"$E/N$", fontsize=14)
    ax.set_title("Energy density", fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(true, alpha=0.3)

    # Right: GS overlap
    ax = axs[1]
    ax.plot(steps_ax, res_ising[CoolingTNS.RESULT_GROUND_STATE_OVERLAP],
            linewidth=2, label="Ising (parity ✓)", color="C0")
    ax.plot(steps_ax, res_niising[CoolingTNS.RESULT_GROUND_STATE_OVERLAP],
            linewidth=2, linestyle="--", label="niIsing (parity ✗)", color="C3")
    ax.set_xlabel("Cooling step", fontsize=14)
    ax.set_ylabel("GS overlap", fontsize=14)
    ax.set_title("Ground state overlap", fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(true, alpha=0.3)

    fig.tight_layout()
    path = joinpath(FIGDIR, "validation_ising_vs_niising.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    println("  Saved: $path")
end

# ============================================================================
# Main
# ============================================================================
function main()
    println("CoolingTNS Validation Plots")
    println("Model: Ising (N=$N_SYS, J=1.0, h=1.0)")
    println("Coupling: XX, g=$G, te=$TE")
    println("Steps: $STEPS")
    println("Output: $FIGDIR")

    figure1_method_consistency()
    figure2_tn_trotter_convergence()
    figure3_cross_backend()
    figure4_physical_invariants()
    figure5_ising_vs_niising()

    println("\n" * "="^60)
    println("All validation figures generated successfully!")
    println("="^60)
end

main()
