"""
    test_mc_continuous_vs_ed.jl

Compare TN MC+Continuous against ED MC+Continuous and ED DM+Continuous
to determine if the TN TDVP evolution is correct.
"""

using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra
using Statistics
using Printf
using Random

Random.seed!(42)

println("="^70)
println("TN MC+Continuous vs ED Reference")
println("="^70)

N = 3
ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)
g = 0.2
te = 1.0
n_steps = 5
coupling_params = CoolingTNS.BasicCouplingParameters("XX", g, n_steps, te, nothing)

# ============================================================================
# ED DM+Continuous (exact reference)
# ============================================================================
println("\n--- ED DM+Continuous (exact) ---")
results_ed_dm, prob_ed = let
    backend = CoolingTNS.EDBackend()
    sim_method = CoolingTNS.DensityMatrix()
    evolution_method = CoolingTNS.ContinuousEvolution()
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        sim_method, evolution_method; Dmax=50, cutoff=1e-10, tau=0.1, pe=0.0, n_trajectories=1
    )
    problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
    state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
    results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)
    results, problem
end

# ============================================================================
# ED MC+Continuous (100 trajectories)
# ============================================================================
println("\n--- ED MC+Continuous (100 trajectories) ---")
n_traj = 100
ed_mc_E_lists = Vector{Vector{Float64}}()
for _ in 1:n_traj
    results_mc, _ = let
        backend = CoolingTNS.EDBackend()
        sim_method = CoolingTNS.MonteCarloWavefunction()
        evolution_method = CoolingTNS.ContinuousEvolution()
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            sim_method, evolution_method; Dmax=50, cutoff=1e-10, tau=0.1, pe=0.0, n_trajectories=1
        )
        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
        results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)
        results, problem
    end
    push!(ed_mc_E_lists, results_mc["E_list"])
end
ed_mc_E_avg = mean(ed_mc_E_lists)
ed_mc_E_stderr = [std([ed_mc_E_lists[t][s] for t in 1:n_traj]) for s in 1:n_steps+1] ./ sqrt(n_traj)

# ============================================================================
# TN MC+Continuous (100 trajectories, with different tau for TDVP)
# ============================================================================
println("\n--- TN MC+Continuous (100 trajectories, different TDVP tau) ---")
for tdvp_tau in [0.1, 0.01]
    println("\n  TDVP tau = $tdvp_tau")
    tn_mc_E_lists = Vector{Vector{Float64}}()
    for _ in 1:n_traj
        results_mc, _ = let
            backend = CoolingTNS.TNBackend()
            sim_method = CoolingTNS.MonteCarloWavefunction()
            evolution_method = CoolingTNS.ContinuousEvolution()
            sim_params = CoolingTNS.UnifiedSimulationParameters(
                sim_method, evolution_method; Dmax=200, cutoff=1e-12, tau=tdvp_tau, pe=0.0, n_trajectories=1
            )
            problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
            state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
            results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)
            results, problem
        end
        push!(tn_mc_E_lists, results_mc["E_list"])
    end
    tn_mc_E_avg = mean(tn_mc_E_lists)
    tn_mc_E_stderr = [std([tn_mc_E_lists[t][s] for t in 1:n_traj]) for s in 1:n_steps+1] ./ sqrt(n_traj)

    println(@sprintf("\n  %-6s %-12s %-12s %-12s %-12s %-12s", "Step", "ED DM E/N", "ED MC E/N", "TN MC E/N", "|ED-TN MC|", "TN stderr"))
    println("  " * "-"^66)
    for step in 1:n_steps+1
        e_ed_dm = results_ed_dm["E_list"][step] / N
        e_ed_mc = ed_mc_E_avg[step] / N
        e_tn_mc = tn_mc_E_avg[step] / N
        diff = abs(e_ed_mc*N - tn_mc_E_avg[step]) / N
        stderr = tn_mc_E_stderr[step] / N
        println(@sprintf("  %-6d %-12.6f %-12.6f %-12.6f %-12.6f %-12.6f",
                step-1, e_ed_dm, e_ed_mc, e_tn_mc, diff, stderr))
    end
end

println("\n" * "="^70)
println("If TN MC+Continuous matches ED MC+Continuous, the evolution is correct.")
println("If not, TDVP has an issue (e.g., reverse_step=false causing bias).")
println("="^70)
