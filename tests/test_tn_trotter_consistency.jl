"""
    test_tn_trotter_consistency.jl

Debug script to verify that MPS (MC+Trotter) and MPO (DM+Trotter) use
the same Trotter decomposition and produce consistent results.

The key issue: MPS uses build_trotter_circuit_bath_coupling + TDVP for system terms,
while MPO uses build_trotter_circuit_interleaved with all terms in the gates.
"""

using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

println("="^70)
println("TN Trotter Consistency Debug")
println("="^70)

# ============================================================================
# Test parameters
# ============================================================================
N = 3
ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)
g = 0.2
te = 1.0
n_steps = 5
tau = 0.05

coupling_params = CoolingTNS.BasicCouplingParameters("XX", g, n_steps, te, nothing)

# ============================================================================
# Test 1: Compare MPS vs MPO after a SINGLE cooling step
# ============================================================================
println("\n--- Test 1: Single cooling step comparison ---")
println("N=$N, Ising, g=$g, te=$te, tau=$tau")

# Run MPO (DM+Trotter) - deterministic reference
results_mpo, prob_mpo = let
    backend = CoolingTNS.TNBackend()
    sim_method = CoolingTNS.DensityMatrix()
    evolution_method = CoolingTNS.TrotterEvolution()
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        sim_method, evolution_method;
        Dmax=200, cutoff=1e-12, tau=tau, pe=0.0, n_trajectories=1
    )
    problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
    state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
    results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)
    results, problem
end

# Run MPS (MC+Trotter) - multiple trajectories for averaging
n_traj = 100
mc_E_lists = Vector{Vector{Float64}}()
for i in 1:n_traj
    results_mc, _ = let
        backend = CoolingTNS.TNBackend()
        sim_method = CoolingTNS.MonteCarloWavefunction()
        evolution_method = CoolingTNS.TrotterEvolution()
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            sim_method, evolution_method;
            Dmax=200, cutoff=1e-12, tau=tau, pe=0.0, n_trajectories=1
        )
        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
        state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
        results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)
        results, problem
    end
    push!(mc_E_lists, results_mc["E_list"])
end

# Compare step by step
using Statistics
mc_E_avg = mean(mc_E_lists)
mc_E_std = [std([mc_E_lists[t][s] for t in 1:n_traj]) for s in 1:n_steps+1]
mc_E_stderr = mc_E_std ./ sqrt(n_traj)

println("\nStep-by-step comparison (MPO vs MC avg, $n_traj trajectories):")
println(@sprintf("%-6s %-12s %-12s %-12s %-12s %-8s", "Step", "MPO E/N", "MC E/N", "diff/N", "MC stderr", "sigmas"))
println("-"^70)
for step in 1:n_steps+1
    E_mpo = results_mpo["E_list"][step] / N
    E_mc = mc_E_avg[step] / N
    diff = abs(E_mpo - E_mc) / N
    stderr = mc_E_stderr[step] / N
    sigmas = stderr > 0 ? abs(E_mpo*N - mc_E_avg[step]) / mc_E_stderr[step] : 0.0
    println(@sprintf("%-6d %-12.6f %-12.6f %-12.6f %-12.6f %-8.1f",
            step-1, E_mpo, E_mc, diff, stderr, sigmas))
end

# ============================================================================
# Test 2: Trotter convergence - do both converge to ED as tau→0?
# ============================================================================
println("\n--- Test 2: Trotter convergence for different tau ---")

# ED reference
results_ed, prob_ed = let
    backend = CoolingTNS.EDBackend()
    sim_method = CoolingTNS.DensityMatrix()
    evolution_method = CoolingTNS.ContinuousEvolution()
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        sim_method, evolution_method;
        Dmax=50, cutoff=1e-10, tau=0.1, pe=0.0, n_trajectories=1
    )
    problem = CoolingTNS.setup_problem(backend, ham_params, coupling_params, sim_params)
    state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
    results = CoolingTNS.run_cooling(problem, state0, coupling_params, sim_params, ham_params)
    results, problem
end

E_ed_final = results_ed["E_list"][end] / N
println("\nED DM+Continuous final E/N = $E_ed_final (exact reference)")

println(@sprintf("\n%-8s %-14s %-14s %-14s %-14s", "tau", "MPO E/N", "|MPO-ED|/N", "MC avg E/N", "|MC-ED|/N"))
println("-"^70)

for tau_test in [0.5, 0.2, 0.1, 0.05, 0.02]
    # MPO
    coupling_test = CoolingTNS.BasicCouplingParameters("XX", g, n_steps, te, nothing)
    results_mpo_t, _ = let
        backend = CoolingTNS.TNBackend()
        sim_method = CoolingTNS.DensityMatrix()
        evolution_method = CoolingTNS.TrotterEvolution()
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            sim_method, evolution_method;
            Dmax=200, cutoff=1e-12, tau=tau_test, pe=0.0, n_trajectories=1
        )
        problem = CoolingTNS.setup_problem(backend, ham_params, coupling_test, sim_params)
        state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
        results = CoolingTNS.run_cooling(problem, state0, coupling_test, sim_params, ham_params)
        results, problem
    end

    # MC (fewer trajectories for speed)
    mc_final_Es = Float64[]
    for _ in 1:30
        results_mc_t, _ = let
            backend = CoolingTNS.TNBackend()
            sim_method = CoolingTNS.MonteCarloWavefunction()
            evolution_method = CoolingTNS.TrotterEvolution()
            sim_params = CoolingTNS.UnifiedSimulationParameters(
                sim_method, evolution_method;
                Dmax=200, cutoff=1e-12, tau=tau_test, pe=0.0, n_trajectories=1
            )
            problem = CoolingTNS.setup_problem(backend, ham_params, coupling_test, sim_params)
            state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)
            results = CoolingTNS.run_cooling(problem, state0, coupling_test, sim_params, ham_params)
            results, problem
        end
        push!(mc_final_Es, results_mc_t["E_list"][end])
    end

    E_mpo = results_mpo_t["E_list"][end] / N
    E_mc = mean(mc_final_Es) / N
    diff_mpo = abs(E_mpo - E_ed_final)
    diff_mc = abs(E_mc - E_ed_final)
    println(@sprintf("%-8.3f %-14.6f %-14.6f %-14.6f %-14.6f", tau_test, E_mpo, diff_mpo, E_mc, diff_mc))
end

println("\n" * "="^70)
println("If MC and MPO disagree at small tau, it confirms different Trotter circuits.")
println("Both should converge to ED as tau→0, but at different rates.")
println("="^70)
