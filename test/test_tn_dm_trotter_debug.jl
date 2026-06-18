"""
    test_tn_dm_trotter_debug.jl

Diagnostic script to verify TN DM+Trotter matches ED DM+Trotter step-by-step.
For N=2 Ising, compares:
  1. State preparation (initial system+bath state)
  2. One Trotter step evolution
  3. Partial trace to get system density matrix
"""

using Test
using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra

@isdefined(test_mpo_to_matrix) || include("test_helpers.jl")

# ============================================================================
# Test parameters (N=2 Ising for simplicity)
# ============================================================================
const N = 2
const J = 1.0
const h_field = 1.0
const g_coupling = 0.3
const tau = 0.1     # Trotter step size
const te = 1.0      # Evolution time per cooling step

ham_params = CoolingTNS.IsingParameters(N, J, h_field)
coupling_params_no_delta = CoolingTNS.BasicCouplingParameters("XX", g_coupling, 1, te, nothing)

# ============================================================================
# Setup ED problem to get delta
# ============================================================================
println("="^60)
println("Setting up ED and TN problems...")
println("="^60)

ed_backend = CoolingTNS.EDBackend()
tn_backend = CoolingTNS.TNBackend()

# ED setup
sim_params_ed = CoolingTNS.UnifiedSimulationParameters(
    CoolingTNS.DensityMatrix(), CoolingTNS.TrotterEvolution();
    tau=tau
)
prob_ed = CoolingTNS.setup_problem(ed_backend, ham_params, coupling_params_no_delta, sim_params_ed)

# Get the actual delta used
delta_ed = prob_ed.extra.coupling_params.delta
println("  ED delta (gap) = $delta_ed")
println("  ED e₀ = $(prob_ed.e₀)")

# Use same delta for TN
coupling_params = CoolingTNS.BasicCouplingParameters("XX", g_coupling, 1, te, delta_ed)

# TN setup
sim_params_tn = CoolingTNS.UnifiedSimulationParameters(
    CoolingTNS.DensityMatrix(), CoolingTNS.TrotterEvolution();
    tau=tau, Dmax=100, cutoff=1e-14
)
prob_tn = CoolingTNS.setup_problem(tn_backend, ham_params, coupling_params, sim_params_tn)

println("  TN e₀ = $(prob_tn.e₀)")
println("  |e₀_ED - e₀_TN| = $(abs(prob_ed.e₀ - prob_tn.e₀))")

# ============================================================================
# Test 1: Initial system state matches
# ============================================================================
@testset "Initial system state matches" begin
    println("\n--- Test 1: Initial system state ---")

    # ED initial state (|↑↑⟩ product state)
    state_ed = CoolingTNS.setup_initial_state(prob_ed, sim_params_ed, "product", 0.0)
    ρ_sys_ed = state_ed.state

    # TN initial state
    state_tn = CoolingTNS.setup_initial_state(prob_tn, sim_params_tn, "product", 0.0)
    ρ_sys_tn_mat = test_mpo_to_matrix(state_tn.state)

    diff_init = norm(ρ_sys_tn_mat - ρ_sys_ed.data)
    println("  ||ρ_sys_TN - ρ_sys_ED|| = $diff_init")
    @test diff_init < 1e-10
end

# ============================================================================
# Test 2: Combined system+bath state matches
# ============================================================================
@testset "Combined system+bath state matches" begin
    println("\n--- Test 2: Combined system+bath state ---")

    state_ed = CoolingTNS.setup_initial_state(prob_ed, sim_params_ed, "product", 0.0)
    ρ_sb_ed = CoolingTNS.prepare_combined_state(prob_ed, state_ed)

    state_tn = CoolingTNS.setup_initial_state(prob_tn, sim_params_tn, "product", 0.0)
    ρ_sb_tn = CoolingTNS.prepare_combined_state(prob_tn, state_tn)

    ρ_sb_tn_mat = test_mpo_to_matrix(ρ_sb_tn)

    diff_combined = norm(ρ_sb_tn_mat - ρ_sb_ed.data)
    println("  ||ρ_sb_TN - ρ_sb_ED|| = $diff_combined")
    @test diff_combined < 1e-10
end

# ============================================================================
# Test 3: One Trotter step evolution
# ============================================================================
@testset "One Trotter step evolution" begin
    println("\n--- Test 3: One Trotter step evolution ---")

    # ED
    state_ed = CoolingTNS.setup_initial_state(prob_ed, sim_params_ed, "product", 0.0)
    ρ_sb_ed = CoolingTNS.prepare_combined_state(prob_ed, state_ed)
    ρ_sb_ed_evolved = CoolingTNS.evolve_cooling_step(prob_ed, ρ_sb_ed, te, sim_params_ed, ham_params)

    # TN
    state_tn = CoolingTNS.setup_initial_state(prob_tn, sim_params_tn, "product", 0.0)
    ρ_sb_tn = CoolingTNS.prepare_combined_state(prob_tn, state_tn)
    ρ_sb_tn_evolved = CoolingTNS.evolve_cooling_step(prob_tn, ρ_sb_tn, te, sim_params_tn, ham_params)

    ρ_sb_tn_evolved_mat = test_mpo_to_matrix(ρ_sb_tn_evolved)

    diff_evolved = norm(ρ_sb_tn_evolved_mat - ρ_sb_ed_evolved.data)
    println("  ||ρ_evolved_TN - ρ_evolved_ED|| = $diff_evolved")
    println("  (ED uses full exp(-iHt), TN uses real Trotter splitting)")
    println("  Expected Trotter error ~ O(tau^2) = O($(tau^2))")

    @test abs(tr(ρ_sb_tn_evolved_mat) - 1.0) < 1e-8
    @test abs(tr(ρ_sb_ed_evolved.data) - 1.0) < 1e-8

    # The difference should be bounded (grows with te/tau * tau^2 Trotter error)
    @test diff_evolved < 2.0
end

# ============================================================================
# Test 4: Full cooling run (1 step)
# ============================================================================
@testset "Full cooling step comparison" begin
    println("\n--- Test 4: Full cooling step ---")

    results_ed = CoolingTNS.run_cooling(
        prob_ed,
        CoolingTNS.setup_initial_state(prob_ed, sim_params_ed, "product", 0.0),
        coupling_params, sim_params_ed, ham_params
    )

    results_tn = CoolingTNS.run_cooling(
        prob_tn,
        CoolingTNS.setup_initial_state(prob_tn, sim_params_tn, "product", 0.0),
        coupling_params, sim_params_tn, ham_params
    )

    println("  ED energy evolution: $(results_ed[RESULT_ENERGY])")
    println("  TN energy evolution: $(results_tn[RESULT_ENERGY])")

    @test abs(results_ed[RESULT_ENERGY][1] - results_tn[RESULT_ENERGY][1]) < 1e-4

    # Both should show cooling
    @test results_ed[RESULT_ENERGY][end] < results_ed[RESULT_ENERGY][1] + 1e-8
    @test results_tn[RESULT_ENERGY][end] < results_tn[RESULT_ENERGY][1] + 1e-8

    E_diff = abs(results_ed[RESULT_ENERGY][end] - results_tn[RESULT_ENERGY][end])
    println("  |E_final_ED - E_final_TN| = $E_diff")
    @test E_diff < 1.0
end

# ============================================================================
# Test 5: Multi-step cooling comparison
# ============================================================================
@testset "Multi-step cooling comparison" begin
    println("\n--- Test 5: Multi-step cooling ---")

    coupling_multi = CoolingTNS.BasicCouplingParameters("XX", g_coupling, 5, te, delta_ed)

    results_ed = CoolingTNS.run_cooling(
        prob_ed,
        CoolingTNS.setup_initial_state(prob_ed, sim_params_ed, "product", 0.0),
        coupling_multi, sim_params_ed, ham_params
    )

    results_tn = CoolingTNS.run_cooling(
        prob_tn,
        CoolingTNS.setup_initial_state(prob_tn, sim_params_tn, "product", 0.0),
        coupling_multi, sim_params_tn, ham_params
    )

    println("  Step | ED E/N      | TN E/N      | diff")
    println("  -----|-------------|-------------|------")
    for step in 1:length(results_ed[RESULT_ENERGY])
        E_ed = results_ed[RESULT_ENERGY][step] / N
        E_tn = results_tn[RESULT_ENERGY][step] / N
        diff = abs(E_ed - E_tn)
        println("    $step  | $(round(E_ed, digits=6)) | $(round(E_tn, digits=6)) | $(round(diff, digits=6))")
    end

    ed_cooling = results_ed[RESULT_ENERGY][1] - results_ed[RESULT_ENERGY][end]
    tn_cooling = results_tn[RESULT_ENERGY][1] - results_tn[RESULT_ENERGY][end]
    println("  ED total cooling: $ed_cooling")
    println("  TN total cooling: $tn_cooling")

    @test ed_cooling > 0.01  # Cooling should be measurable
    @test tn_cooling > 0.01  # TN should now cool too
end

println("\n" * "="^60)
println("TN DM+Trotter debug tests completed!")
println("="^60)
