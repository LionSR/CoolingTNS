"""
    test_single_step_tn_vs_ed.jl

Compare a single cooling step between TN MC+Continuous and ED,
checking initial state, evolved state, and post-measurement state.
"""

using CoolingTNS
using ITensors
using ITensorMPS
using LinearAlgebra
using Printf

function mps_to_ed_vector(ψ::MPS, sites)
    nsites = length(sites)
    dim = 2^nsites
    ψ_vec = zeros(ComplexF64, dim)
    for idx in 0:(dim - 1)
        config = [((idx >> (site - 1)) & 1) == 0 ? "Up" : "Dn" for site in 1:nsites]
        ψ_basis = MPS(sites, config)
        ψ_vec[idx + 1] = inner(ψ_basis, ψ)
    end
    return ψ_vec
end

function trace_bath_from_interleaved_vector(ψ_vec::AbstractVector, N::Int)
    ρ_full = ψ_vec * ψ_vec'
    ρ_sys = zeros(ComplexF64, 2^N, 2^N)
    for si in 0:(2^N - 1), sj in 0:(2^N - 1)
        for bath in 0:(2^N - 1)
            fi = CoolingTNS.map_system_bath_to_full_basis_ed(si, bath, N)
            fj = CoolingTNS.map_system_bath_to_full_basis_ed(sj, bath, N)
            ρ_sys[si + 1, sj + 1] += ρ_full[fi + 1, fj + 1]
        end
    end
    return ρ_sys
end

println("="^60)
println("Single Cooling Step: TN vs ED Diagnostic")
println("="^60)

N = 3
N_total = CoolingTNS.interleaved_total_sites(N)
ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)
g = 0.2
te = 1.0
coupling_str = "XX"

# ============================================================================
# Setup both backends
# ============================================================================

# ED setup
ed_backend = CoolingTNS.EDBackend()
H_sys_ed = CoolingTNS.construct_system_hamiltonian(ham_params, ed_backend, N)
e0_ed, ψ0_ed, gap_ed = CoolingTNS.ground_state_ed(H_sys_ed)
coupling_params_ed = CoolingTNS.BasicCouplingParameters(coupling_str, g, 5, te, gap_ed)
H_sb_ed = CoolingTNS.construct_system_bath_hamiltonian(ham_params, ed_backend, N_total, coupling_params_ed)

# TN setup
tn_backend = CoolingTNS.TNBackend()
sites = siteinds("S=1/2", N_total)
sites_sys = CoolingTNS.interleaved_system_indices(sites, N)
H_sys_tn = CoolingTNS.construct_system_hamiltonian(ham_params, tn_backend, sites_sys)
e0_tn, ψ0_tn, gap_tn = CoolingTNS.find_ground_state(H_sys_tn, tn_backend, sites_sys)
coupling_params_tn = CoolingTNS.BasicCouplingParameters(coupling_str, g, 5, te, gap_tn)
H_sb_tn = CoolingTNS.construct_system_bath_hamiltonian(ham_params, tn_backend, sites, coupling_params_tn)

println("\nSetup comparison:")
println("  ED gap = $gap_ed, TN gap = $gap_tn, diff = $(abs(gap_ed - gap_tn))")
println("  ED e0 = $e0_ed, TN e0 = $e0_tn")

# ============================================================================
# Step 1: Compare initial combined states
# ============================================================================
println("\n--- Step 1: Initial combined states ---")

# ED: system |↑↑↑⟩, bath |↓↓↓⟩
ψ_sys_ed = CoolingTNS.zero_state_ed(N)  # |↑↑↑⟩ = |000⟩
ψ_combined_ed = CoolingTNS.prepare_combined_state_ed(ψ_sys_ed, N, coupling_str)

println("  ED combined state: $(ψ_combined_ed.n_qubits) qubits, norm = $(norm(ψ_combined_ed.data))")
println("  ED nonzero entries: $(count(x -> abs(x) > 1e-10, ψ_combined_ed.data))")

# TN: system |↑↑↑⟩, bath appended
ψ_sys_tn = MPS(sites_sys, "Up")  # |↑↑↑⟩
ψ_combined_tn = CoolingTNS.appendzeros_MPS(ψ_sys_tn, sites, coupling_str)

println("  TN combined MPS: $(length(ψ_combined_tn)) sites, norm = $(norm(ψ_combined_tn))")

# Compare energies of combined states
E_combined_ed = real(ψ_combined_ed.data' * H_sb_ed * ψ_combined_ed.data)
E_combined_tn = real(inner(ψ_combined_tn', H_sb_tn, ψ_combined_tn))
println("  ED combined energy = $E_combined_ed")
println("  TN combined energy = $E_combined_tn")
println("  Diff = $(abs(E_combined_ed - E_combined_tn))")

# ============================================================================
# Step 2: Compare evolved states
# ============================================================================
println("\n--- Step 2: Evolved states ---")

# ED: exact exp(-iHt)
ψ_evolved_ed = CoolingTNS.evolve_cooling_step_ed(H_sb_ed, ψ_combined_ed, te, nothing)

println("  ED evolved norm = $(norm(ψ_evolved_ed.data))")
E_evolved_ed = real(ψ_evolved_ed.data' * H_sb_ed * ψ_evolved_ed.data) / (norm(ψ_evolved_ed.data)^2)
println("  ED evolved energy = $E_evolved_ed (should = initial = $E_combined_ed)")

# TN: package continuous-evolution wrapper, which owns the TDVP convention.
ψ_ed_vec = ψ_evolved_ed.data / norm(ψ_evolved_ed.data)
tn_results = map([0.1, 0.01]) do tdvp_tau
    sim_params_tn = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.MonteCarloWavefunction(),
        CoolingTNS.ContinuousEvolution();
        Dmax=200,
        cutoff=1e-14,
        tau=tdvp_tau,
    )
    ψ_evolved_tn = CoolingTNS.evolve_state(
        ham_params, sim_params_tn, tn_backend, H_sb_tn, ψ_combined_tn, te, sites
    )

    E_evolved_tn = real(inner(ψ_evolved_tn', H_sb_tn, ψ_evolved_tn))
    println("  TN evolved energy (tau=$tdvp_tau) = $E_evolved_tn (should ≈ $E_combined_ed)")
    println("    |E_TN - E_ED| = $(abs(E_evolved_tn - E_evolved_ed))")

    ψ_tn_vec = mps_to_ed_vector(ψ_evolved_tn, sites)
    ψ_tn_vec /= norm(ψ_tn_vec)
    overlap = abs(dot(ψ_ed_vec, ψ_tn_vec))
    println("    State overlap |<ED|TN>| = $overlap")

    return (tau=tdvp_tau, state_vector=ψ_tn_vec, overlap=overlap)
end
final_result = last(tn_results)
ψ_tn_vec_final = final_result.state_vector
final_overlap = final_result.overlap

# ============================================================================
# Step 3: Check system density matrix after partial trace (no measurement)
# ============================================================================
println("\n--- Step 3: System density matrix after tracing bath ---")

# For ED DM: ρ_sys = Tr_bath(|ψ_evolved⟩⟨ψ_evolved|)
ρ_evolved_ed = CoolingTNS.state_to_density_ed(ψ_evolved_ed)
ρ_sys_ed = CoolingTNS.trace_out_bath_ed(ρ_evolved_ed, N)
E_sys_ed = CoolingTNS.expect_ed(H_sys_ed, ρ_sys_ed)
println("  ED system energy after trace = $E_sys_ed")
println("  ED system energy / N = $(E_sys_ed / N)")

# Convert TN evolved to density matrix and trace bath
ρ_sys_tn = trace_bath_from_interleaved_vector(ψ_tn_vec_final, N)
E_sys_tn = real(tr(H_sys_ed * ρ_sys_tn))
println("  TN system energy after trace = $E_sys_tn")
println("  TN system energy / N = $(E_sys_tn / N)")
energy_trace_error = abs(E_sys_tn - E_sys_ed)
println("  |E_sys_TN - E_sys_ED| = $energy_trace_error")

if final_overlap < 1 - 1e-8
    error("TN package evolution does not match ED: final overlap = $final_overlap")
end
if energy_trace_error > 1e-6
    error("Reduced system energy mismatch after tracing bath: $energy_trace_error")
end

println("\n  TN package evolution matches ED for this single-step diagnostic.")

println("\n" * "="^60)
