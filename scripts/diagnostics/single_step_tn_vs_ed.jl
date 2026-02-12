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

println("="^60)
println("Single Cooling Step: TN vs ED Diagnostic")
println("="^60)

N = 3
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
H_sb_ed = CoolingTNS.construct_system_bath_hamiltonian(ham_params, ed_backend, 2*N, coupling_params_ed)

# TN setup
tn_backend = CoolingTNS.TNBackend()
sites = siteinds("S=1/2", 2*N)
sites_sys = sites[1:2:2*N-1]
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

# TN: TDVP
for tdvp_tau in [0.1, 0.01]
    ψ_evolved_tn = tdvp(H_sb_tn, -im * te, ψ_combined_tn;
                        time_step=-im * tdvp_tau, nsite=2, reverse_step=false, normalize=true,
                        maxdim=200, cutoff=1e-14, outputlevel=0)
    normalize!(ψ_evolved_tn)

    E_evolved_tn = real(inner(ψ_evolved_tn', H_sb_tn, ψ_evolved_tn))
    println("  TN evolved energy (tau=$tdvp_tau) = $E_evolved_tn (should ≈ $E_combined_ed)")
    println("    |E_TN - E_ED| = $(abs(E_evolved_tn - E_evolved_ed))")

    # Compare state overlap: convert TN to ED vector
    dim = 2^(2*N)
    ψ_tn_vec = zeros(ComplexF64, dim)
    for idx in 0:dim-1
        config = [((idx >> (k-1)) & 1) == 0 ? "Up" : "Dn" for k in 1:2*N]
        ψ_basis = MPS(sites, config)
        ψ_tn_vec[idx+1] = inner(ψ_basis, ψ_evolved_tn)
    end
    ψ_tn_vec /= norm(ψ_tn_vec)
    ψ_ed_vec = ψ_evolved_ed.data / norm(ψ_evolved_ed.data)

    overlap = abs(dot(ψ_ed_vec, ψ_tn_vec))
    println("    State overlap |<ED|TN>| = $overlap")
end

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

# For TN: use partial trace on MPO
ψ_evolved_tn_final = tdvp(H_sb_tn, -im * te, ψ_combined_tn;
                          time_step=-im * 0.01, nsite=2, reverse_step=false, normalize=true,
                          maxdim=200, cutoff=1e-14, outputlevel=0)
normalize!(ψ_evolved_tn_final)

# Convert TN evolved to density matrix and trace bath
dim = 2^(2*N)
ψ_tn_vec_final = zeros(ComplexF64, dim)
for idx in 0:dim-1
    config = [((idx >> (k-1)) & 1) == 0 ? "Up" : "Dn" for k in 1:2*N]
    ψ_basis = MPS(sites, config)
    ψ_tn_vec_final[idx+1] = inner(ψ_basis, ψ_evolved_tn_final)
end
ψ_tn_vec_final /= norm(ψ_tn_vec_final)

ρ_tn_full = ψ_tn_vec_final * ψ_tn_vec_final'
# Trace out bath (keep system qubits at positions 1,3,5 = bit positions 0,2,4)
ρ_sys_tn = zeros(ComplexF64, 2^N, 2^N)
for si in 0:2^N-1, sj in 0:2^N-1
    for bath in 0:2^N-1
        fi = CoolingTNS.map_system_bath_to_full_basis_ed(si, bath, N)
        fj = CoolingTNS.map_system_bath_to_full_basis_ed(sj, bath, N)
        ρ_sys_tn[si+1, sj+1] += ρ_tn_full[fi+1, fj+1]
    end
end
E_sys_tn = real(tr(H_sys_ed * ρ_sys_tn))
println("  TN system energy after trace = $E_sys_tn")
println("  TN system energy / N = $(E_sys_tn / N)")
println("  |E_sys_TN - E_sys_ED| = $(abs(E_sys_tn - E_sys_ed))")

println("\n  If these match, TDVP evolution is correct.")
println("  If not, there's a fundamental difference in the evolution.")

println("\n" * "="^60)
