"""
    debug_cooling_physics.jl

Diagnose why cooling converges to E=0 (maximally mixed) instead of E=E_GS.
"""

using CoolingTNS
using LinearAlgebra
using SparseArrays
using Printf

# ============================================================================
# Test 1: Run 200 steps and track convergence
# ============================================================================
function test_long_cooling()
    println("="^60)
    println("Test 1: Long cooling run (200 steps)")
    println("="^60)

    N = 4
    ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)
    steps = 200
    cp = CoolingTNS.BasicCouplingParameters("XX", 0.2, steps, 1.0, nothing)

    backend = CoolingTNS.EDBackend()
    sim_method = CoolingTNS.DensityMatrix()
    evolution_method = CoolingTNS.ContinuousEvolution()
    sim_params = CoolingTNS.UnifiedSimulationParameters(sim_method, evolution_method)

    problem = CoolingTNS.setup_problem(backend, ham_params, cp, sim_params)
    state0 = CoolingTNS.setup_initial_state(problem, sim_params, "product", 0.0)

    E0 = problem.e₀
    println("Ground state energy: E0 = $E0, E0/N = $(E0/N)")
    println("Max mixed energy: 0 (since Tr(H)=0 for Ising)")
    println()

    results = CoolingTNS.run_cooling(problem, state0, cp, sim_params, ham_params)

    println("\nStep | E/N        | GS overlap | Purity")
    println("-"^55)
    for step in [1, 10, 20, 50, 100, 150, 200, 201]
        if step <= length(results["E_list"])
            E = results["E_list"][step]
            ov = results["GS_overlap_list"][step]
            p = results["purity_list"][step]
            @printf("%4d | %10.6f | %10.6f | %10.6f\n", step-1, E/N, ov, p)
        end
    end
    println("\n1/2^N = $(1/2^N) (max mixed purity)")
end

# ============================================================================
# Test 2: Verify CPTP map — apply map to identity, check if unital
# ============================================================================
function test_cptp_unitality()
    println("\n" * "="^60)
    println("Test 2: Is the CPTP map unital? (Φ(I/d) = I/d?)")
    println("="^60)

    N = 2
    ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)

    H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
    e0, _, gap = CoolingTNS.ground_state_ed(H_sys)

    cp = CoolingTNS.BasicCouplingParameters("XX", 0.2, 1, 1.0, nothing)
    N_total = 2 * N
    H_sb = CoolingTNS.construct_system_bath_hamiltonian(ham_params, CoolingTNS.EDBackend(), N_total, cp)

    println("N=$N, E0=$e0, gap=$gap")
    println("Δ (bath frequency) = $(cp.delta === nothing ? gap : cp.delta)")

    # Apply one step of the CPTP map to I/d
    d = 2^N
    ρ_max_mixed = CoolingTNS.EDDensityMatrix(Matrix{ComplexF64}(I, d, d) / d, N)

    # Prepare combined: I/d ⊗ |bath_GS><bath_GS|
    ρ_combined = CoolingTNS.prepare_combined_state_ed(ρ_max_mixed, N, "XX")

    # Evolve
    ρ_evolved = CoolingTNS.evolve_ed(H_sb, ρ_combined, 1.0)

    # Trace out bath
    ρ_sys_after = CoolingTNS.trace_out_bath_ed(ρ_evolved, N)

    # Check if result is I/d
    diff = norm(ρ_sys_after.data - ρ_max_mixed.data)
    E_after = real(tr(Matrix(H_sys) * ρ_sys_after.data))

    println("||Φ(I/d) - I/d|| = $diff")
    println("E[Φ(I/d)] = $E_after (should be 0 if unital)")
    println("Φ is unital: $(diff < 1e-10)")
end

# ============================================================================
# Test 3: Check single-step energy change for different initial states
# ============================================================================
function test_single_step_energy()
    println("\n" * "="^60)
    println("Test 3: Single-step energy change vs initial energy")
    println("="^60)

    N = 2
    ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)

    H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
    e0, ψ0, gap = CoolingTNS.ground_state_ed(H_sys)

    cp = CoolingTNS.BasicCouplingParameters("XX", 0.2, 1, 1.0, nothing)
    N_total = 2 * N
    H_sb = CoolingTNS.construct_system_bath_hamiltonian(ham_params, CoolingTNS.EDBackend(), N_total, cp)
    H_sys_mat = Matrix(H_sys)

    println("N=$N, E0=$(round(e0, digits=6)), gap=$(round(gap, digits=6))")

    # Get all eigenstates
    F = eigen(Symmetric(Matrix(H_sys)))
    println("\nSystem eigenvalues: ", round.(F.values, digits=6))

    println("\nInitial E/N → Final E/N → ΔE/N")
    println("-"^50)

    # Test with each energy eigenstate
    for (i, E_i) in enumerate(F.values)
        ψ_i = CoolingTNS.EDStateVector(ComplexF64.(F.vectors[:, i]), N)
        ρ_i = CoolingTNS.state_to_density_ed(ψ_i)

        # Apply one cooling step
        ρ_combined = CoolingTNS.prepare_combined_state_ed(ρ_i, N, "XX")
        ρ_evolved = CoolingTNS.evolve_ed(H_sb, ρ_combined, 1.0)
        ρ_sys_after = CoolingTNS.trace_out_bath_ed(ρ_evolved, N)

        E_after = real(tr(H_sys_mat * ρ_sys_after.data))
        ΔE = E_after - E_i
        @printf("  E_%d: %8.4f → %8.4f  ΔE = %+8.5f  (%s)\n",
                i, E_i/N, E_after/N, ΔE/N, ΔE < -1e-10 ? "COOLING" : (ΔE > 1e-10 ? "HEATING" : "unchanged"))
    end

    # Test with all-up state
    ψ_up = CoolingTNS.zero_state_ed(N)
    ρ_up = CoolingTNS.state_to_density_ed(ψ_up)
    E_init = real(tr(H_sys_mat * ρ_up.data))

    ρ_combined = CoolingTNS.prepare_combined_state_ed(ρ_up, N, "XX")
    ρ_evolved = CoolingTNS.evolve_ed(H_sb, ρ_combined, 1.0)
    ρ_sys_after = CoolingTNS.trace_out_bath_ed(ρ_evolved, N)
    E_after = real(tr(H_sys_mat * ρ_sys_after.data))
    ΔE = E_after - E_init
    @printf("  |↑↑⟩: %8.4f → %8.4f  ΔE = %+8.5f  (%s)\n",
            E_init/N, E_after/N, ΔE/N, ΔE < -1e-10 ? "COOLING" : (ΔE > 1e-10 ? "HEATING" : "unchanged"))

    # Test with maximally mixed state
    ρ_mm = CoolingTNS.EDDensityMatrix(Matrix{ComplexF64}(I, 2^N, 2^N) / 2^N, N)
    E_mm = real(tr(H_sys_mat * ρ_mm.data))

    ρ_combined = CoolingTNS.prepare_combined_state_ed(ρ_mm, N, "XX")
    ρ_evolved = CoolingTNS.evolve_ed(H_sb, ρ_combined, 1.0)
    ρ_sys_after = CoolingTNS.trace_out_bath_ed(ρ_evolved, N)
    E_after = real(tr(H_sys_mat * ρ_sys_after.data))
    ΔE = E_after - E_mm
    @printf("  I/d:  %8.4f → %8.4f  ΔE = %+8.5f  (%s)\n",
            E_mm/N, E_after/N, ΔE/N, ΔE < -1e-10 ? "COOLING" : (ΔE > 1e-10 ? "HEATING" : "unchanged"))
end

# ============================================================================
# Test 4: Eigenspectrum of the CPTP map (superoperator)
# ============================================================================
function test_cptp_spectrum()
    println("\n" * "="^60)
    println("Test 4: CPTP map eigenspectrum")
    println("="^60)

    N = 2
    ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)

    H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)

    cp = CoolingTNS.BasicCouplingParameters("XX", 0.2, 1, 1.0, nothing)
    N_total = 2 * N
    H_sb = CoolingTNS.construct_system_bath_hamiltonian(ham_params, CoolingTNS.EDBackend(), N_total, cp)

    d = 2^N  # System Hilbert space dimension

    # Build superoperator representation of the CPTP map
    # Φ(ρ) = Tr_bath(U (ρ ⊗ ρ_bath) U†)
    # In vec representation: vec(Φ(ρ)) = S · vec(ρ)
    # Use Hermitian basis elements (|a><b| + |b><a|)/2 and i(|a><b| - |b><a|)/2 for off-diagonal
    # But simpler: just compute the map on |a><b| directly using raw matrices
    S = zeros(ComplexF64, d^2, d^2)

    # Get bath ground state and evolution operator
    ψ_bath = CoolingTNS.get_bath_ground_state_ed(N, "XX")
    U = CoolingTNS.get_evolution_operator(H_sb, 1.0)

    for j in 1:d^2
        a = div(j-1, d) + 1
        b = mod(j-1, d) + 1
        ρ_basis = zeros(ComplexF64, d, d)
        ρ_basis[a, b] = 1.0

        # Manually apply the CPTP map without the Hermiticity check
        # 1. Tensor with bath: ρ_basis ⊗ |bath><bath|
        ρ_bath = ψ_bath.data * ψ_bath.data'
        N_total = 2 * N
        dim_total = 2^N_total
        ρ_combined = zeros(ComplexF64, dim_total, dim_total)
        for si in 0:(d-1), sj in 0:(d-1)
            for bi in 0:(d-1), bj in 0:(d-1)
                ci = 0; cj = 0
                for k in 0:(N-1)
                    ci |= ((si >> k) & 1) << (2*k)
                    ci |= ((bi >> k) & 1) << (2*k + 1)
                    cj |= ((sj >> k) & 1) << (2*k)
                    cj |= ((bj >> k) & 1) << (2*k + 1)
                end
                ρ_combined[ci+1, cj+1] = ρ_basis[si+1, sj+1] * ρ_bath[bi+1, bj+1]
            end
        end

        # 2. Evolve
        ρ_evolved_data = U * ρ_combined * U'

        # 3. Partial trace over bath
        ρ_out = zeros(ComplexF64, d, d)
        for si in 0:(d-1), sj in 0:(d-1)
            for b in 0:(d-1)
                ci = 0; cj = 0
                for k in 0:(N-1)
                    ci |= ((si >> k) & 1) << (2*k)
                    ci |= ((b >> k) & 1) << (2*k + 1)
                    cj |= ((sj >> k) & 1) << (2*k)
                    cj |= ((b >> k) & 1) << (2*k + 1)
                end
                ρ_out[si+1, sj+1] += ρ_evolved_data[ci+1, cj+1]
            end
        end

        S[:, j] = vec(ρ_out)
    end

    # Eigenvalues of the superoperator
    evals = eigvals(S)

    # Sort by magnitude
    idx = sortperm(abs.(evals), rev=true)
    evals_sorted = evals[idx]

    println("Top eigenvalues of CPTP superoperator (|λ| descending):")
    for (i, λ) in enumerate(evals_sorted)
        if abs(λ) > 1e-10
            @printf("  λ_%d = %+.6f %+.6fi  (|λ| = %.6f)\n", i, real(λ), imag(λ), abs(λ))
        end
    end

    # Find the fixed point (eigenvector for λ=1)
    F_super = eigen(S)
    unit_idx = argmin(abs.(F_super.values .- 1.0))
    ρ_fixed = reshape(F_super.vectors[:, unit_idx], d, d)
    ρ_fixed = ρ_fixed / tr(ρ_fixed)  # Normalize
    ρ_fixed = (ρ_fixed + ρ_fixed') / 2  # Enforce Hermiticity

    H_sys_mat = Matrix(H_sys)
    E_fixed = real(tr(H_sys_mat * ρ_fixed))
    purity_fixed = real(tr(ρ_fixed * ρ_fixed))

    e0, _, _ = CoolingTNS.ground_state_ed(H_sys)

    println("\nFixed point (λ=1 eigenvector):")
    println("  E_fixed/N = $(round(E_fixed/N, digits=6))")
    println("  Purity = $(round(purity_fixed, digits=6))")
    println("  E_GS/N = $(round(e0/N, digits=6))")
    println("  1/2^N = $(1/2^N)")

    # Check if fixed point is max mixed
    diff_mm = norm(ρ_fixed - Matrix{ComplexF64}(I, d, d)/d)
    println("  ||ρ_fixed - I/d|| = $(round(diff_mm, digits=6))")

    # Print the fixed point density matrix in energy basis
    F_H = eigen(Symmetric(Matrix(H_sys)))
    ρ_energy = F_H.vectors' * ρ_fixed * F_H.vectors
    println("\nFixed point in energy eigenbasis (diagonal):")
    for i in 1:d
        @printf("  p(E_%d = %+.4f) = %.6f\n", i, F_H.values[i], real(ρ_energy[i,i]))
    end
end

# ============================================================================
# Test 5: Scan over te to find optimal evolution time
# ============================================================================
function test_te_scan()
    println("\n" * "="^60)
    println("Test 5: Energy change vs evolution time te")
    println("="^60)

    N = 2
    ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)
    H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
    H_sys_mat = Matrix(H_sys)

    # Use all-up initial state
    ψ_up = CoolingTNS.zero_state_ed(N)
    ρ_up = CoolingTNS.state_to_density_ed(ψ_up)
    E_init = real(tr(H_sys_mat * ρ_up.data))

    println("Initial E/N = $(E_init/N)")
    println("\n   te    |  E_after/N  |  ΔE/N")
    println("-"^40)

    for te in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
        cp = CoolingTNS.BasicCouplingParameters("XX", 0.2, 1, te, nothing)
        N_total = 2 * N
        H_sb = CoolingTNS.construct_system_bath_hamiltonian(ham_params, CoolingTNS.EDBackend(), N_total, cp)

        ρ_combined = CoolingTNS.prepare_combined_state_ed(ρ_up, N, "XX")
        ρ_evolved = CoolingTNS.evolve_ed(H_sb, ρ_combined, te)
        ρ_after = CoolingTNS.trace_out_bath_ed(ρ_evolved, N)
        E_after = real(tr(H_sys_mat * ρ_after.data))

        @printf("  %5.1f  |  %+9.5f  |  %+9.5f\n", te, E_after/N, (E_after - E_init)/N)
    end
end

# Run all tests
test_long_cooling()
test_cptp_unitality()
test_single_step_energy()
test_cptp_spectrum()
test_te_scan()
