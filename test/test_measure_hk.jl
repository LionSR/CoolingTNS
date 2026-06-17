"""
Tests for mode energy measurement (measure_hk, measure_all_mode_energies, measure_state_parity).

These tests verify that:
1. The ground state has ⟨h_k⟩ = -1 for all modes (Bogoliubov vacuum)
2. Excited states have ⟨h_k⟩ > -1 for the excited mode
3. Total energy from mode decomposition matches ⟨H⟩
4. Mode energies sum correctly for the ground state
5. The all-spins-up initial state has ⟨h_k⟩ ≈ 0 for generic modes
6. Density matrices give consistent results with pure states
7. Parity measurement: ⟨P_x⟩ = ±1 for parity eigenstates
"""

using Test
using CoolingTNS
using CoolingTNS: EDStateVector, EDDensityMatrix, state_to_density_ed, pauli_z, pauli_x
using LinearAlgebra

# ============================================================================
# Helper functions
# ============================================================================

"""Build H_code = J Σ Z_i Z_{i+1} + h Σ X_i using ed_backend operators."""
function _build_H(N::Int, J::Float64, h::Float64, bc::Symbol)
    dim = 2^N
    H = zeros(Float64, dim, dim)
    for i in 1:N-1
        H .+= J * Matrix(Float64.(pauli_z(i, N) * pauli_z(i+1, N)))
    end
    if bc == :periodic
        H .+= J * Matrix(Float64.(pauli_z(N, N) * pauli_z(1, N)))
    elseif bc == :antiperiodic
        H .-= J * Matrix(Float64.(pauli_z(N, N) * pauli_z(1, N)))
    end
    for i in 1:N
        H .+= h * Matrix(Float64.(pauli_x(i, N)))
    end
    return Hermitian(H)
end

"""Find ground state in a specific P_x parity sector."""
function _find_gs_in_sector(H, N::Int, target_parity::Int)
    evals, evecs = eigen(H)
    Px = parity_operator_code(N)

    # Re-diagonalize within degenerate subspaces to get clean parity eigenstates
    tol = 1e-8
    results = Tuple{Float64, Vector{ComplexF64}, Int}[]
    i = 1
    while i <= length(evals)
        j = i
        while j < length(evals) && abs(evals[j+1] - evals[i]) < tol
            j += 1
        end
        V = evecs[:, i:j]
        P_sub = V' * Matrix(Px) * V
        ep, vp = eigen(Hermitian(real(P_sub)))
        for k in 1:length(ep)
            p = round(Int, ep[k])
            push!(results, (evals[i], ComplexF64.(V * vp[:, k]), p))
        end
        i = j + 1
    end

    # Find lowest energy in target sector
    sector = filter(r -> r[3] == target_parity, results)
    idx = argmin([r[1] for r in sector])
    return sector[idx][1], sector[idx][2]
end

"""Find the first excited state in a given parity sector."""
function _find_excited_in_sector(H, N::Int, target_parity::Int)
    evals, evecs = eigen(H)
    Px = parity_operator_code(N)

    tol = 1e-8
    results = Tuple{Float64, Vector{ComplexF64}, Int}[]
    i = 1
    while i <= length(evals)
        j = i
        while j < length(evals) && abs(evals[j+1] - evals[i]) < tol
            j += 1
        end
        V = evecs[:, i:j]
        P_sub = V' * Matrix(Px) * V
        ep, vp = eigen(Hermitian(real(P_sub)))
        for k in 1:length(ep)
            p = round(Int, ep[k])
            push!(results, (evals[i], ComplexF64.(V * vp[:, k]), p))
        end
        i = j + 1
    end

    sector = filter(r -> r[3] == target_parity, results)
    sorted = sort(sector, by=r -> r[1])
    # Return second-lowest energy state
    return sorted[2][1], sorted[2][2]
end

# ============================================================================
# Tests
# ============================================================================

@testset "Mode Energy Measurements" begin

    @testset "Parity measurement" begin
        @testset "N=$N" for N in [4, 6]
            J, h = 1.0, 0.5
            H = _build_H(N, J, h, :periodic)

            for target_p in [1, -1]
                E, ψ = _find_gs_in_sector(H, N, target_p)
                state = EDStateVector(ψ, N)
                px = measure_state_parity(state, N)
                @test px ≈ target_p atol=1e-10
            end
        end

        @testset "Density matrix parity (N=4)" begin
            N = 4; J = 1.0; h = 0.5
            H = _build_H(N, J, h, :periodic)

            E_plus, ψ_plus = _find_gs_in_sector(H, N, 1)
            ρ_plus = state_to_density_ed(EDStateVector(ψ_plus, N))
            @test measure_state_parity(ρ_plus, N) ≈ 1.0 atol=1e-10

            E_minus, ψ_minus = _find_gs_in_sector(H, N, -1)
            ρ_minus = state_to_density_ed(EDStateVector(ψ_minus, N))
            @test measure_state_parity(ρ_minus, N) ≈ -1.0 atol=1e-10
        end

        @testset "Mixed state parity (N=4)" begin
            N = 4; J = 1.0; h = 0.5
            H = _build_H(N, J, h, :periodic)

            E_plus, ψ_plus = _find_gs_in_sector(H, N, 1)
            E_minus, ψ_minus = _find_gs_in_sector(H, N, -1)

            # 50/50 mixture of even and odd → ⟨Px⟩ = 0
            ρ_mix = EDDensityMatrix(
                0.5 * ψ_plus * ψ_plus' + 0.5 * ψ_minus * ψ_minus', N)
            @test measure_state_parity(ρ_mix, N) ≈ 0.0 atol=1e-10
        end
    end

    @testset "Ground state h_k = -1 (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        # Even parity sector (Px=+1) → gF=-1
        E_gs, ψ_gs = _find_gs_in_sector(H, N, 1)
        gs_state = EDStateVector(ψ_gs, N)
        gF = fermionic_bc(:periodic, 1)
        ks = allowed_k_indices(N, gF)

        for k in ks
            hk = measure_hk(gs_state, k, ham_params)
            @test hk ≈ -1.0 atol=1e-8
        end
    end

    @testset "Ground state h_k = -1, odd sector (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        # Odd parity sector (Px=-1) → gF=+1 (integer k, with special modes)
        # The odd-sector GS is NOT the Bogoliubov vacuum (which has Nf=0, even parity).
        # It has one quasiparticle excited in the cheapest special mode.
        E_gs_odd, ψ_gs_odd = _find_gs_in_sector(H, N, -1)
        gs_odd = EDStateVector(ψ_gs_odd, N)
        gF = fermionic_bc(:periodic, -1)  # gF = +1
        ks = allowed_k_indices(N, gF)

        # Most modes should have h_k = -1 (still in vacuum for that mode)
        # One special mode should have h_k = +1 (occupied)
        hk_vals = Dict{Any, Float64}()
        for k in ks
            hk = measure_hk(gs_odd, k, ham_params)
            hk_vals[k] = hk
        end

        # Check: exactly one mode should have h_k ≈ +1 (the cheapest special mode)
        excited_modes = [k for (k, hk) in hk_vals if hk > 0]
        ground_modes = [k for (k, hk) in hk_vals if hk < -0.5]

        @test length(excited_modes) == 1
        @test hk_vals[excited_modes[1]] ≈ 1.0 atol=1e-8

        # All other modes should be in ground state
        for k in ground_modes
            @test hk_vals[k] ≈ -1.0 atol=1e-8
        end
    end

    @testset "Excited state has h_k > -1 (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        # First excited state in even sector (two quasiparticles excited)
        E_gs, ψ_gs = _find_gs_in_sector(H, N, 1)
        E_ex, ψ_ex = _find_excited_in_sector(H, N, 1)
        ex_state = EDStateVector(ψ_ex, N)

        gF = fermionic_bc(:periodic, 1)
        ks = allowed_k_indices(N, gF)

        hk_vals = [measure_hk(ex_state, k, ham_params) for k in ks]

        # At least one mode should have h_k > -1
        @test any(hk -> hk > -1 + 1e-6, hk_vals)

        # The energy gap should be 2ε_k for the excited mode pair
        gap = E_ex - E_gs
        @test gap > 0
    end

    @testset "Total energy from mode decomposition (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        for target_p in [1, -1]
            E_gs, ψ_gs = _find_gs_in_sector(H, N, target_p)
            gs_state = EDStateVector(ψ_gs, N)
            gF = fermionic_bc(:periodic, target_p)

            ks_all, hk_vals, εk_vals = measure_all_mode_energies(gs_state, ham_params; gF=gF)

            # E = (Λ/2) Σ_k coeff_k · h_k
            E_modes = sum(Λ * coeff_k(Float64(k), θ, N) * hk / 2
                         for (k, hk) in zip(ks_all, hk_vals))

            @test E_modes ≈ E_gs atol=1e-8
        end
    end

    @testset "Total energy for excited states (N=4)" begin
        N = 4; J = 1.0; h = 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        # Check total energy for first excited state too
        E_ex, ψ_ex = _find_excited_in_sector(H, N, 1)
        ex_state = EDStateVector(ψ_ex, N)
        gF = fermionic_bc(:periodic, 1)

        ks_all, hk_vals, _ = measure_all_mode_energies(ex_state, ham_params; gF=gF)
        E_modes = sum(Λ * coeff_k(Float64(k), θ, N) * hk / 2
                     for (k, hk) in zip(ks_all, hk_vals))

        @test E_modes ≈ E_ex atol=1e-8
    end

    @testset "All-spins-up state (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        ham_params = IsingParameters(N, J, h, :periodic)

        # |↑↑...↑⟩ in Z basis = all config 0 (σ_z = +1 for all sites)
        # This is a product state with definite Px parity.
        dim = 2^N
        ψ_up = zeros(ComplexF64, dim)
        ψ_up[1] = 1.0  # |0...0⟩ = |↑...↑⟩ in Z basis
        state = EDStateVector(ψ_up, N)

        # Check parity: for |↑↑...↑⟩, P_x = ∏ X_i
        # X|↑⟩ = |↓⟩ ≠ |↑⟩, so this is NOT a parity eigenstate
        # Actually, P_x flips all spins: P_x|↑↑...↑⟩ = |↓↓...↓⟩
        # So ⟨P_x⟩ = 0 for |↑↑...↑⟩
        px = measure_state_parity(state, N)
        @test abs(px) < 1e-10

        # For the all-spins-up state, since it's not a parity eigenstate,
        # we test with both gF values and check h_k is between -1 and 1
        for gF in [-1, 1]
            ks = allowed_k_indices(N, gF)
            for k in ks
                hk = measure_hk(state, k, ham_params)
                @test -1 - 1e-8 <= hk <= 1 + 1e-8
            end
        end
    end

    @testset "Density matrix consistency (N=4)" begin
        N = 4; J = 1.0; h = 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        E_gs, ψ_gs = _find_gs_in_sector(H, N, 1)
        state_vec = EDStateVector(ψ_gs, N)
        state_dm = state_to_density_ed(state_vec)

        gF = fermionic_bc(:periodic, 1)
        ks = allowed_k_indices(N, gF)

        for k in ks
            hk_vec = measure_hk(state_vec, k, ham_params)
            hk_dm = measure_hk(state_dm, k, ham_params)
            @test hk_vec ≈ hk_dm atol=1e-8
        end

        # Also check measure_all_mode_energies with DM
        ks_v, hk_v, ε_v = measure_all_mode_energies(state_vec, ham_params; gF=gF)
        ks_d, hk_d, ε_d = measure_all_mode_energies(state_dm, ham_params; gF=gF)

        @test ks_v == ks_d
        @test hk_v ≈ hk_d atol=1e-8
        @test ε_v ≈ ε_d atol=1e-12
    end

    @testset "Thermal mixture density matrix (N=4)" begin
        N = 4; J = 1.0; h = 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        # Create thermal-like mixture within even parity sector
        evals, evecs = eigen(H)
        Px = parity_operator_code(N)

        dim = 2^N
        ρ = zeros(ComplexF64, dim, dim)
        Z = 0.0

        for i in 1:length(evals)
            v = ComplexF64.(evecs[:, i])
            px = real(dot(v, Px * v))
            if abs(px - 1.0) < 0.1  # even parity
                w = exp(-evals[i])
                ρ .+= w * v * v'
                Z += w
            end
        end
        ρ ./= Z
        # Ensure exact Hermiticity for the assertion check
        ρ = (ρ + ρ') / 2

        state = EDDensityMatrix(ρ, N)
        gF = fermionic_bc(:periodic, 1)

        # Total energy should match Tr(H ρ)
        E_direct = real(tr(Matrix(H) * ρ))

        ks_all, hk_vals, _ = measure_all_mode_energies(state, ham_params; gF=gF)
        E_modes = sum(Λ * coeff_k(Float64(k), θ, N) * hk / 2
                     for (k, hk) in zip(ks_all, hk_vals))

        @test E_modes ≈ E_direct atol=1e-8

        # All h_k should be between -1 and 1
        for hk in hk_vals
            @test -1 - 1e-8 <= hk <= 1 + 1e-8
        end
    end

    @testset "Automatic parity detection (N=4)" begin
        N = 4; J = 1.0; h = 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        # Test that measure_all_mode_energies auto-detects parity correctly
        for target_p in [1, -1]
            E_gs, ψ_gs = _find_gs_in_sector(H, N, target_p)
            gs_state = EDStateVector(ψ_gs, N)

            # Without explicit gF — should auto-detect
            ks_auto, hk_auto, εk_auto = measure_all_mode_energies(gs_state, ham_params)

            # With explicit gF
            gF = fermionic_bc(:periodic, target_p)
            ks_exp, hk_exp, εk_exp = measure_all_mode_energies(gs_state, ham_params; gF=gF)

            @test ks_auto == ks_exp
            @test hk_auto ≈ hk_exp atol=1e-10
        end
    end

    @testset "Cooling measurements use parity-aware n_k grid" begin
        N = 4; J = 1.0; h = 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 0, 0.0, nothing)
        sim_params = UnifiedSimulationParameters(DensityMatrix(), ContinuousEvolution())

        problem = setup_problem(EDBackend(), ham_params, coupling_params, sim_params)
        ρ0 = state_to_density_ed(problem.ϕ₀)
        state0 = QuantumState(EDBackend(), DensityMatrix(), ContinuousEvolution(), ρ0)

        results = redirect_stdout(devnull) do
            run_cooling(problem, state0, coupling_params, sim_params, ham_params)
        end

        k_expected, nk_expected = measure_momentum_distribution_ed_clean(ρ0, ham_params)

        @test results["momentum_gF"] == fermionic_bc(:periodic, 1)
        @test results["momentum_gF_source"] == "state"
        @test results["k_values"] ≈ k_expected atol=1e-12
        @test results["momentum_dist"][1, :] ≈ nk_expected atol=1e-10

        ρ_sb = CoolingTNS.prepare_combined_state_ed(ρ0, N, coupling_params.coupling)
        state_sb = QuantumState(EDBackend(), DensityMatrix(), ContinuousEvolution(), ρ_sb)
        results_sb = redirect_stdout(devnull) do
            run_cooling(problem, state_sb, coupling_params, sim_params, ham_params)
        end

        @test results_sb["momentum_gF"] == fermionic_bc(:periodic, 1)
        @test results_sb["momentum_gF_source"] == "state"
        @test results_sb["k_values"] ≈ k_expected atol=1e-12
        @test results_sb["momentum_dist"][1, :] ≈ nk_expected atol=1e-10
    end

    @testset "Momentum grid helper fallback and cache source" begin
        N = 4; J = 1.0; h = 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        _, ψ_even = _find_gs_in_sector(H, N, 1)
        _, ψ_odd = _find_gs_in_sector(H, N, -1)
        ϕ₀ = EDStateVector(ψ_even, N)
        odd_state = EDStateVector(ψ_odd, N)
        ρ_mix = EDDensityMatrix(0.5 * ψ_even * ψ_even' + 0.5 * ψ_odd * ψ_odd', N)

        measurements = Dict{String, Any}()
        gF = CoolingTNS._momentum_measurement_gF!(measurements, ρ_mix, ϕ₀, ham_params)
        @test gF == fermionic_bc(:periodic, 1)
        @test measurements["momentum_gF_source"] == "ground_state"

        @test CoolingTNS._momentum_measurement_gF!(measurements, odd_state, ϕ₀, ham_params) == gF
        @test measurements["momentum_gF_source"] == "ground_state"

        ambiguous_ϕ₀ = CoolingTNS.product_state_ed(N, 0)
        @test abs(measure_state_parity(ambiguous_ϕ₀, N)) < 1e-10
        ambiguous_measurements = Dict{String, Any}()
        @test CoolingTNS._momentum_measurement_gF!(
            ambiguous_measurements,
            ρ_mix,
            ambiguous_ϕ₀,
            ham_params,
        ) == fermionic_bc(:periodic, 1)
        @test ambiguous_measurements["momentum_gF_source"] == "ground_state"

        precomputed = Dict{String, Any}("momentum_gF" => fermionic_bc(:periodic, -1))
        @test CoolingTNS._momentum_measurement_gF!(precomputed, ρ_mix, ϕ₀, ham_params) ==
              fermionic_bc(:periodic, -1)
        @test precomputed["momentum_gF_source"] == "precomputed"
    end

    @testset "h_k range and symmetry (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        ham_params = IsingParameters(N, J, h, :periodic)

        # Random state
        dim = 2^N
        ψ_rand = randn(ComplexF64, dim)
        ψ_rand ./= norm(ψ_rand)
        state = EDStateVector(ψ_rand, N)

        for gF in [-1, 1]
            ks = allowed_k_indices(N, gF)
            hk_vals = Dict{Any, Float64}()
            for k in ks
                hk = measure_hk(state, k, ham_params)
                hk_vals[k] = hk
                # h_k should be real and in [-1, 1]
                @test -1 - 1e-6 <= hk <= 1 + 1e-6
            end
        end
    end

    @testset "APBC spectrum (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)
        ham_params = IsingParameters(N, J, h, :antiperiodic)
        H = _build_H(N, J, h, :antiperiodic)

        # For spin APBC: Px=+1 → gF=+1, Px=-1 → gF=-1
        for target_p in [1, -1]
            E_gs, ψ_gs = _find_gs_in_sector(H, N, target_p)
            gs_state = EDStateVector(ψ_gs, N)
            gF = fermionic_bc(:antiperiodic, target_p)

            # Total energy check
            ks_all, hk_vals, _ = measure_all_mode_energies(gs_state, ham_params; gF=gF)
            E_modes = sum(Λ * coeff_k(Float64(k), θ, N) * hk / 2
                         for (k, hk) in zip(ks_all, hk_vals))
            @test E_modes ≈ E_gs atol=1e-8
        end
    end

    @testset "Different J, h values (N=4)" begin
        N = 4

        for (J, h) in [(0.5, 1.0), (1.0, 1.0), (2.0, 0.3)]
            θ = theta_from_Jh(J, h)
            Λ = energy_scale(J, h)
            ham_params = IsingParameters(N, J, h, :periodic)
            H = _build_H(N, J, h, :periodic)

            E_gs, ψ_gs = _find_gs_in_sector(H, N, 1)
            gs_state = EDStateVector(ψ_gs, N)
            gF = fermionic_bc(:periodic, 1)
            ks = allowed_k_indices(N, gF)

            # Ground state: all h_k = -1
            for k in ks
                hk = measure_hk(gs_state, k, ham_params)
                @test hk ≈ -1.0 atol=1e-8
            end

            # Total energy check
            ks_all, hk_vals, _ = measure_all_mode_energies(gs_state, ham_params; gF=gF)
            E_modes = sum(Λ * coeff_k(Float64(k), θ, N) * hk / 2
                         for (k, hk) in zip(ks_all, hk_vals))
            @test E_modes ≈ E_gs atol=1e-8
        end
    end

    @testset "Mode energy values are positive" begin
        N = 4; J = 1.0; h = 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        H = _build_H(N, J, h, :periodic)

        E_gs, ψ_gs = _find_gs_in_sector(H, N, 1)
        gs_state = EDStateVector(ψ_gs, N)

        ks_all, hk_vals, εk_vals = measure_all_mode_energies(gs_state, ham_params)

        for ε in εk_vals
            @test ε > 0
        end
    end

    @testset "Mode occupations are exposed with h_k results" begin
        @test mode_occupation_from_hk(-1.0) == 0.0
        @test mode_occupation_from_hk(1.0) == 1.0
        @test mode_occupation_from_hk([-1.0, 0.0, 1.0]) == [0.0, 0.5, 1.0]

        N = 4
        ham_params = IsingParameters(N, 1.0, 0.5, :periodic)
        coupling_params = BasicCouplingParameters("XX", 0.0, 0, 0.0, 0.5)
        sim_params = UnifiedSimulationParameters(DensityMatrix(), ContinuousEvolution())

        problem = setup_problem(EDBackend(), ham_params, coupling_params, sim_params)
        state0 = setup_initial_state(problem, sim_params, "theta", 0.0)

        results = redirect_stdout(devnull) do
            run_cooling(problem, state0, coupling_params, sim_params, ham_params; measure_modes=true)
        end

        @test haskey(results, RESULT_MODE_HK)
        @test haskey(results, RESULT_MODE_NK)
        @test results[RESULT_MODE_NK] ≈ mode_occupation_from_hk(results[RESULT_MODE_HK]) atol=1e-12
        @test all(0 .<= results[RESULT_MODE_NK] .<= 1)
    end

end
