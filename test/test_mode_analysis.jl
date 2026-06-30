"""
Tests for mode_analysis.jl: parameter mapping, analytic dispersion, JW convention,
and cross-validation against exact diagonalization.

These tests verify that:
1. The parameter mapping between (J,h) and θ is correct and invertible
2. The analytic mode energies match ED eigenvalue gaps
3. The vacuum energy matches the ED ground state energy in the correct parity sector
4. The JW convention gives the correct raw Fourier occupation matching
   Bogoliubov predictions
5. The fermionic boundary conditions are correctly determined from spin BC and parity
"""

using Test
using CoolingTNS
using LinearAlgebra

# ============================================================================
# Helper functions for building Hamiltonians and measuring in the test
# ============================================================================

"""Build single-site operator on site `site` in N-site system (complex)."""
function _test_site_op(op::AbstractMatrix, site::Int, N::Int)
    I2 = Matrix{ComplexF64}(I, 2, 2)
    ops = [copy(I2) for _ in 1:N]
    ops[site] = ComplexF64.(op)
    foldl(kron, ops)
end

"""Build H_code = J Σ Z_i Z_{i+1} + h Σ X_i with given BC."""
function _test_build_H_code(N::Int, J::Float64, h::Float64, bc::Symbol)
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)
    for i in 1:N-1
        H .+= J * _test_site_op(Z, i, N) * _test_site_op(Z, i+1, N)
    end
    if bc == :periodic
        H .+= J * _test_site_op(Z, N, N) * _test_site_op(Z, 1, N)
    elseif bc == :antiperiodic
        H .-= J * _test_site_op(Z, N, N) * _test_site_op(Z, 1, N)
    end
    for i in 1:N
        H .+= h * _test_site_op(X, i, N)
    end
    return H
end

"""Build H_notes = (cosθ/2) Σ X_i X_{i+1} + (sinθ/2) Σ Z_i with given BC (spin BC)."""
function _test_build_H_notes(N::Int, θ::Float64, bc::Symbol)
    X = ComplexF64[0 1; 1 0]
    Z = ComplexF64[1 0; 0 -1]
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)
    for i in 1:N-1
        H .+= (cos(θ)/2) * _test_site_op(X, i, N) * _test_site_op(X, i+1, N)
    end
    if bc == :periodic
        H .+= (cos(θ)/2) * _test_site_op(X, N, N) * _test_site_op(X, 1, N)
    elseif bc == :antiperiodic
        H .-= (cos(θ)/2) * _test_site_op(X, N, N) * _test_site_op(X, 1, N)
    end
    for i in 1:N
        H .+= (sin(θ)/2) * _test_site_op(Z, i, N)
    end
    return H
end

"""Build parity operator P_x = Π X_i (code) or P_z = Π Z_i (notes)."""
function _test_parity_x(N::Int)
    X = ComplexF64[0 1; 1 0]
    foldl(*, [_test_site_op(X, i, N) for i in 1:N])
end

function _test_parity_z(N::Int)
    Z = ComplexF64[1 0; 0 -1]
    foldl(*, [_test_site_op(Z, i, N) for i in 1:N])
end

"""Split eigenvalues of H into parity sectors using parity operator P."""
function _test_split_by_parity(H::AbstractMatrix, P::AbstractMatrix)
    evals, evecs = eigen(Hermitian(real(H)))
    dim = length(evals)

    # Re-diagonalize P within degenerate subspaces to get clean parity eigenstates
    tol = 1e-8
    all_E = Float64[]
    all_P = Int[]
    i = 1
    while i <= dim
        j = i
        while j < dim && abs(evals[j+1] - evals[i]) < tol
            j += 1
        end
        V = evecs[:, i:j]
        P_sub = V' * P * V
        ep, _ = eigen(Hermitian(real(P_sub)))
        for k in 1:length(ep)
            push!(all_E, evals[i])
            push!(all_P, round(Int, ep[k]))
        end
        i = j + 1
    end

    even_E = sort(all_E[all_P .== 1])
    odd_E = sort(all_E[all_P .== -1])
    return even_E, odd_E
end

"""Return energy eigenpairs in a given parity sector."""
function _test_sector_eigensystem(H::AbstractMatrix, P::AbstractMatrix, parity::Int)
    evals, evecs = eigen(Hermitian(real(H)))
    dim = length(evals)
    p_vals = Float64[]
    # Re-diag in degenerate subspaces
    tol = 1e-8
    result_evals = Float64[]
    result_evecs = Vector{Vector{ComplexF64}}()
    i = 1
    while i <= dim
        j = i
        while j < dim && abs(evals[j+1] - evals[i]) < tol
            j += 1
        end
        V = evecs[:, i:j]
        P_sub = V' * P * V
        ep, vp = eigen(Hermitian(real(P_sub)))
        for k in 1:length(ep)
            push!(result_evals, evals[i])
            push!(result_evecs, V * vp[:, k])
            push!(p_vals, round(Int, ep[k]))
        end
        i = j + 1
    end

    mask = p_vals .== parity
    sector_E = result_evals[mask]
    sector_V = result_evecs[mask]
    order = sortperm(sector_E)
    return sector_E[order], sector_V[order]
end

"""Find ground state in a given parity sector."""
function _test_gs_in_sector(H::AbstractMatrix, P::AbstractMatrix, parity::Int)
    sector_E, sector_V = _test_sector_eigensystem(H, P, parity)
    return first(sector_E), first(sector_V)
end

"""Return gaps reachable from the fixed-sector ground state by a local operator."""
function _test_accessible_gaps_from_sector_ground(
    H::AbstractMatrix,
    P::AbstractMatrix,
    O::AbstractMatrix,
    parity::Int;
    gap_tol::Float64=1e-9,
    amplitude_tol::Float64=1e-8,
)
    sector_E, sector_V = _test_sector_eigensystem(H, P, parity)
    ψ0 = first(sector_V)
    E0 = first(sector_E)
    gaps = Float64[]
    for j in eachindex(sector_E)
        gap = sector_E[j] - E0
        amplitude = abs(sector_V[j]' * O * ψ0)
        gap > gap_tol && amplitude > amplitude_tol && push!(gaps, gap)
    end
    return sort(gaps)
end

"""Build the free-fermion many-body spectrum from positive quasiparticle energies."""
function _test_many_body_spectrum_from_modes(energies)
    spectrum = [0.0]
    for ε in energies
        spectrum = vcat(spectrum, spectrum .+ ε)
    end
    return sort(spectrum .- sum(energies) / 2)
end

"""Build canonical fermion annihilation operators in the occupation basis."""
function _test_fermion_annihilation_ops(N::Int)
    I2 = Matrix{ComplexF64}(I, 2, 2)
    Z = ComplexF64[1 0; 0 -1]
    σm = ComplexF64[0 1; 0 0]
    ops = Matrix{ComplexF64}[]
    for site in 1:N
        factors = [copy(I2) for _ in 1:N]
        for j in 1:(site - 1)
            factors[j] = Z
        end
        factors[site] = σm
        push!(ops, foldl(kron, factors))
    end
    return ops
end

"""Build the stated open-chain fermion Hamiltonian directly from a_n, a_n†."""
function _test_obc_fermion_hamiltonian_from_realspace(θ::Float64, N::Int)
    s, c = sin(θ), cos(θ)
    a = _test_fermion_annihilation_ops(N)
    adag = [Matrix(op') for op in a]
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)
    for n in 1:(N - 1)
        H .+= (c / 2) * (a[n] - adag[n]) * (a[n + 1] + adag[n + 1])
    end
    for n in 1:N
        H .-= (s / 2) * (a[n] * adag[n] - adag[n] * a[n])
    end
    return H
end

"""Build 1/2 Ψ† H_BdG Ψ from the Nambu matrix, with Ψ=(a,a†)."""
function _test_obc_fermion_hamiltonian_from_bdg(θ::Float64, N::Int)
    a = _test_fermion_annihilation_ops(N)
    adag = [Matrix(op') for op in a]
    Ψ = vcat(a, adag)
    Ψdag = vcat(adag, a)
    Hbdg = obc_bdg_matrix(θ, N)
    dim = 2^N
    H = zeros(ComplexF64, dim, dim)
    for i in 1:(2N), j in 1:(2N)
        H .+= (Hbdg[i, j] / 2) * Ψdag[i] * Ψ[j]
    end
    return H
end

# ============================================================================
# Tests
# ============================================================================

@testset "Mode Analysis" begin

    @testset "Parameter mapping roundtrip" begin
        for (J, h) in [(1.0, 0.5), (0.3, 0.7), (1.0, 0.0), (0.0, 1.0),
                        (-0.5, 0.5), (0.5, -0.3), (-1.0, -1.0)]
            θ = theta_from_Jh(J, h)
            sc = sqrt(J^2 + h^2)
            J_rt, h_rt = Jh_from_theta(θ; scale=sc)
            @test J_rt ≈ J atol=1e-12
            @test h_rt ≈ h atol=1e-12
        end
    end

    @testset "Energy scale" begin
        @test energy_scale(1.0, 0.0) ≈ 2.0
        @test energy_scale(0.0, 1.0) ≈ 2.0
        @test energy_scale(1.0, 1.0) ≈ 2√2
        @test energy_scale(3.0, 4.0) ≈ 10.0
    end

    @testset "Stored positive mode gap validation" begin
        N = 4
        J, h = 1.0, 0.5
        k_indices = allowed_k_indices(N, 1)
        gaps = mode_energies_Jh(k_indices, J, h, N)

        @test validate_mode_ek_values_match_grid(gaps, k_indices, N, J, h) ≈ gaps
        @test_throws DimensionMismatch validate_mode_ek_values_match_grid(
            gaps[1:end-1], k_indices, N, J, h)

        bad_gaps = copy(gaps)
        bad_gaps[2] += 0.1
        @test_throws ArgumentError validate_mode_ek_values_match_grid(
            bad_gaps, k_indices, N, J, h)
    end

    @testset "Mode measurement cycle row validation" begin
        mode_hk = [
            -1.0 0.0
            NaN  NaN
             0.5 1.0
        ]
        mode_nk = mode_occupation_from_hk(mode_hk)

        measured = mode_measurement_cycle_rows(3, [0, 2])
        @test measured.cycles == [0, 2]
        @test measured.rows == [1, 3]
        @test mode_measurement_cycle_rows(3).cycles == [0, 1, 2]
        @test mode_measurement_cycle_rows(3, 2).rows == [3]
        @test_throws ArgumentError mode_measurement_cycle_rows(3, Int[])
        @test_throws ArgumentError mode_measurement_cycle_rows(3, [2, 0])
        @test_throws ArgumentError mode_measurement_cycle_rows(3, [0, 0])
        @test_throws ArgumentError mode_measurement_cycle_rows(3, [0, 3])

        @test validate_mode_measurement_rows(
            mode_hk, mode_nk, [0, 2]; energy=[-1.0, 100.0, -0.5]
        ).rows == [1, 3]
        @test validate_mode_measurement_rows(
            mode_hk, mode_nk, 2; energy=[-1.0, 100.0, -0.5]
        ).cycles == [2]
        @test_throws ArgumentError validate_mode_measurement_rows(
            mode_hk, nothing, [0, 2])
        bad_finite_hk_nk = copy(mode_nk)
        bad_finite_hk_nk[3, 1] = NaN
        @test_throws ArgumentError validate_mode_measurement_rows(
            mode_hk, bad_finite_hk_nk, [0, 2])
        @test_throws ArgumentError validate_mode_measurement_rows(
            mode_hk, mode_nk, [0, 1, 2])
        @test_throws ArgumentError validate_mode_measurement_rows(
            mode_hk, mode_nk, [0, 2]; energy=-0.5)
        @test_throws ArgumentError validate_mode_measurement_rows(
            mode_hk, mode_nk, [0, 2]; energy=[-1.0, 100.0, NaN])
        @test_throws DimensionMismatch validate_mode_measurement_rows(
            mode_hk, mode_nk, [0, 2]; energy=[-1.0, -0.5])
    end

    @testset "Bogoliubov text matches MapToSpin phase convention" begin
        file_contains(path, needle) = open(path) do io
            any(line -> occursin(needle, line), eachline(io))
        end
        file_lacks(path, needle) = !file_contains(path, needle)

        note_path = joinpath(@__DIR__, "..", "Notes", "NotesED", "MapToSpin.tex")
        obc_path = joinpath(@__DIR__, "..", "Notes", "NotesED", "OBC.tex")
        cooling_tn_path = joinpath(@__DIR__, "..", "Notes", "NotesTN", "CoolingAlgTN.tex")
        gaussian_main_path = joinpath(@__DIR__, "..", "Notes", "GaussianPaper", "journal_main.tex")
        gaussian_supp_path = joinpath(@__DIR__, "..", "Notes", "GaussianPaper", "journal_supp.tex")
        largeN_bond_doc_path = joinpath(@__DIR__, "..", "docs", "largeN_effective_bond_dimensions.md")
        mode_path = joinpath(@__DIR__, "..", "src", "mode_analysis.jl")
        ed_path = joinpath(@__DIR__, "..", "src", "ed_backend_complex_jw.jl")

        @test file_contains(note_path, raw"i\sin(\varphi_k) \ta_{-k}^\dag")
        @test file_contains(note_path, raw"\label{eq:code_energy_from_hk}")
        @test file_contains(note_path, raw"\label{eq:bog_occupation_from_hk}")
        @test file_contains(note_path, raw"\label{eq:raw_fourier_occupation}")
        @test file_contains(note_path, raw"\epsilon_k\tilde n_k")
        @test file_contains(note_path, raw"\label{eq:hk_individual_def}")
        @test file_lacks(note_path, raw"\cos(\varphi_k) \ta_k + \sin(\varphi_k) \ta_{-k}^\dag")
        @test file_contains(obc_path, raw"\label{eq:obc_mode_contribution}")
        @test file_contains(obc_path, raw"Thus \(H_k=\frac{\epsilon_k}{2}h_k^{\mathrm{OBC}}\), not \(\epsilon_k h_k^{\mathrm{OBC}}\)")
        @test file_lacks(obc_path, raw"The mode energy $E_k = \epsilon_k (2\gamma_k^\dag \gamma_k - 1)$")
        @test file_contains(cooling_tn_path, raw"\label{eq:TNModeEnergyReconstruction}")
        @test file_contains(cooling_tn_path, raw"two-quasiparticle scale, \cref{eq:mode_pair_threshold}")
        @test file_contains(cooling_tn_path, raw"In the notation of \cref{eq:HIsing}")
        @test file_lacks(cooling_tn_path, raw"Eq.~\eqref{eq:mode_pair_threshold}")
        @test file_lacks(cooling_tn_path, raw"Eq.~\eqref{eq:HIsing}")
        @test file_contains(cooling_tn_path, raw"h_k=2\hat a_k^\dagger\hat a_k-1")
        @test file_contains(cooling_tn_path, raw"n_k^{\mathrm{Bog}}=\frac{1+\langle h_k\rangle}{2}")
        @test file_contains(cooling_tn_path, raw"\operatorname{coeff}_k")
        @test file_contains(cooling_tn_path, raw"ground state within a fixed spin-parity sector need not have")
        @test file_contains(cooling_tn_path, raw"$\langle h_q\rangle=+1$")
        @test file_contains(cooling_tn_path, raw"zero occupation of the chosen mode")
        @test file_contains(cooling_tn_path, raw"special mode with $w_k<0$ the energy contribution is minimized by")
        @test file_contains(cooling_tn_path, raw"APBC")
        @test file_contains(cooling_tn_path, raw"integer-grid cases")
        @test file_contains(cooling_tn_path, raw"\tan(2\varphi_k)&=\frac{r_k}{w_k}")
        @test file_contains(mode_path, "``σ_x → -σ_z``, ``σ_z → σ_x``")
        @test file_contains(mode_path, "that maps ``σ_z → σ_x``")
        @test file_contains(mode_path, "Spin APBC (``g_I = -1``) with ``P_x = +1``")
        @test file_contains(mode_path, "Spin APBC (``g_I = -1``) with ``P_x = -1``")
        @test file_lacks(mode_path, "that maps ``σ_z → -σ_x``")
        @test file_contains(cooling_tn_path, raw"\hat a_k&=\cos(\varphi_k)\tilde a_k")
        @test file_contains(cooling_tn_path, raw"For the special modes with $\sin\phi_k=0$, there is no Bogoliubov mixing")
        @test file_contains(cooling_tn_path, raw"r^{\mathrm{code}}_k=-r_k")
        @test file_contains(cooling_tn_path, raw"\varphi_k^{\mathrm{code}}=-\varphi_k")
        @test file_contains(cooling_tn_path, raw"\tilde a^{\mathrm{code}}_k")
        @test file_contains(cooling_tn_path, raw"=\tilde a^{\mathrm{notes}}_{-k}")
        @test file_contains(cooling_tn_path, raw"r^{\mathrm{code}}_k&=+\cos\theta\sin\phi_k")
        @test file_contains(note_path, raw"parity-unconstrained, mode-wise energy-minimizing BdG reference")
        @test file_contains(note_path, raw"\tilde n_k^{\mathrm{ref}}=1")
        @test file_contains(note_path, raw"the true sector ground state can differ from both the chosen-operator vacuum")
        @test file_contains(cooling_tn_path, raw"M_{\mathrm{TDVP}}=\left\lceil \frac{t_e}{\tau}\right\rceil")
        @test file_contains(cooling_tn_path, raw"\label{eq:TDVPKrylovExpansion}")
        @test file_contains(cooling_tn_path, raw"\mathcal{K}_m(H_{SB},\ket{\Psi_0})")
        @test file_contains(cooling_tn_path, raw"Two cooling cycles are not physically meaningful")
        @test file_contains(cooling_tn_path, raw"This is the presently relevant \(D_{\max}=96\) TDVP evidence")
        @test file_contains(cooling_tn_path, raw"more meaningful than the two-cycle calibration")
        @test file_contains(cooling_tn_path, raw"D_{\mathrm{sys}}^{\mathrm{eff}}\geq 96")
        @test file_contains(cooling_tn_path, raw"cycle 3 sweep 4")
        @test file_contains(cooling_tn_path, raw"\texttt{not\_converged\_system\_and\_evolved\_and}")
        @test file_contains(cooling_tn_path, raw"\texttt{\_tdvp\_sweep\_cap}")
        @test file_contains(largeN_bond_doc_path, "Five-Cycle MCWF+TDVP R=5, Dmax=96 Probe")
        @test file_contains(
            largeN_bond_doc_path,
            "0.66394232 | >=96 | >=96 | " *
            "not_converged_system_and_evolved_and_tdvp_sweep_cap",
        )
        @test file_contains(mode_path, "â_k = cos(varphi_k) ã_k + i sin(varphi_k) ã†_{-k}")
        @test file_lacks(mode_path, "â_k = cos(varphi_k) ã_k + sin(varphi_k) ã†_{-k}")
        @test file_contains(ed_path, "ã_k = (1/√N) Σ_j exp(-i n φ_k) a_j.")
        @test file_contains(ed_path, "exp(+i (m-n) φ_k)")
        @test file_lacks(ed_path, "where a_k = (1/√N) Σ_j exp(+2πikj/N) a_j")
        @test file_contains(gaussian_main_path, "real Nambu gauge")
        @test file_contains(gaussian_main_path, raw"\bar{a}_{-k}^\dagger \equiv -i\tilde{a}_{-k}^\dagger")
        @test file_contains(gaussian_main_path, raw"i\sin(\varphi_k)\bar{a}_{-k}^\dagger")
        @test file_contains(gaussian_supp_path, raw"G_k^\dagger \tilde{H}_k G_k")
        @test file_contains(gaussian_supp_path, raw"\label{eq:bdg_gauge_map}")
        @test file_contains(gaussian_supp_path, raw"real Nambu gauge of \cref{eq:Hk_2x2_matrix}")
        @test file_contains(gaussian_supp_path, "phase-explicit BdG form")
    end

    @testset "Open-boundary BdG matrices use canonical JW convention" begin
        N = 4
        θ = 0.37
        s, c = sin(θ), cos(θ)
        A, B = obc_bdg_matrices(θ, N)

        @test diag(A) ≈ fill(s, N)
        @test diag(B) ≈ zeros(N)
        for n in 1:(N - 1)
            @test A[n, n + 1] ≈ -c / 2
            @test A[n + 1, n] ≈ -c / 2
            @test B[n, n + 1] ≈ -c / 2
            @test B[n + 1, n] ≈ c / 2
        end

        @test A + B ≈ Tridiagonal(zeros(N - 1), fill(s, N), fill(-c, N - 1))
        @test A - B ≈ Tridiagonal(fill(-c, N - 1), fill(s, N), zeros(N - 1))
    end

    @testset "Open-boundary BdG matrix equals stated real-space fermion Hamiltonian" begin
        for N in [2, 3, 4], θ in [0.31, 0.92]
            H_realspace = _test_obc_fermion_hamiltonian_from_realspace(θ, N)
            H_bdg = _test_obc_fermion_hamiltonian_from_bdg(θ, N)
            @test H_bdg ≈ H_realspace atol=1e-12
        end
    end

    @testset "Open-boundary BdG spectrum matches exact ED (N=$N, J=$J, h=$h)" for
            N in [2, 3, 4, 5],
            (J, h) in [(1.0, 0.5), (0.7, -0.3), (0.4, 1.1)]

        H_code = _test_build_H_code(N, J, h, :open)
        exact_spectrum = sort(eigvals(Hermitian(real(H_code))))
        bdg_spectrum = _test_many_body_spectrum_from_modes(obc_mode_energies_Jh(J, h, N))

        @test bdg_spectrum ≈ exact_spectrum atol=1e-10
    end

    @testset "Fermionic BC" begin
        # Spin PBC (gI=+1) with Px=+1: gF = -1*1 = -1
        @test fermionic_bc(:periodic, 1) == -1
        # Spin PBC (gI=+1) with Px=-1: gF = -1*(-1) = +1
        @test fermionic_bc(:periodic, -1) == 1
        # Spin APBC (gI=-1) with Px=+1: gF = -(-1)*1 = +1
        @test fermionic_bc(:antiperiodic, 1) == 1
        # Spin APBC (gI=-1) with Px=-1: gF = -(-1)*(-1) = -1
        @test fermionic_bc(:antiperiodic, -1) == -1
    end

    @testset "Reference parity sector for automatic Fourier grids" begin
        @test CoolingTNS._reference_parity_sector(1.0) == 1
        @test CoolingTNS._reference_parity_sector(0.95) == 1
        @test CoolingTNS._reference_parity_sector(-1.0) == -1
        @test CoolingTNS._reference_parity_sector(-0.95) == -1

        @test CoolingTNS._reference_parity_sector(0.0) == 1
        @test CoolingTNS._reference_parity_sector(0.5) == 1
        @test CoolingTNS._reference_parity_sector(-0.5) == 1
        @test CoolingTNS._reference_parity_sector(0.0; default=-1) == -1

        even_sector = CoolingTNS._reference_parity_sector_with_source(0.95)
        @test even_sector.parity == 1
        @test even_sector.source === :state
        @test fermionic_grid_source_label(even_sector.source) == FERMIONIC_GRID_SOURCE_STATE
        mixed_sector = CoolingTNS._reference_parity_sector_with_source(0.0)
        @test mixed_sector.parity == 1
        @test mixed_sector.source === :reference
        @test fermionic_grid_source_label(mixed_sector.source) == FERMIONIC_GRID_SOURCE_REFERENCE
        odd_reference = CoolingTNS._reference_parity_sector_with_source(0.0; default=-1)
        @test odd_reference.parity == -1
        @test odd_reference.source === :reference
        @test FERMIONIC_GRID_SOURCE_LABELS == (
            FERMIONIC_GRID_SOURCE_STATE,
            FERMIONIC_GRID_SOURCE_REFERENCE,
            FERMIONIC_GRID_SOURCE_PRECOMPUTED,
        )
        @test_throws ArgumentError fermionic_grid_source_label(:unknown)

        @test CoolingTNS._reference_fermionic_bc(:periodic, 0.0) ==
              fermionic_bc(:periodic, 1)
        @test CoolingTNS._reference_fermionic_bc(:antiperiodic, -0.95) ==
              fermionic_bc(:antiperiodic, -1)

        @test_throws AssertionError CoolingTNS._reference_parity_sector(0.0; default=0)
    end

    @testset "Allowed k-indices" begin
        # gF=+1 (fermionic PBC): integer k
        ks_pbc = allowed_k_indices(4, 1)
        @test ks_pbc == [-1, 0, 1, 2]

        # gF=-1 (fermionic APBC): half-integer k
        ks_apbc = allowed_k_indices(4, -1)
        @test ks_apbc == [-3//2, -1//2, 1//2, 3//2]

        # N=6
        ks6_pbc = allowed_k_indices(6, 1)
        @test ks6_pbc == [-2, -1, 0, 1, 2, 3]
        ks6_apbc = allowed_k_indices(6, -1)
        @test ks6_apbc == [-5//2, -3//2, -1//2, 1//2, 3//2, 5//2]

        # Symbol interface
        @test allowed_k_indices(4, :periodic) == allowed_k_indices(4, 1)
        @test allowed_k_indices(4, :antiperiodic) == allowed_k_indices(4, -1)
    end

    @testset "Complex JW uses canonical k-grid" begin
        source = read(joinpath(@__DIR__, "..", "src", "ed_backend_complex_jw.jl"), String)
        @test !occursin(r"function\s+_allowed_k_indices\b", source)
        @test occursin("allowed_k_indices(N, gF)", source)
    end

    @testset "Fourier observable support predicate" begin
        @test supports_ising_fourier_observables(IsingParameters(4, 1.0, 0.5, :periodic))
        @test supports_ising_fourier_observables(IsingParameters(4, 1.0, 0.5, :antiperiodic))
        @test !supports_ising_fourier_observables(IsingParameters(3, 1.0, 0.5, :periodic))
        @test !supports_ising_fourier_observables(IsingParameters(4, 1.0, 0.5, :open))
        @test !supports_ising_fourier_observables(NiIsingParameters(4, 1.0, -1.05, 0.5, :periodic))
        @test !supports_ising_fourier_observables(RydbergParameters(4, 1.0, 0.0, 1.0, :periodic))
        @test !supports_ising_fourier_observables(nothing)

        @test require_ising_fourier_observables(
            IsingParameters(4, 1.0, 0.5, :periodic)) === nothing

        odd_err = try
            require_ising_fourier_observables(IsingParameters(3, 1.0, 0.5, :periodic))
            nothing
        catch err
            err
        end
        @test odd_err isa ArgumentError
        @test occursin("even N", odd_err.msg)

        open_err = try
            require_ising_fourier_observables(IsingParameters(4, 1.0, 0.5, :open))
            nothing
        catch err
            err
        end
        @test open_err isa ArgumentError
        @test occursin("spin :periodic or :antiperiodic", open_err.msg)

        ni_err = try
            require_ising_fourier_observables(
                NiIsingParameters(4, 1.0, -1.05, 0.5, :periodic))
            nothing
        catch err
            err
        end
        @test ni_err isa ArgumentError
        @test occursin("integrable transverse-field Ising", ni_err.msg)

        nothing_err = try
            require_ising_fourier_observables(nothing)
            nothing
        catch err
            err
        end
        @test nothing_err isa ArgumentError
        @test occursin("got nothing", nothing_err.msg)
    end

    @testset "Mode detuning reference is the parity-preserving two-quasiparticle scale" begin
        ham = IsingParameters(64, 1.0, -1.05, :periodic)
        expected_even = minimum(filter(
            >(sqrt(eps(Float64))),
            mode_energies_Jh(
                generic_k_indices(ham.N, fermionic_bc(ham.bc, 1)),
                ham.params.J,
                ham.params.h,
                ham.N,
            ),
        ))
        expected_odd = minimum(filter(
            >(sqrt(eps(Float64))),
            mode_energies_Jh(
                generic_k_indices(ham.N, fermionic_bc(ham.bc, -1)),
                ham.params.J,
                ham.params.h,
                ham.N,
            ),
        ))

        @test ising_mode_detuning_reference(ham) ≈ 2 * expected_even atol=1e-14
        @test ising_mode_detuning_reference(ham) > 0
        @test ising_mode_detuning_reference(ham; parity=-1) ≈ 2 * expected_odd atol=1e-14
        @test_throws ArgumentError ising_mode_detuning_reference(
            IsingParameters(4, 1.0, 0.5, :open)
        )
        @test_throws ArgumentError ising_mode_detuning_reference(
            NiIsingParameters(4, 1.0, -1.05, 0.5, :periodic)
        )
        @test_throws ArgumentError ising_mode_detuning_reference(ham; parity=0)
    end

    @testset "Mode detuning admissibility predicates are centralized" begin
        periodic_ham = IsingParameters(64, 1.0, -1.05, :periodic)
        antiperiodic_ham = IsingParameters(64, 1.0, -1.05, :antiperiodic)

        @test ising_mode_detuning_preserves_px("XX")
        @test !ising_mode_detuning_preserves_px("XY")
        @test !ising_mode_detuning_preserves_px("ZZ")
        @test_throws ArgumentError ising_mode_detuning_preserves_px("bad")

        @test !ising_mode_detuning_has_special_modes(periodic_ham)
        @test ising_mode_detuning_has_special_modes(periodic_ham; parity=-1)
        @test ising_mode_detuning_has_special_modes(antiperiodic_ham)
        @test !ising_mode_detuning_has_special_modes(antiperiodic_ham; parity=-1)
        @test_throws ArgumentError ising_mode_detuning_has_special_modes(
            IsingParameters(64, 1.0, -1.05, :open)
        )
        @test_throws ArgumentError ising_mode_detuning_has_special_modes(
            NiIsingParameters(64, 1.0, -1.05, 0.5, :periodic)
        )
        @test_throws ArgumentError ising_mode_detuning_has_special_modes(periodic_ham; parity=0)

        @test ising_mode_detuning_reference(periodic_ham) > 0
    end

    @testset "Parity-preserving detuning reference matches local X selection rule" begin
        J, h = 1.0, -1.05
        X = ComplexF64[0 1; 1 0]
        for N in [4, 6]
            ham = IsingParameters(N, J, h, :periodic)
            H = _test_build_H_code(N, J, h, :periodic)
            Px = _test_parity_x(N)
            O = _test_site_op(X, 1, N)
            # This is the default periodic even sector used by mode campaigns.
            # Its fermionic grid is half-integer, so there are no special modes
            # and the generic pair threshold is the first local-X transition.
            @test fermionic_bc(ham.bc, 1) == -1
            accessible_gaps = _test_accessible_gaps_from_sector_ground(H, Px, O, 1)

            @test ising_mode_detuning_reference(ham) ≈ minimum(accessible_gaps) atol=1e-10
        end
    end

    @testset "Integer-grid detuning reference excludes special-mode transitions" begin
        J, h = 1.0, -1.05
        X = ComplexF64[0 1; 1 0]
        for N in [4, 6], (bc, parity) in [(:periodic, -1), (:antiperiodic, 1)]
            ham = IsingParameters(N, J, h, bc)
            H = _test_build_H_code(N, J, h, bc)
            Px = _test_parity_x(N)
            O = _test_site_op(X, 1, N)
            @test fermionic_bc(bc, parity) == 1
            accessible_gaps = _test_accessible_gaps_from_sector_ground(H, Px, O, parity)
            reference = ising_mode_detuning_reference(ham; parity)

            # On the integer grid, lower local-X transitions can rearrange the
            # special modes.  The detuning reference is instead the generic
            # two-quasiparticle threshold, which remains visible in ED.
            @test any(gap -> isapprox(gap, reference; atol=1e-10), accessible_gaps)
            @test minimum(accessible_gaps) < reference
        end
    end

    @testset "H_notes from JW equals ED Hamiltonian (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)

        # Build H_notes with PBC
        H_notes = _test_build_H_notes(N, θ, :periodic)
        H_code = _test_build_H_code(N, J, h, :periodic)

        # Eigenvalues should be related by scale factor Λ
        evals_notes = sort(real.(eigvals(H_notes)))
        evals_code = sort(real.(eigvals(H_code)))
        @test evals_code ≈ Λ * evals_notes atol=1e-10
    end

    @testset "Chosen-operator vacuum energy matches even-sector ED ground state (N=$N, bc=$bc)" for
            N in [4, 6], bc in [:periodic]

        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)

        # Build H_code with PBC
        H_code = _test_build_H_code(N, J, h, bc)
        Px = _test_parity_x(N)

        @test norm(H_code * Px - Px * H_code) < 1e-10  # [H, Px] = 0

        # Even parity sector (Px=+1) → gF=-1 (half-integer k, no special modes).
        # Here the chosen-operator vacuum has Nf=0 and is the sector ground state.
        gF_even = fermionic_bc(bc, 1)
        E_gs_even, _ = _test_gs_in_sector(H_code, Px, 1)
        E_vac_even = vacuum_energy_Jh(N, J, h, gF_even)
        @test E_gs_even ≈ E_vac_even atol=1e-10

        # Odd parity sector (Px=-1) → gF=+1 (integer k, with special modes).
        # The chosen-operator vacuum has Nf=0, so it does not live in the odd sector.
        # The odd-sector GS is obtained by occupying the special mode with the
        # lowest signed w_k.
        # When w_k < 0, occupying that mode LOWERS the energy below E_vac.
        gF_odd = fermionic_bc(bc, -1)
        E_gs_odd, _ = _test_gs_in_sector(H_code, Px, -1)
        E_vac_odd = vacuum_energy_Jh(N, J, h, gF_odd)
        # The cheapest excitation: flip the special mode with min(w_k)
        w_special = [Λ * w_k_coefficient(Float64(k), θ, N)
                     for k in allowed_k_indices(N, gF_odd)
                     if !is_generic_mode(k, N)]
        E_odd_predicted = E_vac_odd + minimum(w_special)
        @test E_gs_odd ≈ E_odd_predicted atol=1e-10
    end

    @testset "Mode energies match eigenvalue gaps (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)

        H_code = _test_build_H_code(N, J, h, :periodic)
        Px = _test_parity_x(N)

        # Even parity sector (gF=-1)
        even_E, _ = _test_split_by_parity(H_code, Px)

        gF = -1
        ks = allowed_k_indices(N, gF)

        # Mode energies (code units)
        ε_modes = [mode_energy_Jh(Float64(k), J, h, N) for k in ks if Float64(k) > 0]

        # In the even half-integer-grid sector, the chosen-operator vacuum is the GS.
        # Excited states are obtained by creating quasiparticle pairs.
        # The smallest excitation gap within the sector should be 2ε_min
        # (creating a pair in the lowest mode).
        E_gs = minimum(even_E)
        gaps = sort(even_E .- E_gs)

        # The first nonzero gap should be 2 * min(ε_k) for k > 0
        ε_min = minimum(ε_modes)
        first_gap = gaps[findfirst(g -> g > 1e-8, gaps)]
        @test first_gap ≈ 2ε_min atol=1e-8
    end

    @testset "Full predicted spectrum matches ED (N=$N, sector=$sector_name)" for
            N in [4, 6],
            (sector_name, parity_val) in [("even", 1), ("odd", -1)]

        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)

        H_code = _test_build_H_code(N, J, h, :periodic)
        Px = _test_parity_x(N)

        even_E, odd_E = _test_split_by_parity(H_code, Px)
        ed_sector = parity_val == 1 ? even_E : odd_E

        gF = fermionic_bc(:periodic, parity_val)
        ks = allowed_k_indices(N, gF)

        # Build predicted spectrum by enumerating all occupation patterns
        # consistent with the correct fermion parity
        target_Nf_parity = parity_val == 1 ? 0 : 1  # even sector → Nf even, odd → Nf odd

        # For gF=-1: all modes are paired (k, -k) with k > 0
        # For gF=+1: special modes k=0, N/2 plus paired modes
        positive_ks = [k for k in ks if Float64(k) > 0]
        special_ks = [k for k in ks if !is_generic_mode(k, N)]
        paired_ks = [k for k in ks if Float64(k) > 0 && is_generic_mode(k, N)]

        # Build spectrum
        predicted = Float64[]

        if gF == -1
            # Only paired modes, no special
            @assert isempty(special_ks)
            n_pairs = length(paired_ks)
            # Each paired mode (k,-k) has states: (n_k, n_{-k}) ∈ {(0,0),(1,0),(0,1),(1,1)}
            # h_k = n_k + n_{-k} - 1, energy contribution = ε_k * h_k
            for occ_pattern in 0:(4^n_pairs - 1)
                Nf = 0
                E = 0.0
                pat = occ_pattern
                for (idx, k) in enumerate(paired_ks)
                    nk = pat % 2; pat ÷= 2
                    nmk = pat % 2; pat ÷= 2
                    Nf += nk + nmk
                    hk = nk + nmk - 1
                    E += mode_energy_Jh(Float64(k), J, h, N) * hk
                end
                if Nf % 2 == target_Nf_parity
                    push!(predicted, E)
                end
            end
        else
            # gF=+1: special modes + paired modes
            n_special = length(special_ks)
            n_pairs = length(paired_ks)
            for special_occ in 0:(2^n_special - 1)
                for pair_occ in 0:(4^n_pairs - 1)
                    Nf = 0
                    E = 0.0
                    # Special modes
                    sp = special_occ
                    for (idx, k) in enumerate(special_ks)
                        nk = sp % 2; sp ÷= 2
                        Nf += nk
                        hk = 2nk - 1
                        wk = Λ * w_k_coefficient(Float64(k), θ, N)
                        E += wk * hk / 2
                    end
                    # Paired modes
                    pa = pair_occ
                    for (idx, k) in enumerate(paired_ks)
                        nk = pa % 2; pa ÷= 2
                        nmk = pa % 2; pa ÷= 2
                        Nf += nk + nmk
                        hk = nk + nmk - 1
                        E += mode_energy_Jh(Float64(k), J, h, N) * hk
                    end
                    if Nf % 2 == target_Nf_parity
                        push!(predicted, E)
                    end
                end
            end
        end

        @test length(predicted) == length(ed_sector)
        @test sort(predicted) ≈ sort(ed_sector) atol=1e-8
    end

    @testset "Raw Fourier occupation from JW matches Bogoliubov prediction (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)

        # Build H_notes (the JW formulas apply to the notes' Hamiltonian)
        H_notes = _test_build_H_notes(N, θ, :periodic)
        Pz = _test_parity_z(N)

        # Ground state in Pz=+1 (even) sector → gF=-1
        E_gs, ψ_gs = _test_gs_in_sector(H_notes, Pz, 1)

        gF = -1
        ks = allowed_k_indices(N, gF)

        # Get JW operators
        a_ops = [jordan_wigner_transform_complex(n, N) for n in 1:N]

        for k in ks
            # Measure ⟨ã†_k ã_k⟩
            nk = 0.0 + 0.0im
            for m in 1:N, n in 1:N
                phase = exp(2π * im * Float64(k) * (m - n) / N) / N
                nk += phase * dot(ψ_gs, a_ops[m][2] * a_ops[n][1] * ψ_gs)
            end
            nk_measured = real(nk)

            # Bogoliubov prediction: tilde n_k = sin²(varphi_bogo)
            varphi_bogo = bogoliubov_angle(Float64(k), θ, N)
            nk_pred = sin(varphi_bogo)^2

            @test nk_measured ≈ nk_pred atol=1e-10
        end
    end

    @testset "Diagnostic scripts use canonical JW convention" begin
        include(joinpath(@__DIR__, "..", "scripts", "diagnostics", "verify_sigma_z_sign.jl"))
        include(joinpath(@__DIR__, "..", "scripts", "diagnostics", "verify_nk_correct_basis.jl"))

        for θ in (0.2, 0.4, 0.7)
            result = verify_sign(N=4, theta=θ, verbose=false)
            wrapper_result = verify_nk(N=4, theta=θ, verbose=false)

            @test result.sigma_z_error < 1e-12
            @test result.max_canonical_error < 1e-10
            @test result.obsolete_error > 1e-3
            @test wrapper_result.canonical_error ≈ result.canonical_error atol=1e-12
            @test wrapper_result.max_canonical_error ≈ result.max_canonical_error atol=1e-12
        end
    end

    @testset "JW-built fermionic H matches H_notes (N=$N)" for N in [4, 6]
        θ = 0.4  # arbitrary angle

        # Build H_notes directly from Pauli operators with PBC
        H_notes_direct = _test_build_H_notes(N, θ, :periodic)

        dim = 2^N
        II = Matrix{ComplexF64}(I, dim, dim)
        a_ops = [jordan_wigner_transform_complex(n, N) for n in 1:N]

        # Build H_notes from JW operators using the notes' formula.
        # IMPORTANT: The boundary term σ_{N,x} σ_{1,x} acquires an extra -P factor
        # (parity operator) from the JW string wrapping around the chain.
        # Bulk (n=1..N-1): σ_{n,x}σ_{n+1,x} = (a_n - a_n†)(a_{n+1} + a_{n+1}†)
        # Boundary (n=N):  σ_{N,x}σ_{1,x} = -P · (a_N - a_N†)(a_1 + a_1†)

        H_jw = zeros(ComplexF64, dim, dim)

        # Bulk hopping + pairing
        for n in 1:N-1
            an, adn = a_ops[n]
            anp1, adnp1 = a_ops[n+1]
            H_jw .+= (cos(θ)/2) * (an*anp1 + an*adnp1 - adn*anp1 - adn*adnp1)
        end

        # Boundary term with parity insertion
        Z = ComplexF64[1 0; 0 -1]
        Pz = foldl(*, [_test_site_op(Z, i, N) for i in 1:N])
        aN, adN = a_ops[N]
        a1, ad1 = a_ops[1]
        H_jw .+= (cos(θ)/2) * (-Pz) * ((aN - adN) * (a1 + ad1))

        # On-site: (sinθ/2) Σ σ_z = (sinθ/2) Σ (2a†a - 1)
        for n in 1:N
            an, adn = a_ops[n]
            H_jw .+= (sin(θ)/2) * (2*adn*an - II)
        end

        @test norm(H_notes_direct - H_jw) < 1e-10
    end

    @testset "Parity operator Px commutes with H_code" begin
        for N in [4, 6]
            J, h = 1.0, 0.5
            H = _test_build_H_code(N, J, h, :periodic)
            Px = _test_parity_x(N)
            @test norm(H * Px - Px * H) < 1e-10
        end
    end

    @testset "Parity operator Pz commutes with H_notes" begin
        for N in [4, 6]
            θ = 0.4
            H = _test_build_H_notes(N, θ, :periodic)
            Pz = _test_parity_z(N)
            @test norm(H * Pz - Pz * H) < 1e-10
        end
    end

    @testset "Mode energy symmetry" begin
        θ = 0.3; N = 6
        # ε_k = ε_{-k} (time-reversal symmetry)
        for k in allowed_k_indices(N, -1)
            @test mode_energy(Float64(k), θ, N) ≈ mode_energy(Float64(-k), θ, N) atol=1e-15
        end
    end

    @testset "Special mode energies" begin
        θ = 0.4; N = 4
        # k=0: w_0 = sinθ - cosθ, ε_0 = |w_0|
        w0 = w_k_coefficient(0.0, θ, N)
        @test w0 ≈ sin(θ) - cos(θ) atol=1e-15
        @test mode_energy(0.0, θ, N) ≈ abs(w0) atol=1e-15

        # k=N/2: w_{N/2} = sinθ + cosθ, ε_{N/2} = |w_{N/2}|
        wNh = w_k_coefficient(Float64(N÷2), θ, N)
        @test wNh ≈ sin(θ) + cos(θ) atol=1e-15
        @test mode_energy(Float64(N÷2), θ, N) ≈ abs(wNh) atol=1e-15
    end

    @testset "Code-unit positive gap grid" begin
        N = 6
        J, h = 1.0, 0.5
        for gF in [-1, 1]
            ks = allowed_k_indices(N, gF)
            gaps = mode_energies_Jh(ks, J, h, N)
            @test gaps ≈ [mode_energy_Jh(Float64(k), J, h, N) for k in ks] atol=1e-15
            @test all(>=(0), gaps)
        end
    end

    @testset "Bogoliubov angle for special modes" begin
        θ = 0.4; N = 4
        # For k=0 and k=N/2, r_k = 0, so varphi_bogo = 0
        @test bogoliubov_angle(0.0, θ, N) == 0.0
        @test bogoliubov_angle(Float64(N÷2), θ, N) == 0.0
    end

    @testset "coeff_k consistency" begin
        θ = 0.4; N = 6
        # For generic modes: coeff_k = ε_k
        for k in allowed_k_indices(N, -1)
            @test coeff_k(Float64(k), θ, N) ≈ mode_energy(Float64(k), θ, N) atol=1e-15
        end

        # For special modes (gF=+1): coeff_k = w_k (signed)
        @test coeff_k(0.0, θ, N) ≈ w_k_coefficient(0.0, θ, N) atol=1e-15
        @test coeff_k(Float64(N÷2), θ, N) ≈ w_k_coefficient(Float64(N÷2), θ, N) atol=1e-15
    end

    @testset "Vacuum energy formula E_vac = -½ Σ coeff_k" begin
        θ = 0.4; N = 6
        for gF in [-1, 1]
            ks = allowed_k_indices(N, gF)
            E_vac_explicit = -sum(coeff_k(Float64(k), θ, N) for k in ks) / 2
            E_vac_func = vacuum_energy(N, θ, gF)
            @test E_vac_explicit ≈ E_vac_func atol=1e-15
        end
    end

    @testset "Plotting dispersion helpers follow canonical mode convention" begin
        N = 6
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)

        for (bc, gF) in [(:periodic, 1), (:antiperiodic, -1)]
            ks = allowed_k_indices(N, gF)
            expected_momenta = [2π * Float64(k) / N for k in ks]
            k_values = generate_k_values(N, bc)

            @test k_values ≈ expected_momenta atol=1e-15
            @test generate_k_values(N, gF) ≈ expected_momenta atol=1e-15

            dispersion = compute_energy_dispersion(k_values, J, h)
            expected_dispersion = [mode_energy_Jh(Float64(k), J, h, N) for k in ks]

            @test all(dispersion .>= 0)
            @test dispersion ≈ expected_dispersion atol=1e-15

            occupations = compute_bdg_reference_occupation(k_values, J, h)
            expected_occupations = map(ks) do k
                kf = Float64(k)
                if abs(sin(2π * kf / N)) < 1e-12
                    wk = w_k_coefficient(kf, θ, N)
                    abs(wk) < 1e-12 ? 0.5 : (wk < 0 ? 1.0 : 0.0)
                else
                    sin(bogoliubov_angle(kf, θ, N))^2
                end
            end

            @test occupations ≈ expected_occupations atol=1e-15
            @test compute_ground_state_occupation(k_values, J, h) ≈ occupations atol=1e-15
        end

        critical_k_values = generate_k_values(N, :periodic)
        critical_occupations = compute_bdg_reference_occupation(
            critical_k_values, 1.0, 1.0)
        k0_index = findfirst(k -> abs(k) < 1e-12, critical_k_values)

        @test k0_index !== nothing
        @test critical_occupations[k0_index] ≈ 0.5 atol=1e-15
    end

    @testset "BdG reference occupation is not always the sector ground state" begin
        N = 4
        J, h = 1.0, 0.5
        ham_params = IsingParameters(N, J, h, :antiperiodic)
        H_code = _test_build_H_code(N, J, h, :antiperiodic)
        Px = _test_parity_x(N)

        _, ψ_even = _test_gs_in_sector(H_code, Px, 1)
        state_even = CoolingTNS.EDStateVector(ψ_even, N)
        gF_even = fermionic_bc(:antiperiodic, 1)

        k_values, sector_occupation = measure_raw_fourier_occupation_ed(
            state_even, ham_params; gF=gF_even)
        bdg_reference = compute_bdg_reference_occupation(k_values, J, h)

        @test gF_even == 1
        @test maximum(abs.(sector_occupation .- bdg_reference)) > 0.9
        @test compute_ground_state_occupation(k_values, J, h) ≈ bdg_reference atol=1e-15
    end

    @testset "Energy reconstruction from mode h_k" begin
        N = 6
        J, h = 1.0, 0.5
        ham_params = IsingParameters(N, J, h, :periodic)
        gF = fermionic_bc(:periodic, 1)
        ks = allowed_k_indices(N, gF)

        coefficients = ising_mode_energy_contribution_coefficients(ks, ham_params)
        @test coefficients ≈
            ising_mode_energy_contribution_coefficients(ks, N, J, h) atol=1e-12
        positive_gaps = mode_energies_Jh(ks, J, h, N)
        @test positive_gaps ≈ 2 .* abs.(coefficients) atol=1e-12

        special_ks = allowed_k_indices(N, 1)
        special_coefficients = ising_mode_energy_contribution_coefficients(special_ks, ham_params)
        special_positive_gaps = mode_energies_Jh(special_ks, J, h, N)
        @test special_positive_gaps ≈ 2 .* abs.(special_coefficients) atol=1e-12
        @test any(<(0), special_coefficients)

        hk_vac = fill(-1.0, length(ks))
        E_vac = ising_energy_from_mode_hk(ks, hk_vac, ham_params)
        @test E_vac ≈ vacuum_energy_Jh(N, J, h, gF) atol=1e-12
        @test E_vac ≈ sum(coefficients .* hk_vac) atol=1e-12

        hk_matrix = [hk_vac'; (hk_vac .+ 0.1)']
        E_steps = ising_energy_from_mode_hk(ks, hk_matrix, ham_params)
        @test length(E_steps) == 2
        @test E_steps[1] ≈ E_vac atol=1e-12
        @test E_steps[2] ≈ ising_energy_from_mode_hk(ks, hk_matrix[2, :], ham_params) atol=1e-12

        @test_throws ArgumentError ising_energy_from_mode_hk(ks[1:end-1], hk_vac, ham_params)
        @test_throws ArgumentError ising_energy_from_mode_hk(ks, reshape(hk_vac, :, 1), ham_params)
    end

    @testset "Antiperiodic BC spectrum (N=$N)" for N in [4, 6]
        J, h = 1.0, 0.5
        θ = theta_from_Jh(J, h)
        Λ = energy_scale(J, h)

        # Build H_code with APBC
        H_code = _test_build_H_code(N, J, h, :antiperiodic)
        Px = _test_parity_x(N)
        @test norm(H_code * Px - Px * H_code) < 1e-10

        # For spin APBC (gI=-1):
        # Px=+1 → gF = -(-1)*1 = +1 (fermionic PBC, integer k, with special modes)
        # Px=-1 → gF = -(-1)*(-1) = -1 (fermionic APBC, half-integer k, no special modes)
        gF_even = fermionic_bc(:antiperiodic, 1)  # = +1
        gF_odd = fermionic_bc(:antiperiodic, -1)   # = -1

        @test gF_even == 1
        @test gF_odd == -1

        # For spin APBC, the even (Px=+1) sector uses gF=+1 and therefore
        # contains the special modes. The chosen-operator vacuum has even
        # fermion number and is allowed in this sector. For the parameters
        # below, occupying only the negative-w0 special mode would violate
        # parity, while occupying both special modes raises the energy; hence
        # the sector ground state is the chosen-operator vacuum.
        E_gs_even, _ = _test_gs_in_sector(H_code, Px, 1)
        E_vac_even = vacuum_energy_Jh(N, J, h, gF_even)
        @test E_gs_even ≈ E_vac_even atol=1e-10

        # For gF=-1 (half-integer k, no special modes): vacuum has Nf=0 → even Px.
        # But this sector is Px=-1 (odd), which needs Nf=odd.
        # The cheapest single-quasiparticle excitation costs ε_min.
        # (Note: unlike special modes, generic modes always have ε_k > 0.)
        E_gs_odd, _ = _test_gs_in_sector(H_code, Px, -1)
        E_vac_odd = vacuum_energy_Jh(N, J, h, gF_odd)
        ε_modes_odd = [mode_energy_Jh(Float64(k), J, h, N)
                       for k in allowed_k_indices(N, gF_odd) if Float64(k) > 0]
        ε_min_odd = minimum(ε_modes_odd)
        # Odd-sector GS = vacuum + one quasiparticle
        @test E_gs_odd ≈ E_vac_odd + ε_min_odd atol=1e-10
    end

end
