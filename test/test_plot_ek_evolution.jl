using Test
using CoolingTNS

include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_ek_evolution.jl"))

@testset "Mode energy plot convention" begin
    N = 4
    J = 1.0
    h = 0.5
    ham_params = IsingParameters(N, J, h, :periodic)
    k_indices = allowed_k_indices(N, -1)
    mode_hk = [
        -1.0 -1.0 -1.0 -1.0
        -0.5  0.0  0.5  1.0
    ]

    contributions = _mode_energy_contributions(mode_hk, k_indices, N, J, h)
    reconstructed = ising_energy_from_mode_hk(k_indices, mode_hk, ham_params)

    @test vec(sum(contributions; dims=2)) ≈ reconstructed atol=1e-12
    @test _mode_phase_over_pi(k_indices, N) == [2 * Float64(k) / N for k in k_indices]

    @test _mode_matrix_steps_by_modes(mode_hk, length(k_indices), RESULT_MODE_HK; n_steps=2) == mode_hk
    @test_throws DimensionMismatch _mode_matrix_steps_by_modes(
        transpose(mode_hk), length(k_indices), RESULT_MODE_HK; n_steps=2)
    @test_throws DimensionMismatch _mode_matrix_steps_by_modes(ones(2, 3), length(k_indices), RESULT_MODE_HK)
    @test_logs (:warn, r"canonical steps-by-modes") begin
        _mode_matrix_steps_by_modes(ones(length(k_indices), length(k_indices)),
                                    length(k_indices), RESULT_MODE_HK; n_steps=length(k_indices))
    end
    @test_logs (:warn, r"canonical steps-by-modes") begin
        _mode_matrix_steps_by_modes(ones(length(k_indices), length(k_indices)),
                                    length(k_indices), RESULT_MODE_HK)
    end
end

@testset "Mode energy plot covers signed special modes" begin
    N = 4
    J = 1.0
    h = 0.5
    ham_params = IsingParameters(N, J, h, :periodic)
    k_indices = allowed_k_indices(N, 1)
    mode_hk = [
        -1.0 -1.0 -1.0 -1.0
         1.0 -1.0  0.0  1.0
    ]

    contributions = _mode_energy_contributions(mode_hk, k_indices, N, J, h; n_steps=2)
    reconstructed = ising_energy_from_mode_hk(k_indices, mode_hk, ham_params)
    coeffs = _mode_energy_coefficients(k_indices, N, J, h)

    @test 0 in k_indices
    @test N ÷ 2 in k_indices
    @test any(c -> c < 0, coeffs)
    @test vec(sum(contributions; dims=2)) ≈ reconstructed atol=1e-12
end

@testset "Mode energy plot validates stored positive energies" begin
    N = 4
    J = 1.0
    h = 0.5
    k_indices = allowed_k_indices(N, 1)
    coeffs = _mode_energy_coefficients(k_indices, N, J, h)
    stored_εk = abs.(2 .* coeffs)

    @test _checked_mode_energy_coefficients(stored_εk, k_indices, N, J, h) ≈ coeffs
    @test_logs (:warn, r"Stored mode energies differ") begin
        _checked_mode_energy_coefficients(fill(0.0, length(k_indices)), k_indices, N, J, h)
    end
    @test_throws DimensionMismatch _checked_mode_energy_coefficients(
        stored_εk[1:end-1], k_indices, N, J, h)
end

@testset "Mode energy plot refuses Fourier occupations as energies" begin
    data = Dict{String, Any}(
        RESULT_MOMENTUM_DISTRIBUTION => ones(2, 4),
        RESULT_K_VALUES => collect(range(-pi, pi; length=4)),
    )

    @test !_has_mode_energy_data(data)
    @test_logs (:warn, r"Not plotting epsilon_k\*n_k as an energy") begin
        _warn_missing_mode_energy_data("example.h5", data)
    end
end
