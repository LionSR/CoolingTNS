using Test
using CoolingTNS
using HDF5
using PythonCall

pyimport("matplotlib").use("Agg"; force=true)

include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_ek_evolution.jl"))
include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_nk_evolution.jl"))

_ek_axis_lines(ax) = pyconvert(Vector, ax.lines)
_ek_line_label(line) = pyconvert(String, line.get_label())
_ek_line_ydata(line) = pyconvert(Vector{Float64}, line.get_ydata())

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

@testset "Mode energy plot validates stored positive gaps" begin
    N = 4
    J = 1.0
    h = 0.5
    k_indices = allowed_k_indices(N, 1)
    coeffs = _mode_energy_coefficients(k_indices, N, J, h)
    stored_εk = abs.(2 .* coeffs)

    @test _checked_mode_energy_coefficients(stored_εk, k_indices, N, J, h) ≈ coeffs
    @test_logs (:warn, r"Stored positive quasiparticle gaps differ") begin
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
    @test_logs (:warn, r"Not plotting epsilon_k\*tilde_n_k as an energy") begin
        _warn_missing_mode_energy_data("example.h5", data)
    end
end

@testset "Evolution step selection stays in bounds" begin
    @test select_evolution_steps(1) == [1]
    @test select_evolution_steps(2) == [1, 2]
    @test select_evolution_steps(4) == [1, 2, 3, 4]
    @test select_evolution_steps(8) == [1, 2, 4, 6, 8]

    @test select_evolution_steps(3; steps_to_plot=[3, 1]) == [3, 1]
    @test_throws ArgumentError select_evolution_steps(0)
    @test_throws ArgumentError select_evolution_steps(3; steps_to_plot=[0, 1])
    @test_throws ArgumentError select_evolution_steps(3; steps_to_plot=[1, 4])
    @test length(get_evolution_colors(get_pyplot(), 1)) == 1
    @test_throws ArgumentError get_evolution_colors(get_pyplot(), 0)
end

@testset "Momentum distribution orientation helper" begin
    modes_by_steps = [
        0.10 0.20
        0.30 0.40
        0.50 0.60
    ]
    steps_by_modes = permutedims(modes_by_steps)
    square_steps_by_modes = [
        0.1 0.2
        0.3 0.4
    ]

    @test _momentum_distribution_modes_by_steps(modes_by_steps, 3) == modes_by_steps
    @test _momentum_distribution_modes_by_steps(steps_by_modes, 3) == modes_by_steps
    @test _momentum_distribution_modes_by_steps(square_steps_by_modes, 2) == permutedims(square_steps_by_modes)
    @test_throws DimensionMismatch _momentum_distribution_modes_by_steps(ones(2, 2), 3)
end

@testset "K-space evolution plot colors use Julia indexing" begin
    mktempdir() do dir
        filename = joinpath(dir, "two_step_modes.h5")
        N = 4
        J = 1.0
        h = 0.5
        k_values = collect(range(0.0, 2π; length=N))
        k_indices = allowed_k_indices(N, -1)
        coeffs = _mode_energy_coefficients(k_indices, N, J, h)
        mode_hk = [
            -1.0 -0.5  0.0  0.5
            -0.8 -0.4  0.1  0.6
        ]

        h5open(filename, "w") do file
            write(file, RESULT_MOMENTUM_DISTRIBUTION, [
                0.10 0.20
                0.30 0.40
                0.50 0.60
                0.70 0.80
            ])
            write(file, RESULT_K_VALUES, k_values)
            write(file, RESULT_MODE_HK, mode_hk)
            write(file, RESULT_MODE_K_INDICES, Float64.(k_indices))
            write(file, RESULT_MODE_ENERGIES, abs.(2 .* coeffs))
            write(file, RESULT_ENERGY, [-1.0, -0.8])
            write(file, "N", N)
            write(file, "J", J)
            write(file, "h", h)
            write(file, "bc", "periodic")
        end

        fig_nk = plot_nk_evolution(filename; steps_to_plot=[1, 2], save_fig=false)
        fig_ek = plot_ek_evolution(filename; steps_to_plot=[1, 2], save_fig=false)

        @test fig_nk !== nothing
        @test fig_ek !== nothing
        @test occursin("tilde", pyconvert(String, fig_nk.axes[0].get_ylabel()))

        plt = get_pyplot()
        plt.close(fig_nk)
        plt.close(fig_ek)
    end
end

@testset "Mode energy plot respects strided measurement cycles" begin
    mktempdir() do dir
        filename = joinpath(dir, "strided_mode_energy.h5")
        N = 4
        J = 1.0
        h = 0.5
        k_indices = allowed_k_indices(N, -1)
        coeffs = _mode_energy_coefficients(k_indices, N, J, h)
        mode_hk = [
            -1.0 -0.5  0.0  0.5
             NaN  NaN  NaN  NaN
            -0.8 -0.4  0.1  0.6
        ]

        h5open(filename, "w") do file
            write(file, RESULT_MODE_HK, mode_hk)
            write(file, RESULT_MODE_K_INDICES, Float64.(k_indices))
            write(file, RESULT_MODE_ENERGIES, abs.(2 .* coeffs))
            write(file, RESULT_MODE_MEASUREMENT_CYCLES, [0, 2])
            write(file, RESULT_ENERGY, [-1.0, -0.9, -0.8])
            write(file, "N", N)
            write(file, "J", J)
            write(file, "h", h)
            write(file, "bc", "periodic")
        end

        fig = plot_ek_evolution(filename; save_fig=false)
        labels = _ek_line_label.(_ek_axis_lines(fig.axes[0]))
        @test "Initial" in labels
        @test "Cycle 2" in labels
        @test !("Cycle 1" in labels)

        plotted_ydata = _ek_line_ydata.(_ek_axis_lines(fig.axes[0])[1:2])
        @test all(values -> all(isfinite, values), plotted_ydata)

        get_pyplot().close(fig)

        selected_fig = plot_ek_evolution(filename; steps_to_plot=[2], save_fig=false)
        selected_labels = _ek_line_label.(_ek_axis_lines(selected_fig.axes[0]))
        @test "Cycle 2" in selected_labels
        @test !("Initial" in selected_labels)
        get_pyplot().close(selected_fig)
    end
end
