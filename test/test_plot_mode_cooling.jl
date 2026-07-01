using Test
using CoolingTNS
using HDF5
using PythonCall

pyimport("matplotlib").use("Agg"; force=true)

include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_mode_cooling.jl"))

_mode_axis_lines(ax) = pyconvert(Vector, ax.lines)
_mode_axis_ylabel(ax) = pyconvert(String, ax.get_ylabel())
_mode_line_xdata(line) = pyconvert(Vector{Float64}, line.get_xdata())
_mode_line_ydata(line) = pyconvert(Vector{Float64}, line.get_ydata())
_mode_line_labels(ax) = [pyconvert(String, line.get_label()) for line in _mode_axis_lines(ax)]

@testset "Mode cooling plot data convention" begin
    mode_hk = [-1.0 0.0; 1.0 -0.5]
    mode_nk = mode_occupation_from_hk(mode_hk)

    @test _mode_occupation_from_plot_data(Dict{String, Any}(RESULT_MODE_HK => mode_hk)) ≈ mode_nk

    stored_nk = copy(mode_nk)
    data = Dict{String, Any}(RESULT_MODE_HK => mode_hk, RESULT_MODE_NK => stored_nk)
    @test _mode_occupation_from_plot_data(data) == stored_nk

    nan_hk = [-1.0 NaN; 0.0 1.0]
    nan_nk = mode_occupation_from_hk(nan_hk)
    @test isequal(_mode_occupation_from_plot_data(
        Dict{String, Any}(RESULT_MODE_HK => nan_hk, RESULT_MODE_NK => nan_nk)
    ), nan_nk)

    bad_shape = Dict{String, Any}(RESULT_MODE_HK => mode_hk, RESULT_MODE_NK => [0.0, 0.5])
    @test_throws DimensionMismatch _mode_occupation_from_plot_data(bad_shape)

    bad_nk = copy(mode_nk)
    bad_nk[1, 1] = 0.25
    @test_throws ArgumentError _mode_occupation_from_plot_data(
        Dict{String, Any}(RESULT_MODE_HK => mode_hk, RESULT_MODE_NK => bad_nk)
    )

    @test _occupation_ylim(mode_nk) == (-0.05, 1.05)
    @test _occupation_ylim([NaN]) == (-0.05, 1.05)

    @test_throws ErrorException _mode_occupation_from_plot_data(Dict{String, Any}())

    fig = plot_mode_occupation_from_data(
        stored_nk,
        [-1//2, 1//2],
        [0.8, 1.2];
        title="test",
    )
    @test occursin("Bogoliubov", _mode_axis_ylabel(fig.axes[0]))
    @test occursin("Bog", _mode_axis_ylabel(fig.axes[0]))
    @test occursin("Bogoliubov", _mode_axis_ylabel(fig.axes[1]))
    get_pyplot().close(fig)

    detuning_fig = plot_mode_occupation_from_data(
        stored_nk,
        [-1//2, 1//2],
        [0.8, 1.2];
        delta=1.1,
        title="detuning label test",
    )
    labels = _mode_line_labels(detuning_fig.axes[0])
    @test any(label -> occursin("nearest |Δ|", label), labels)
    @test all(label -> !occursin("resonant:", label), labels)
    get_pyplot().close(detuning_fig)
end

@testset "Mode cooling plots only measured strided cycles" begin
    mode_nk = [
        0.10 0.20
        NaN  NaN
        0.50 0.60
    ]
    k_indices = [-1//2, 1//2]
    εk_values = [0.8, 1.2]

    measured = _mode_measurement_cycle_rows(3, [0, 2])
    @test measured.cycles == [0, 2]
    @test measured.rows == [1, 3]
    @test _mode_measurement_cycle_rows(3).cycles == [0, 1, 2]
    @test_throws ArgumentError _mode_measurement_cycle_rows(3, Int[])
    @test_throws ArgumentError _mode_measurement_cycle_rows(3, [2, 0])
    @test_throws ArgumentError _mode_measurement_cycle_rows(3, [0, 0])
    @test_throws ArgumentError _mode_measurement_cycle_rows(3, [0, 3])

    fig = plot_mode_occupation_from_data(
        mode_nk,
        k_indices,
        εk_values;
        measurement_cycles=[0, 2],
    )
    first_line = first(_mode_axis_lines(fig.axes[0]))
    @test _mode_line_xdata(first_line) == [0.0, 2.0]
    @test _mode_line_ydata(first_line) == [0.10, 0.50]
    get_pyplot().close(fig)

    mktempdir() do dir
        filename = joinpath(dir, "strided_modes.h5")
        h5open(filename, "w") do file
            write(file, RESULT_MODE_NK, mode_nk)
            write(file, RESULT_MODE_K_INDICES, Float64.(k_indices))
            write(file, RESULT_MODE_ENERGIES, εk_values)
            write(file, RESULT_MODE_MEASUREMENT_CYCLES, [0, 2])
        end

        h5_fig = plot_mode_cooling_from_h5(filename)
        h5_first_line = first(_mode_axis_lines(h5_fig.axes[0]))
        @test _mode_line_xdata(h5_first_line) == [0.0, 2.0]
        @test _mode_line_ydata(h5_first_line) == [0.10, 0.50]
        get_pyplot().close(h5_fig)
    end
end

@testset "Mode cooling nearest-detuning convention" begin
    εk_values = [0.8, 1.2, 1.2, 1.7]

    @test bath_detuning_energy([-1.2]) == 1.2
    @test bath_detuning_energy([1.0, 1.2]) === nothing
    @test bath_detuning_energy(Float64[]) === nothing
    @test bath_detuning_energy("1.2") === nothing
    @test nearest_bath_detuning_indices(εk_values, -1.2) == [2, 3]
    @test nearest_bath_detuning_indices(εk_values, nothing) == Int[]
    @test nearest_bath_detuning_indices(εk_values, 0.0) == Int[]
    @test nearest_bath_detuning_indices([1.0, 1.0 + 5e-13, 1.5], 1.0) == [1, 2]
    @test nearest_bath_resonance_indices(εk_values, -1.2) ==
        nearest_bath_detuning_indices(εk_values, -1.2)
end
