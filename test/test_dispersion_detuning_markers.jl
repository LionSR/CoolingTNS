using Test
using PythonCall

include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_energy_dispersion.jl"))
include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_dispersion_with_gs.jl"))

function _axis_lines(ax)
    return pyconvert(Vector, ax.lines)
end

function _line_xdata(line)
    return pyconvert(Vector{Float64}, line.get_xdata())
end

function _line_ydata(line)
    return pyconvert(Vector{Float64}, line.get_ydata())
end

_is_constant_at(values, target; atol=1e-12) =
    length(values) >= 2 && all(v -> isapprox(v, target; atol=atol, rtol=0), values)

function _has_horizontal_line_at(ax, y)
    return any(line -> _is_constant_at(_line_ydata(line), y), _axis_lines(ax))
end

function _has_vertical_line_at(ax, x)
    return any(line -> _is_constant_at(_line_xdata(line), x), _axis_lines(ax))
end

@testset "Dispersion detuning markers use energy axis" begin
    delta = -0.7
    δ_abs = abs(delta)

    fig = plot_energy_dispersion(4, 1.0, 0.5, :periodic; delta=delta, save_fig=false)
    ax = fig.axes[0]
    @test _has_horizontal_line_at(ax, δ_abs)
    @test !_has_vertical_line_at(ax, delta / pi)

    fig_gs = plot_dispersion_with_ground_state(4, 1.0, 0.5, :periodic; delta=delta, save_fig=false)
    energy_ax = fig_gs.axes[0]
    occupation_ax = fig_gs.axes[1]
    @test _has_horizontal_line_at(energy_ax, δ_abs)
    @test !_has_vertical_line_at(energy_ax, delta / pi)
    @test !_has_horizontal_line_at(occupation_ax, δ_abs)
end
