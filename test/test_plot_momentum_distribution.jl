using Test
using HDF5
using PythonCall

pyimport("matplotlib").use("Agg"; force=true)

include(joinpath(@__DIR__, "..", "scripts", "plotting", "plotting.jl"))

function _write_momentum_distribution_file(
    path;
    k_values,
    momentum_dist,
    delta=nothing,
    mode_energies=nothing,
    ham_name=nothing,
    parsed_args=nothing,
)
    h5open(path, "w") do file
        write(file, RESULT_MOMENTUM_DISTRIBUTION, momentum_dist)
        write(file, RESULT_K_VALUES, k_values)
        delta !== nothing && write(file, "delta", delta)
        mode_energies !== nothing && write(file, RESULT_MODE_ENERGIES, mode_energies)
        ham_name !== nothing && write(file, "ham_name", ham_name)

        if parsed_args !== nothing
            group = create_group(file, HDF5_PARSED_ARGS_GROUP)
            for (key, value) in parsed_args
                write(group, string(key), value)
            end
        end
    end

    return path
end

function _momentum_axis_lines(ax)
    return pyconvert(Vector, ax.lines)
end

function _momentum_line_xdata(line)
    return pyconvert(Vector{Float64}, line.get_xdata())
end

function _momentum_line_ydata(line)
    return pyconvert(Vector{Float64}, line.get_ydata())
end

function _momentum_line_label(line)
    return pyconvert(String, line.get_label())
end

_momentum_is_constant_at(values, target; atol=1e-12) =
    length(values) >= 2 && all(v -> isapprox(v, target; atol=atol, rtol=0), values)

function _momentum_has_vertical_line_at(ax, x)
    return any(line -> _momentum_is_constant_at(_momentum_line_xdata(line), x),
               _momentum_axis_lines(ax))
end

function _momentum_has_horizontal_line_at(ax, y)
    return any(line -> _momentum_is_constant_at(_momentum_line_ydata(line), y),
               _momentum_axis_lines(ax))
end

function _momentum_has_label_containing(ax, text)
    return any(line -> occursin(text, _momentum_line_label(line)), _momentum_axis_lines(ax))
end

@testset "Momentum distribution detuning markers use momentum axis" begin
    plt = get_pyplot()

    mktempdir() do dir
        k_values = [-pi / 2, 0.0, pi / 2, pi]
        momentum_dist = [0.10 0.20 0.30 0.40;
                         0.15 0.25 0.35 0.45;
                         0.20 0.30 0.40 0.50;
                         0.25 0.35 0.45 0.55]
        delta = -1.2

        stored_energy_file = _write_momentum_distribution_file(
            joinpath(dir, "stored_mode_energies.h5");
            k_values=k_values,
            momentum_dist=momentum_dist,
            delta=delta,
            mode_energies=[0.8, 1.2, 1.2, 1.7],
        )

        fig = plot_momentum_distribution(stored_energy_file; save_fig=false)
        ax = fig.axes[0]
        @test length(pyconvert(Vector, fig.axes)) == 1
        @test _momentum_has_vertical_line_at(ax, 0.0)
        @test _momentum_has_vertical_line_at(ax, pi / 2)
        @test !_momentum_has_horizontal_line_at(ax, delta)
        @test _momentum_has_label_containing(ax, "1.2")
        plt.close(fig)

        N = 4
        J = 1.0
        h = 0.5
        bc = :periodic
        canonical_k_values = CoolingTNS.generate_k_values(N, bc)
        canonical_energies = CoolingTNS.compute_energy_dispersion(canonical_k_values, J, h)
        target_index = argmin(abs.(canonical_energies .- canonical_energies[2]))

        parsed_args_file = _write_momentum_distribution_file(
            joinpath(dir, "parsed_args_ising.h5");
            k_values=canonical_k_values,
            momentum_dist=repeat(reshape(collect(range(0.1, 0.4; length=N)), 1, :), 4, 1),
            delta=canonical_energies[2],
            parsed_args=Dict("problem" => "ising", "J" => J, "h" => h),
        )

        fig_from_parsed_args = plot_momentum_distribution(parsed_args_file; save_fig=false)
        ax_from_parsed_args = fig_from_parsed_args.axes[0]
        @test length(pyconvert(Vector, fig_from_parsed_args.axes)) == 1
        @test _momentum_has_vertical_line_at(ax_from_parsed_args, canonical_k_values[target_index])
        @test !_momentum_has_horizontal_line_at(ax_from_parsed_args, canonical_energies[2])
        plt.close(fig_from_parsed_args)

        no_energy_file = _write_momentum_distribution_file(
            joinpath(dir, "no_energy_source.h5");
            k_values=k_values,
            momentum_dist=momentum_dist,
            delta=delta,
        )

        fig_no_marker = plot_momentum_distribution(no_energy_file; save_fig=false)
        ax_no_marker = fig_no_marker.axes[0]
        @test length(pyconvert(Vector, fig_no_marker.axes)) == 1
        @test !_momentum_has_vertical_line_at(ax_no_marker, 0.0)
        @test !_momentum_has_horizontal_line_at(ax_no_marker, delta)
        plt.close(fig_no_marker)
    end
end
