using Test

function include_plotting_helpers_twice_stderr()
    plotutils_path = joinpath(@__DIR__, "..", "scripts", "plotting", "PlotUtils.jl")
    plotting_path = joinpath(@__DIR__, "..", "scripts", "plotting", "plotting.jl")
    library_script_paths = [
        joinpath(@__DIR__, "..", "scripts", "plotting", "plot_ek_evolution.jl"),
        joinpath(@__DIR__, "..", "scripts", "plotting", "plot_nk_evolution.jl"),
        joinpath(@__DIR__, "..", "scripts", "plotting", "plot_energy_dispersion.jl"),
        joinpath(@__DIR__, "..", "scripts", "plotting", "plot_dispersion_with_gs.jl"),
        joinpath(@__DIR__, "..", "scripts", "plotting", "plot_mode_cooling.jl"),
    ]
    loaded_definitions = Ref((
        get_pyplot=false,
        plot_data=false,
        plot_ek=false,
        plot_nk=false,
        plot_energy=false,
        plot_dispersion=false,
        plot_modes=false,
    ))

    stderr_text = mktemp() do stderr_path, io
        close(io)
        open(stderr_path, "w") do stderr_io
            redirect_stderr(stderr_io) do
                probe = Module(:PlottingIncludeGuardProbe)
                Base.include(probe, plotutils_path)
                Base.include(probe, plotutils_path)
                Base.include(probe, plotting_path)
                Base.include(probe, plotting_path)
                for script_path in library_script_paths
                    Base.include(probe, script_path)
                    Base.include(probe, script_path)
                end
                loaded_definitions[] = (
                    get_pyplot=isdefined(probe, :get_pyplot),
                    plot_data=isdefined(probe, :plot_data),
                    plot_ek=isdefined(probe, :plot_ek_evolution),
                    plot_nk=isdefined(probe, :plot_nk_evolution),
                    plot_energy=isdefined(probe, :plot_energy_dispersion),
                    plot_dispersion=isdefined(probe, :plot_dispersion_with_ground_state),
                    plot_modes=isdefined(probe, :plot_mode_occupation_from_data),
                )
            end
        end
        return read(stderr_path, String)
    end

    return (
        stderr=stderr_text,
        get_pyplot_loaded=loaded_definitions[].get_pyplot,
        plot_data_loaded=loaded_definitions[].plot_data,
        plot_ek_loaded=loaded_definitions[].plot_ek,
        plot_nk_loaded=loaded_definitions[].plot_nk,
        plot_energy_loaded=loaded_definitions[].plot_energy,
        plot_dispersion_loaded=loaded_definitions[].plot_dispersion,
        plot_modes_loaded=loaded_definitions[].plot_modes,
    )
end

@testset "Plotting helper include guards" begin
    result = include_plotting_helpers_twice_stderr()
    @test result.get_pyplot_loaded
    @test result.plot_data_loaded
    @test result.plot_ek_loaded
    @test result.plot_nk_loaded
    @test result.plot_energy_loaded
    @test result.plot_dispersion_loaded
    @test result.plot_modes_loaded
    @test !occursin("overwritten on the same line", result.stderr)
    @test !occursin("check for duplicate calls to `include`", result.stderr)
end
