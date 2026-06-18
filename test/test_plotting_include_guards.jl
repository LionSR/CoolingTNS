using Test

function include_plotting_helpers_twice_stderr()
    plotutils_path = joinpath(@__DIR__, "..", "scripts", "plotting", "PlotUtils.jl")
    plotting_path = joinpath(@__DIR__, "..", "scripts", "plotting", "plotting.jl")
    loaded_definitions = Ref((get_pyplot=false, plot_data=false))

    stderr_text = mktemp() do stderr_path, io
        close(io)
        open(stderr_path, "w") do stderr_io
            redirect_stderr(stderr_io) do
                probe = Module(:PlottingIncludeGuardProbe)
                Base.include(probe, plotutils_path)
                Base.include(probe, plotutils_path)
                Base.include(probe, plotting_path)
                Base.include(probe, plotting_path)
                loaded_definitions[] = (
                    get_pyplot=isdefined(probe, :get_pyplot),
                    plot_data=isdefined(probe, :plot_data),
                )
            end
        end
        return read(stderr_path, String)
    end

    return (
        stderr=stderr_text,
        get_pyplot_loaded=loaded_definitions[].get_pyplot,
        plot_data_loaded=loaded_definitions[].plot_data,
    )
end

@testset "Plotting helper include guards" begin
    result = include_plotting_helpers_twice_stderr()
    @test result.get_pyplot_loaded
    @test result.plot_data_loaded
    @test !occursin("overwritten on the same line", result.stderr)
    @test !occursin("check for duplicate calls to `include`", result.stderr)
end
