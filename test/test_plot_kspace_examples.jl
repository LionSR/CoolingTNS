using Test
using CoolingTNS
using HDF5

include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_nk_evolution.jl"))

@testset "K-space plotting examples" begin
    mktempdir() do dir
        filename = joinpath(dir, "ed_dm_with_metadata_group.h5")
        h5open(filename, "w") do file
            write(file, RESULT_MOMENTUM_DISTRIBUTION, [0.1 0.3; 0.2 0.4])
            write(file, RESULT_K_VALUES, [0.0, pi])
            write(file, "N", 2)
            write(file, "J", 1.0)
            write(file, "h", 0.5)
            write(file, "bc", "periodic")

            parsed_args = create_group(file, CoolingTNS.HDF5_PARSED_ARGS_GROUP)
            write(parsed_args, "backend", "ED")
            write(parsed_args, "sim_method", "density_matrix")
        end

        fig = plot_nk_evolution(filename; save_fig=false)
        @test fig !== nothing
        get_pyplot().close(fig)
    end
end
