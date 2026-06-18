using Test
using HDF5

include(joinpath(@__DIR__, "..", "scripts", "validation",
                 "summarize_largeN_bond_dimensions.jl"))

@testset "Large-N bond-dimension summary script" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 12)
            gn = create_group(f, "N4")
            write(gn, "N", 4)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R2")

            write(gr, "M", 2)
            write(gr, "E_mean", [-1.0, 2.0, 4.0])
            write(gr, "relative_energy_mean", [0.0, 1.0, 2.0])
            write(gr, "system_max_bond", [1 1; 6 9; 12 10])
            write(gr, "system_mean_bond", [1.0 1.0; 4.0 5.0; 8.0 5.0])
            write(gr, "evolved_max_bond", [0 0; 12 6; 8 14])
            write(gr, "evolved_mean_bond", [NaN NaN; 8.0 7.0; 9.0 11.0])

            bd = create_group(gr, "final_bond_dims")
            write(bd, "trajectory_1", [4, 8, 12])
            write(bd, "trajectory_2", [12, 12, 12])
        end

        rows = summarize_file(path)
        @test length(rows) == 1
        row = only(rows)
        @test row.N == 4
        @test row.method == "mcwf"
        @test row.R == 2
        @test row.M == 2
        @test row.threshold == 12
        @test row.final_e_over_n == 1.0
        @test row.relative_energy == 2.0
        @test row.final_system_max == 12
        @test row.final_system_mean == 6.5
        @test row.peak_evolved_max == 14
        @test row.peak_evolved_mean == 10.0
        @test row.system_saturation_cycle == 2
        @test row.evolved_saturation_cycle == 1
        @test row.q50 == 10.0
        @test row.q75 == 11.0
        @test row.q90 ≈ 11.6
        @test row.frac50 ≈ 5 / 6
        @test row.frac75 ≈ 2 / 3
        @test row.frac90 ≈ 2 / 3
    finally
        rm(path; force=true)
    end
end
