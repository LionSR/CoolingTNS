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
            write(gm, "detuning_protocol_source", "fixed_range")
            write(gm, "detuning_delta_min", 0.5)
            write(gm, "detuning_delta_max", 3.0)
            write(gm, "detuning_delta_max_factor", NaN)
            gr = create_group(gm, "R2")

            write(gr, "M", 2)
            write(gr, "E_mean", [-1.0, 2.0, 4.0])
            write(gr, "relative_energy_mean", [0.0, 1.0, 2.0])
            write(gr, "system_max_bond", [1 1; 6 9; 12 10])
            write(gr, "system_mean_bond", [1.0 1.0; 4.0 5.0; 8.0 5.0])
            write(gr, "evolved_max_bond", [0 0; 12 6; 8 14])
            write(gr, "evolved_mean_bond", [NaN NaN; 8.0 7.0; 9.0 11.0])
            write(gr, "delta_values", [0.5, 3.0])

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
        @test row.delta_protocol == "fixed_range"
        @test row.delta_range == "[0.50000000,3.00000000]"
        @test row.delta_factor == "n/a"
        @test row.threshold == 12
        @test row.final_e_over_n == 1.0
        @test row.relative_energy == 2.0
        @test row.best_e_over_n == -0.25
        @test row.best_relative_energy == 0.0
        @test row.tail_e_over_n ≈ 5 / 12
        @test row.tail_relative_energy == 1.0
        @test row.tail_count == 3
        @test row.system_effective_bond == ">=12"
        @test row.evolved_effective_bond == ">=14"
        @test row.bond_status == "not_converged_system_and_evolved_cap"
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

        output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_markdown(rows)
                end
            end
            read(output_path, String)
        end
        @test occursin(
            "| file | N | method | R | M | delta_protocol | delta_range | delta_factor | Dcap |",
            output,
        )
        @test occursin("| final E/N | relE | best E/N | best relE | tail E/N |", output)
        @test occursin(
            "| $(basename(path)) | 4 | mcwf | 2 | 2 | fixed_range | " *
            "[0.50000000,3.00000000] | n/a | 12 | >=12 | >=14 | " *
            "not_converged_system_and_evolved_cap | 1.00000000 | 2.00000 | " *
            "-0.25000000 | 0.00000 | 0.41666667 | 1.00000 | 3 |",
            output,
        )
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary handles missing detuning metadata" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 4)
            gn = create_group(f, "N2")
            write(gn, "N", 2)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [-1.0, 0.0])
            write(gr, "relative_energy_mean", [0.0, 1.0])
            write(gr, "system_max_bond", [1, 2])
            write(gr, "system_mean_bond", [1.0, 2.0])
            write(gr, "evolved_max_bond", [0, 4])
            write(gr, "evolved_mean_bond", [NaN, 4.0])
        end

        row = only(summarize_file(path))
        @test row.delta_protocol == "unknown"
        @test row.delta_range == "unknown"
        @test row.delta_factor == "unknown"
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary reports gap-scaled detuning metadata" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 4)
            gn = create_group(f, "N2")
            write(gn, "N", 2)
            gm = create_group(gn, "mcwf")
            write(gm, "detuning_protocol_source", "gap_scaled_range")
            write(gm, "detuning_delta_min", 0.75)
            write(gm, "detuning_delta_max", 3.0)
            write(gm, "detuning_delta_max_factor", 4.0)
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [-1.0, 0.0])
            write(gr, "relative_energy_mean", [0.0, 1.0])
            write(gr, "system_max_bond", [1, 2])
            write(gr, "system_mean_bond", [1.0, 2.0])
            write(gr, "evolved_max_bond", [0, 4])
            write(gr, "evolved_mean_bond", [NaN, 4.0])
        end

        row = only(summarize_file(path))
        @test row.delta_protocol == "gap_scaled_range"
        @test row.delta_range == "[0.75000000,3.00000000]"
        @test row.delta_factor == "4.000"
    finally
        rm(path; force=true)
    end
end
