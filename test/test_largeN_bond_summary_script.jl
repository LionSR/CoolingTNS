using Test
using HDF5

include(joinpath(@__DIR__, "..", "scripts", "validation",
                 "summarize_largeN_bond_dimensions.jl"))

@testset "Large-N bond-dimension summary script" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 12)
            write(f, "evolution_method", "continuous")
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
            write(gr, "tdvp_sweep_max_bond", [0 0; 6 13; 8 14])
            write(gr, "tdvp_sweep_saturation_cycle", [0, 1])
            write(gr, "elapsed_seconds", [10.0, 15.5])
            write(gr, "requested_steps", [3, 3])
            write(gr, "completed_steps", [2, 2])
            write(gr, "stop_reasons", ["", "bond_cap"])
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
        @test row.evolution == "continuous"
        @test row.R == 2
        @test row.M == 2
        @test row.completed_requested == "2/3"
        @test row.elapsed_total_seconds == 25.5
        @test row.stop_reason == "bond_capx1/2"
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
        @test row.tdvp_sweep_effective_bond == ">=14"
        @test row.bond_status ==
              "not_converged_system_and_evolved_and_tdvp_sweep_cap"
        @test row.final_system_max == 12
        @test row.final_system_mean == 6.5
        @test row.peak_evolved_max == 14
        @test row.peak_evolved_mean == 10.0
        @test row.peak_tdvp_sweep_max == 14
        @test row.system_saturation_cycle == 2
        @test row.evolved_saturation_cycle == 1
        @test row.tdvp_sweep_saturation_cycle == 1
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
            "| file | N | method | evolution | R | M | completed/requested | elapsed_total | stop_reason | delta_protocol | delta_range | delta_factor | Dcap |",
            output,
        )
        @test occursin("| final E/N | relE | best E/N | best relE | tail E/N |", output)
        @test occursin(
            "| $(basename(path)) | 4 | mcwf | continuous | 2 | 2 | " *
            "2/3 | 25.5 | bond_capx1/2 | fixed_range | " *
            "[0.50000000,3.00000000] | n/a | 12 | >=12 | >=14 | >=14 | " *
            "not_converged_system_and_evolved_and_tdvp_sweep_cap | " *
            "1.00000000 | 2.00000 | " *
            "-0.25000000 | 0.00000 | 0.41666667 | 1.00000 | 3 |",
            output,
        )
        @test occursin("| 2 | 2 | 2/3 | 25.5 | bond_capx1/2 | fixed_range |", output)

        compact_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_compact_markdown(rows)
                end
            end
            read(output_path, String)
        end
        @test occursin(
            "| file | N | method | evolution | R | M | completed/requested | final E/N | best E/N | mode max abs dE/N | Dcap |",
            compact_output,
        )
        @test occursin("| elapsed_total | stop_reason |", compact_output)
        @test occursin(
            "| $(basename(path)) | 4 | mcwf | continuous | 2 | 2 | " *
            "2/3 | 1.00000000 | -0.25000000 | n/a | 12 | >=12 | >=14 | >=14 | " *
            "not_converged_system_and_evolved_and_tdvp_sweep_cap | 25.5 | bond_capx1/2 |",
            compact_output,
        )
        @test parse_args(["--compact", path]).compact
        @test parse_args(["--compact", path]).paths == [path]
        @test_throws ArgumentError parse_args(["--unknown", path])
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
        @test row.evolution == "unknown"
        @test row.completed_requested == "1/1"
        @test isnan(row.elapsed_total_seconds)
        @test row.stop_reason == "none"
        @test row.delta_protocol == "unknown"
        @test row.delta_range == "unknown"
        @test row.delta_factor == "unknown"
        @test row.tdvp_sweep_effective_bond == "n/a"
        @test ismissing(row.peak_tdvp_sweep_max)
        @test ismissing(row.tdvp_sweep_saturation_cycle)
        @test row.bond_status == "not_converged_evolved_cap"

        output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_markdown([row])
                end
            end
            read(output_path, String)
        end
        @test occursin(
            "| $(basename(path)) | 2 | mcwf | unknown | 1 | 1 | " *
            "1/1 | NaN | none | unknown | " *
            "unknown | unknown | 4 | 2 | >=4 | n/a | not_converged_evolved_cap |",
            output,
        )
        @test occursin("| 1 | 1 | 1/1 | NaN | none | unknown |", output)
        @test occursin("| 4.00 | n/a | none | 1 | n/a |", output)
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary reports mode energy reconstruction" begin
    path = tempname() * ".h5"
    try
        N = 4
        J, h = 1.0, 0.5
        ham_params = CoolingTNS.IsingParameters(N, J, h, :periodic)
        k_indices = CoolingTNS.allowed_k_indices(N, -1)
        mode_hk = Matrix{Float64}(undef, 3, length(k_indices))
        mode_hk[1, :] .= -1.0
        mode_hk[2, :] .= NaN
        mode_hk[3, :] .= -0.2
        mode_energy = CoolingTNS.ising_energy_from_mode_hk(
            k_indices, mode_hk[[1, 3], :], ham_params
        )
        final_energy_offset = 0.04

        h5open(path, "w") do f
            write(f, "Dmax", 8)
            write(f, "model", "ising")
            write(f, "bc", "periodic")
            write(f, "J", J)
            write(f, "h", h)
            write(f, "evolution_method", "continuous")
            gn = create_group(f, "N4")
            write(gn, "N", N)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [mode_energy[1], 100.0, mode_energy[2] + final_energy_offset])
            write(gr, "relative_energy_mean", [0.0, 1.0, 2.0])
            write(gr, "system_max_bond", [1, 2, 3])
            write(gr, "system_mean_bond", [1.0, 2.0, 3.0])
            write(gr, "evolved_max_bond", [0, 3, 4])
            write(gr, "evolved_mean_bond", [NaN, 3.0, 4.0])
            write(gr, CoolingTNS.RESULT_MODE_HK, mode_hk)
            write(gr, CoolingTNS.RESULT_MODE_K_INDICES, Float64.(k_indices))
            write(gr, CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES, [0, 2])
            write(gr, CoolingTNS.RESULT_MODE_GF, -1)
            write(gr, CoolingTNS.RESULT_MODE_GF_SOURCE, "state")
        end

        row = only(summarize_file(path))
        @test row.mode_gF == -1
        @test row.mode_gF_source == "state"
        @test row.mode_measured_rows == "2/3"
        @test row.mode_final_e_over_n ≈ mode_energy[2] / N
        @test row.mode_final_abs_err_over_n ≈ final_energy_offset / N
        @test row.mode_max_abs_err_over_n ≈ final_energy_offset / N

        output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_markdown([row])
                end
            end
            read(output_path, String)
        end
        @test occursin(
            "| mode gF | mode source | mode rows | mode final E/N | mode final abs dE/N | mode max abs dE/N |",
            output,
        )
        @test occursin("| -1 | state | 2/3 |", output)

        compact_output = mktemp() do output_path, io
            close(io)
            open(output_path, "w") do out
                redirect_stdout(out) do
                    print_compact_markdown([row])
                end
            end
            read(output_path, String)
        end
        @test occursin("| mode max abs dE/N |", compact_output)
        @test occursin("| $(format_float(final_energy_offset / N, 3)) | 8 |", compact_output)
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary rejects nonintegrable mode metadata" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 8)
            write(f, "model", "niising")
            write(f, "bc", "open")
            write(f, "J", 1.0)
            write(f, "h", NaN)
            gn = create_group(f, "N4")
            write(gn, "N", 4)
            gm = create_group(gn, "mcwf")
            gr = create_group(gm, "R1")

            write(gr, "M", 1)
            write(gr, "E_mean", [0.0])
            write(gr, "relative_energy_mean", [0.0])
            write(gr, "system_max_bond", [1])
            write(gr, "system_mean_bond", [1.0])
            write(gr, "evolved_max_bond", [0])
            write(gr, "evolved_mean_bond", [NaN])
            write(gr, CoolingTNS.RESULT_MODE_HK, reshape(fill(-1.0, 4), 1, 4))
            write(gr, CoolingTNS.RESULT_MODE_K_INDICES, Float64.([-1.5, -0.5, 0.5, 1.5]))
            write(gr, CoolingTNS.RESULT_MODE_MEASUREMENT_CYCLES, [0])
            write(gr, CoolingTNS.RESULT_MODE_GF, -1)
            write(gr, CoolingTNS.RESULT_MODE_GF_SOURCE, "state")
        end

        err = try
            summarize_file(path)
            nothing
        catch err
            err
        end
        @test err isa ErrorException
        @test occursin(
            "mode-energy reconstruction is defined here only for the integrable Ising chain",
            sprint(showerror, err),
        )
    finally
        rm(path; force=true)
    end
end

@testset "Large-N summary treats all-zero TDVP sweep placeholders as missing" begin
    path = tempname() * ".h5"
    try
        h5open(path, "w") do f
            write(f, "Dmax", 4)
            write(f, "evolution_method", "continuous")
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
            write(gr, "tdvp_sweep_max_bond", [0, 0])
            write(gr, "tdvp_sweep_saturation_cycle", [0])
        end

        row = only(summarize_file(path))
        @test row.tdvp_sweep_effective_bond == "n/a"
        @test ismissing(row.peak_tdvp_sweep_max)
        @test ismissing(row.tdvp_sweep_saturation_cycle)
        @test row.bond_status == "not_converged_evolved_cap"
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
