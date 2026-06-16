using Test
using CoolingTNS
using HDF5

include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_mode_cooling.jl"))

@testset "Mode cooling plot data convention" begin
    mode_hk = [-1.0 0.0; 1.0 -0.5]
    mode_nk = mode_occupation_from_hk(mode_hk)

    @test _mode_occupation_from_plot_data(Dict{String, Any}(RESULT_MODE_HK => mode_hk)) ≈ mode_nk

    stored_nk = [0.1 0.2; 0.3 0.4]
    data = Dict{String, Any}(RESULT_MODE_HK => mode_hk, RESULT_MODE_NK => stored_nk)
    @test _mode_occupation_from_plot_data(data) == stored_nk

    @test _occupation_ylim(mode_nk) == (-0.05, 1.05)
    @test _occupation_ylim([NaN]) == (-0.05, 1.05)

    @test select_evolution_steps(0) == Int[]
    @test select_evolution_steps(1) == [1]
    @test select_evolution_steps(2) == [1, 2]
    @test select_evolution_steps(21) == [1, 6, 11, 16, 21]
    @test select_evolution_steps(5; steps_to_plot=[0, 1, 3, 7]) == [1, 3]

    dummy_plt = (cm=(viridis=identity,),)
    @test get_evolution_colors(dummy_plt, 0) == Any[]
    @test get_evolution_colors(dummy_plt, 1) == [0.0]
    @test collect(get_evolution_colors(dummy_plt, 3)) == [0.0, 0.5, 1.0]

    mktempdir() do dir
        filename = joinpath(dir, "with_metadata_group.h5")
        h5open(filename, "w") do file
            write(file, RESULT_MOMENTUM_DISTRIBUTION, [0.1 0.2; 0.3 0.4])
            write(file, RESULT_K_VALUES, [0.0, pi])

            parsed_args = create_group(file, CoolingTNS.HDF5_PARSED_ARGS_GROUP)
            write(parsed_args, "N", 2)
            write(parsed_args, "backend", "ED")
        end

        h5_data = read_h5_data(filename)
        @test h5_data !== nothing
        @test h5_data[RESULT_MOMENTUM_DISTRIBUTION] == [0.1 0.2; 0.3 0.4]
        @test h5_data[RESULT_K_VALUES] == [0.0, pi]
        @test !haskey(h5_data, CoolingTNS.HDF5_PARSED_ARGS_GROUP)
    end

    k_values = [-pi, 0.0, pi]
    step_by_mode = [0.1 0.2 0.3; 0.4 0.5 0.6]
    canonical_kspace = kspace_evolution_plot_data(
        Dict{String, Any}(
            RESULT_MOMENTUM_DISTRIBUTION => step_by_mode,
            RESULT_K_VALUES => k_values,
            "N" => 3,
            "J" => 1.5,
            "h" => 0.7,
            "bc" => "periodic",
        ),
    )
    @test canonical_kspace.momentum_dist == step_by_mode
    @test canonical_kspace.k_values == k_values
    @test canonical_kspace.total_steps == 2
    @test canonical_kspace.N == 3
    @test canonical_kspace.J == 1.5
    @test canonical_kspace.h == 0.7
    @test canonical_kspace.bc == :periodic

    legacy_mode_by_step = permutedims(step_by_mode)
    legacy_kspace = kspace_evolution_plot_data(
        Dict{String, Any}(
            RESULT_MOMENTUM_DISTRIBUTION => legacy_mode_by_step,
            RESULT_K_VALUES => k_values,
        ),
    )
    @test legacy_kspace.momentum_dist == step_by_mode
    @test legacy_kspace.total_steps == 2

    square_step_by_mode = [0.1 0.2; 0.3 0.4]
    square_kspace = kspace_evolution_plot_data(
        Dict{String, Any}(
            RESULT_MOMENTUM_DISTRIBUTION => square_step_by_mode,
            RESULT_K_VALUES => [0.0, pi],
        ),
    )
    @test square_kspace.momentum_dist == square_step_by_mode

    @test_throws ErrorException kspace_evolution_plot_data(Dict{String, Any}())
    @test_throws DimensionMismatch kspace_evolution_plot_data(
        Dict{String, Any}(
            RESULT_MOMENTUM_DISTRIBUTION => [0.1 0.2; 0.3 0.4],
            RESULT_K_VALUES => [0.0, pi, 2pi],
        ),
    )

    detuning_calls = Any[]
    dummy_ax = (
        axhline=(; y, color, linestyle, linewidth, label, alpha) -> begin
            push!(
                detuning_calls,
                (y=y, color=color, linestyle=linestyle, linewidth=linewidth, label=label, alpha=alpha),
            )
            return :line
        end,
    )
    @test mark_bath_detuning_energy!(dummy_ax, -2.5) == :line
    @test only(detuning_calls).y == 2.5
    @test only(detuning_calls).label == "|delta|"
    @test mark_bath_detuning_energy!(dummy_ax, nothing) === nothing
    @test mark_bath_detuning_energy!(dummy_ax, 0.0) === nothing
    @test length(detuning_calls) == 1

    mode_energies = [0.5, 1.5, 1.5, 3.0]
    @test nearest_bath_resonance_indices(mode_energies, 1.5) == [2, 3]
    @test nearest_bath_resonance_indices(mode_energies, -1.45) == [2, 3]
    @test nearest_bath_resonance_indices(mode_energies, nothing) == Int[]
    @test nearest_bath_resonance_indices(Float64[], 1.0) == Int[]

    plot_k_values = [0.0, pi / 3, 2pi / 3, pi]
    stored_energy_data = Dict{String, Any}(
        RESULT_MODE_ENERGIES => mode_energies,
        "J" => 10.0,
        "h" => 20.0,
    )
    @test momentum_plot_mode_energies(stored_energy_data, plot_k_values) == mode_energies

    J, h = 1.0, 0.5
    fallback_data = Dict{String, Any}("J" => [J], "h" => h)
    @test momentum_plot_mode_energies(fallback_data, plot_k_values) ≈
          compute_energy_dispersion(plot_k_values, J, h)

    bad_energy_data = Dict{String, Any}(RESULT_MODE_ENERGIES => mode_energies[1:3])
    @test_logs (:warn, r"Skipping stored mode energies") begin
        @test momentum_plot_mode_energies(bad_energy_data, plot_k_values) === nothing
    end

    resonance_calls = Any[]
    dummy_momentum_ax = (
        axvline=(; x, color, linestyle, linewidth, alpha, label) -> begin
            push!(
                resonance_calls,
                (x=x, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=label),
            )
            return :vline
        end,
    )
    handles = mark_bath_resonance_momentum!(
        dummy_momentum_ax,
        plot_k_values,
        mode_energies,
        1.5;
        momentum_scale=pi,
    )
    @test handles == [:vline, :vline]
    @test [call.x for call in resonance_calls] == [1 / 3, 2 / 3]
    @test resonance_calls[1].label == "nearest epsilon_k ~= |delta|"
    @test resonance_calls[2].label == "_nolegend_"

    resonance_from_data_calls = Any[]
    dummy_data_ax = (
        axvline=(; x, color, linestyle, linewidth, alpha, label) -> begin
            push!(resonance_from_data_calls, x)
            return :data_vline
        end,
    )
    data_handles = mark_bath_resonance_from_data!(
        dummy_data_ax,
        Dict{String, Any}(
            "delta" => 1.5,
            RESULT_MODE_ENERGIES => mode_energies,
        ),
        plot_k_values;
        momentum_scale=pi,
    )
    @test data_handles == [:data_vline, :data_vline]
    @test resonance_from_data_calls == [1 / 3, 2 / 3]
    @test mark_bath_resonance_from_data!(dummy_data_ax, Dict{String, Any}(), plot_k_values) === nothing

    @test_throws ErrorException _mode_occupation_from_plot_data(Dict{String, Any}())
end
