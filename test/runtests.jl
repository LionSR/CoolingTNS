using Test
using CoolingTNS

include("test_helpers.jl")

const RUN_FULL_TESTS = full_tests_enabled()

if !RUN_FULL_TESTS
    @info "Skipping slow/stochastic tests (set COOLINGTNS_FULL_TESTS=1 to enable)."
end

@testset "CoolingTNS" begin
    @testset "Fast" begin
        include("test_interleaved_layout.jl")
        include("test_hamiltonians.jl")
        include("test_initial_states.jl")
        include("test_bath_measurements.jl")
        include("test_noise.jl")
        include("test_result_structs.jl")
        include("test_cooling_interface.jl")
        include("test_optimization_lookup.jl")
        include("test_multi_frequency.jl")
    end

    @testset "Correctness" begin
        include("test_correctness.jl")
        include("test_tdvp_convention.jl")
        include("test_trotter_time.jl")
    end

    @testset "Mode Analysis" begin
        include("test_mode_analysis.jl")
        include("test_measure_hk.jl")
        include("test_tn_mode_observables.jl")
        include("test_plot_mode_cooling.jl")
        include("test_plot_ek_evolution.jl")
        include("test_dispersion_detuning_markers.jl")
    end

    if RUN_FULL_TESTS
        @testset "Slow / Full" begin
            include("test_tn_dm_trotter_debug.jl")
        end
    end
end
