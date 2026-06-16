using Test
using CoolingTNS

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

    @test_throws ErrorException _mode_occupation_from_plot_data(Dict{String, Any}())
end
