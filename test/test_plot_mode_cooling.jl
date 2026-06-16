using Test
using CoolingTNS

include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_mode_cooling.jl"))

@testset "Mode cooling plot data convention" begin
    mode_hk = [-1.0 0.0; 1.0 -0.5]
    mode_nk = mode_occupation_from_hk(mode_hk)

    @test _mode_occupation_from_plot_data(Dict{String, Any}("mode_hk" => mode_hk)) ≈ mode_nk

    stored_nk = [0.1 0.2; 0.3 0.4]
    data = Dict{String, Any}("mode_hk" => mode_hk, "mode_nk" => stored_nk)
    @test _mode_occupation_from_plot_data(data) == stored_nk

    @test _occupation_ylim(mode_nk) == (-0.05, 1.05)
    @test _occupation_ylim([NaN]) == (-0.05, 1.05)

    @test_throws ErrorException _mode_occupation_from_plot_data(Dict{String, Any}())
end
