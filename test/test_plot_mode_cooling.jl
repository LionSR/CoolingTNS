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

    @test_throws ErrorException _mode_occupation_from_plot_data(Dict{String, Any}())
end

@testset "Mode cooling resonance convention" begin
    εk_values = [0.8, 1.2, 1.2, 1.7]

    @test bath_detuning_energy([-1.2]) == 1.2
    @test bath_detuning_energy([1.0, 1.2]) === nothing
    @test nearest_bath_resonance_indices(εk_values, -1.2) == [2, 3]
    @test nearest_bath_resonance_indices(εk_values, nothing) == Int[]
    @test nearest_bath_resonance_indices(εk_values, 0.0) == Int[]

    repo_root = normpath(joinpath(@__DIR__, ".."))
    plot_text = read(joinpath(repo_root, "scripts", "plotting", "plot_mode_cooling.jl"), String)
    diagnostic_text = read(joinpath(repo_root, "scripts", "diagnostics", "mode_cooling_diagnostic.jl"), String)

    @test occursin("nearest_bath_resonance_indices", plot_text)
    @test occursin("res_indices = Set(CoolingTNS.nearest_bath_resonance_indices", diagnostic_text)
    @test !occursin("argmin(abs.(εk_values", plot_text)
    @test !occursin("res_idx = argmin", diagnostic_text)
end
