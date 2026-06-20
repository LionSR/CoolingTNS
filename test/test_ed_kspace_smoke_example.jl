using Test
using CoolingTNS

@testset "ED k-space smoke example" begin
    example_path = joinpath(@__DIR__, "..", "examples", "test_ed_kspace.jl")
    text = read(example_path, String)
    nonintegrable_constructor = "Ni" * "IsingParameters"
    old_plot_line = "CoolingTNS." * "plot_momentum_distribution"
    old_heatmap_line = "CoolingTNS." * "plot_momentum_distribution_heatmap"

    @test occursin("IsingParameters(N, J, h, bc)", text)
    @test !occursin(nonintegrable_constructor, text)
    @test !occursin(old_plot_line, text)
    @test !occursin(old_heatmap_line, text)
    @test occursin("φ/π", text)
    @test occursin(raw"\\tilde n_k", text)
    @test occursin("n_k^Bog", text)
    @test !occursin("n_k = %.6f", text)
    @test occursin("measure_modes=true", text)

    include(example_path)
    @test isdefined(@__MODULE__, :test_ed_kspace)
    @test isdefined(@__MODULE__, :run_ed_kspace_case)
end
