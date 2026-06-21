using Test

@testset "Bogoliubov angle and occupation notation in source comments" begin
    source_files = [
        joinpath(@__DIR__, "..", "src", "ed_backend_complex_jw.jl"),
        joinpath(@__DIR__, "..", "src", "mode_analysis.jl"),
        joinpath(@__DIR__, "..", "src", "dispersion.jl"),
        joinpath(@__DIR__, "..", "scripts", "plotting", "plot_mode_cooling.jl"),
    ]

    for file in source_files
        text = read(file, String)
        @test !occursin("sin²(φ_k)", text)
    end

    joined = join(read.(source_files, String), "\n")
    @test occursin("sin²(varphi_k)", joined)
    @test occursin("φ_k = 2πk/N", joined)
    @test occursin("n_k^{Bog}", joined)
    @test occursin("Bogoliubov occupation number ``n_k^{Bog}``", joined)
    @test occursin("Bogoliubov occupations, ⟨h_k⟩", joined)
    @test occursin("Σ_k ε_k (n_k^{OBC} - 1/2)", joined)

    forbidden = [
        "(ε_k, n_k, ⟨h_k⟩)",
        "h_k = 2 n_k - 1",
        "h_k = 2n_k - 1",
        "Σ_k ε_k (n_k - 1/2)",
        "mode energies, occupations, ⟨h_k⟩",
    ]
    @test all(phrase -> !occursin(phrase, joined), forbidden)
end
