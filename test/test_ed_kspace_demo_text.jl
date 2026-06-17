using Test

@testset "ED k-space demo text uses canonical dispersion" begin
    demo_path = joinpath(@__DIR__, "..", "examples", "ed_kspace_demo.jl")
    guide_path = joinpath(@__DIR__, "..", "CLAUDE.md")
    text = read(demo_path, String)
    guide_text = read(guide_path, String)

    canonical_prefactor = "2√(J² + h²)"
    canonical_sign = "1 - " * "sin(2θ)cos"
    canonical_angle = "θ = atan(h, J)"
    resonance_condition = "ε_k ≈ |δ|"
    guide_canonical = "2sqrt(J^2+h^2) * sqrt(1 - sin(2θ) * cos(2π*k/N))"
    guide_real_label = raw"\ref{eq:bdg_block}"

    legacy_sign = "1 + " * "sin(2θ)cos"
    legacy_resonance = "ε_k ≈ δ " * "(bath frequency)"
    spin_bc_grid_label = "PBC: k ∈"
    guide_legacy = "sqrt(1 + sin(2θ) * cos(2π*k/N))"
    guide_missing_label = "eq:mode_energy"

    @test occursin(canonical_prefactor, text)
    @test occursin(canonical_sign, text)
    @test occursin(canonical_angle, text)
    @test occursin(resonance_condition, text)
    @test occursin(guide_canonical, guide_text)
    @test occursin(guide_real_label, guide_text)

    @test !occursin(legacy_sign, text)
    @test !occursin(legacy_resonance, text)
    @test !occursin(spin_bc_grid_label, text)
    @test !occursin(guide_legacy, guide_text)
    @test !occursin(guide_missing_label, guide_text)
end
