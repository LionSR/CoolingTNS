using Test

_slide_normalize_ws(s::AbstractString) = replace(s, r"\s+" => " ")

@testset "Slide k-space notation follows notes convention" begin
    slide_path = joinpath(@__DIR__, "..", "slides", "seminar_integrable_ising.tex")
    plot_path = joinpath(@__DIR__, "..", "scripts", "plotting", "plot_mode_energy_consistency_ed.jl")
    map_path = joinpath(@__DIR__, "..", "Notes", "NotesED", "MapToSpin.tex")
    tn_note_path = joinpath(@__DIR__, "..", "Notes", "NotesTN", "CoolingAlgTN.tex")

    slide_flat = _slide_normalize_ws(read(slide_path, String))
    plot_flat = _slide_normalize_ws(read(plot_path, String))
    map_flat = _slide_normalize_ws(read(map_path, String))
    tn_flat = _slide_normalize_ws(read(tn_note_path, String))

    @test occursin("K-space notation used in diagnostics", slide_flat)
    @test occursin("\\tilde n_k &= \\langle \\tilde a_k^\\dagger \\tilde a_k\\rangle", slide_flat)
    @test occursin("n_k^{\\mathrm{Bog}} &= \\langle \\hat n_k\\rangle = \\frac{1+\\langle h_k\\rangle}{2}", slide_flat)
    @test occursin("E_{\\mathrm{modes}} =\\frac{\\Lambda}{2}\\sum_{k\\in\\mathrm{grid}(g_F)} \\operatorname{coeff}_k\\,\\langle h_k\\rangle", slide_flat)
    @test occursin("with signed special-mode coefficients", slide_flat)
    @test occursin(raw"In the displayed finite-window examples, multi-frequency detunings can lower the measured energy relative to single-$\Delta$ protocols", slide_flat)
    @test occursin(raw"use large-$N$ TN runs as convergence diagnostics", slide_flat)
    @test occursin("bond-dimension and trajectory checks before making scalable ground-state cooling claims", slide_flat)
    @test occursin("raw Fourier occupation", slide_flat)
    @test occursin("quasiparticle occupation", slide_flat)

    @test occursin("ising_energy_from_mode_hk", plot_flat)
    @test occursin("The energy reconstruction uses \\(h_k\\) and the signed coefficient", map_flat)
    @test occursin("writing a mode contribution as \$\\epsilon_k n_k\$ without specifying which occupation is meant, gives a different quantity", tn_flat)

    checked_sources = [slide_flat, plot_flat]
    obsolete_ascii = "e_k = " * "epsilon_k n_k"
    obsolete_tex = "e_k = " * "\\epsilon_k n_k"
    obsolete_unicode = "e_k = " * "\u03b5_k n_k"
    obsolete_title = "k-space diagnostics via Jordan-Wigner"
    obsolete_momentum_occupation = "momentum occupations n_k"
    @test all(text -> !occursin(obsolete_ascii, text), checked_sources)
    @test all(text -> !occursin(obsolete_tex, text), checked_sources)
    @test all(text -> !occursin(obsolete_unicode, text), checked_sources)
    @test !occursin(obsolete_title, slide_flat)
    @test !occursin(obsolete_momentum_occupation, slide_flat)
    @test !occursin(raw"Multi-frequency detunings improve cooling performance compared to single-$\Delta$ protocols", slide_flat)
    @test !occursin(raw"extend TN runs to larger $N$ and perform parameter/convergence scans", slide_flat)
end
