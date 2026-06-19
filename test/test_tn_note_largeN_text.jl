using Test

normalize_ws(s::AbstractString) = replace(s, r"\s+" => " ")

@testset "TN note large-N cooling qualifications" begin
    note_path = joinpath(@__DIR__, "..", "Notes", "NotesTN", "CoolingAlgTN.tex")
    map_path = joinpath(@__DIR__, "..", "Notes", "NotesED", "MapToSpin.tex")
    evidence_path = joinpath(@__DIR__, "..", "docs", "largeN_effective_bond_dimensions.md")

    note_text = read(note_path, String)
    note_flat = normalize_ws(note_text)
    note_lower = lowercase(note_flat)
    map_flat = normalize_ws(read(map_path, String))
    evidence_flat = normalize_ws(read(evidence_path, String))

    # These anchors intentionally couple the note to the source documents.
    # Update them together when the notation or bond-dimension evidence changes.
    @test occursin("not as a final demonstration of ground-state preparation", note_flat)
    @test occursin("Two cooling cycles are not physically meaningful", note_flat)
    @test occursin("they do not establish scalable cooling to the ground state", note_flat)
    @test occursin("historical truncation diagnostics", note_flat)
    @test occursin("D_{\\mathrm{sb}}^{\\mathrm{eff}}=394, 862, 518, 737", note_flat)
    @test occursin("R = 1: Dsys_eff = 309, Dsb_eff = 394", evidence_flat)
    @test occursin("R = 2: Dsys_eff = 637, Dsb_eff = 862", evidence_flat)
    @test occursin("R = 5: Dsys_eff = 399, Dsb_eff = 518", evidence_flat)
    @test occursin("R = 10: Dsys_eff = 489, Dsb_eff = 737", evidence_flat)

    @test occursin("n_k^{\\mathrm{Bog}} = \\langle \\hat n_k\\rangle = \\frac{1+\\langle h_k\\rangle}{2}", map_flat)
    @test occursin("This occupation should not be confused with the raw Fourier occupation", map_flat)
    @test occursin("\\operatorname{coeff}_k=w_k", map_flat)

    forbidden = [
        "algorithm effectively prepares the ground state",
        "confirming the ability of the cooling algorithm to prepare low-energy states",
        "demonstrating the effectiveness of the cooling algorithm in preparing low-energy states",
        "demonstrating its robustness against depolarizing noise",
        "the algorithm still achieves significant cooling and ground state preparation fidelity",
        "the error remains small even for the largest system size",
        "ground state fidelity decreases exponentially",
        "the steady-state energy density",
        "the energy of the steady state is estimated",
    ]
    @test all(!occursin(phrase, note_lower) for phrase in forbidden)
end
