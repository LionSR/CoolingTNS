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
    @test occursin("Dtdvp_sweep_eff", evidence_flat)
    @test occursin("tdvp_sweep_max_bond", evidence_flat)
    @test occursin("tdvp_sweep_saturation_cycle", evidence_flat)
    @test occursin("includes the evolution time `te` and suffixes for non-default detuning schedules and randomized evolution times", evidence_flat)
    @test occursin("not_converged_tdvp_sweep_cap", evidence_flat)
    @test occursin("not_converged_system_and_evolved_and_tdvp_sweep_cap", evidence_flat)
    @test occursin("peak tdvp sweep max", evidence_flat)
    @test occursin("tdvp sweep sat", evidence_flat)
    @test occursin("Forty-Cycle MCWF+TDVP Stop-on-Cap Diagnostics", evidence_flat)
    @test occursin("| 2 | 96 | 3/40 | 0.98249764 | 95 | >=96 | not_converged_evolved_cap | 3:6 | 148.7 s | 1075.2 s |", evidence_flat)
    @test occursin("| 2 | 128 | 3/40 | 0.98236229 | 120 | >=128 | not_converged_evolved_cap | 3:8 | 281.9 s | 1171.6 s |", evidence_flat)
    @test occursin("does not move the first cap event beyond cycle 3", evidence_flat)
    @test occursin("summarized as `n/a`, not as a measured zero-dimensional state", evidence_flat)
    @test occursin("reports `Dtdvp_sweep_eff = n/a` for these legacy files", evidence_flat)
    @test occursin("MCWF+TDVP Stop-on-Cap Scan at te=1.0", evidence_flat)
    @test occursin("| 5 | 96 | 6/40 | 0.95391585 | 0.95391585 | 93 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | 6 | 6 | 678.1 s |", evidence_flat)
    @test occursin("lowering `te` from `2.0` to `1.0` materially delays the first cap event", evidence_flat)
    @test occursin("test `te = 0.5` at `Dmax = 96`", evidence_flat)
    @test occursin("Focused MCWF+TDVP R=2 and R=5 Probes at te=0.5", evidence_flat)
    @test occursin("| 2 | 0.5 | 96 | 11/40 | 1.10014440 | 1.10014440 | 86 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | 11 | 11 | 619.7 s |", evidence_flat)
    @test occursin("| 5 | 0.5 | 96 | 12/40 | 1.03468046 | 1.02937119 | 90 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | 12 | 12 | 744.7 s |", evidence_flat)
    @test occursin("for `R = 2`, the cap moves from cycle 5 at `te = 1.0` to cycle 11 at `te = 0.5`", evidence_flat)
    @test occursin("for `R = 5`, it moves from cycle 6 to cycle 12", evidence_flat)
    @test occursin("argues against the simple rule \"make `te` smaller\"", evidence_flat)
    @test occursin("Random-Schedule MCWF+TDVP Probe at te=1.0", evidence_flat)
    @test occursin("| 2 | random | 1.0 | 96 | 6/40 | 0.99951656 | 0.99951656 | >=96 | >=96 | >=96 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 6 | 6 | 6 | 1051.2 s |", evidence_flat)
    @test occursin("| 5 | random | 1.0 | 96 | 5/40 | 1.12466000 | 1.12466000 | 78 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | none | 5 | 5 | 341.3 s |", evidence_flat)
    @test occursin("`0.99951656` instead of `1.04543594`", evidence_flat)
    @test occursin("this one-seed diagnostic does not give evidence of a scalable low-entanglement route to the ground state", evidence_flat)
    @test occursin("Randomized-Time MCWF+TDVP Probe at Mean te=1.0", evidence_flat)
    @test occursin("| 2 | 1.0 | 96 | 7/40 | 1.09926433 | 1.09097226 | >=96 | >=96 | >=96 | not_converged_system_and_evolved_and_tdvp_sweep_cap | 7 | 7 | 7 | 1041.6 s |", evidence_flat)
    @test occursin("| 10 | 1.0 | 96 | 8/40 | 1.07882431 | 1.07882431 | 91 | >=96 | >=96 | not_converged_evolved_and_tdvp_sweep_cap | none | 8 | 8 | 861.9 s |", evidence_flat)
    @test occursin("| 10 | 8 | 2.46945966 | 1.06039527 | 1.07882431 | 91 | 96 | 861.9 s |", evidence_flat)
    @test occursin("The best randomized-time result is `R = 10` with `E/N = 1.07882431`", evidence_flat)
    @test occursin("does not support it as a solution to the large-N bond-growth bottleneck", evidence_flat)
    @test occursin("--init-state theta --theta 0.0 --measure-modes", evidence_flat)
    @test occursin("mode_gF_source = \"reference\"", evidence_flat)
    @test occursin("mixed-parity state for the periodic Ising chain in the code basis", evidence_flat)

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
