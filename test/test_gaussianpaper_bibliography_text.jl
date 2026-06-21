using Test

function braced_values(pattern::Regex, text::AbstractString)
    values = String[]
    for match in eachmatch(pattern, text)
        append!(values, strip.(split(match.captures[1], ",")))
    end
    return sort(unique(filter(!isempty, values)))
end

@testset "GaussianPaper grouped mode weights keep special-mode half factors" begin
    paper_dir = joinpath(@__DIR__, "..", "Notes", "GaussianPaper")
    main_text = read(joinpath(paper_dir, "journal_main.tex"), String)
    supp_text = read(joinpath(paper_dir, "journal_supp.tex"), String)
    joined = main_text * "\n" * supp_text

    @test occursin(raw"H_{S} = \sum_{k=0}^{N/2} \eta_k h_k", main_text)
    @test occursin(raw"&= \sum_{k=0}^{N/2} \eta_k h_k,", supp_text)
    @test occursin(raw"\eta_k", supp_text)
    @test occursin(raw"\epsilon_k/2, & k=0,N/2.", main_text)
    @test occursin(raw"\epsilon_k/2, & k=0,N/2.", supp_text)
    @test occursin("The factor \$1/2\$ for the special modes is inherited from the full-grid formula", main_text)
    @test occursin("This half-weight for the special modes is the direct consequence of the", supp_text)
    @test occursin(raw"E =\sum_{k=0}^{N/2} E_k = \sum_{k=0}^{N/2} \eta_k \tr( h_k \sigma_k).", main_text)
    @test !occursin(raw"H_{S} = \sum_{k=0}^{N/2} \epsilon_k h_k", joined)
    @test !occursin(raw"H_S
     &= \sum_{k=0}^{N/2} \epsilon_k h_k,", joined)
end

@testset "GaussianPaper bibliography and figure artifact notes" begin
    paper_dir = joinpath(@__DIR__, "..", "Notes", "GaussianPaper")
    main_text = read(joinpath(paper_dir, "journal_main.tex"), String)
    supp_text = read(joinpath(paper_dir, "journal_supp.tex"), String)
    bib_text = read(joinpath(paper_dir, "library.bib"), String)
    readme_text = read(joinpath(paper_dir, "README.md"), String)

    tex_text = main_text * "\n" * supp_text
    cite_keys = braced_values(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]*)\}", tex_text)
    bib_keys = sort([match.captures[1] for match in eachmatch(r"@\w+\{([^,\s]+)", bib_text)])
    missing_bib_keys = setdiff(cite_keys, bib_keys)
    extra_bib_keys = setdiff(bib_keys, cite_keys)

    @test !isempty(cite_keys)
    @test isempty(missing_bib_keys)
    @test isempty(extra_bib_keys)
    @test !occursin("Bibliography database placeholder", bib_text)
    @test !occursin("Add full publication metadata", bib_text)

    figures = sort(unique(vcat(
        braced_values(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex_text),
        braced_values(r"\\begin\{overpic\}(?:\[[^\]]*\])?\{([^}]*)\}", tex_text),
    )))
    readme_figures = sort([match.captures[1] for match in eachmatch(r"(?m)^- `([^`]+)`\s*$", readme_text)])
    missing_figure_notes = setdiff(figures, readme_figures)
    extra_figure_notes = setdiff(readme_figures, figures)

    @test !isempty(figures)
    @test isempty(missing_figure_notes)
    @test isempty(extra_figure_notes)
    @test occursin("publication figure PDFs are not tracked", readme_text)
    @test occursin("renders labelled placeholder boxes", readme_text)
end

@testset "GaussianPaper finite-size claims remain qualified" begin
    paper_dir = joinpath(@__DIR__, "..", "Notes", "GaussianPaper")
    supp_text = read(joinpath(paper_dir, "journal_supp.tex"), String)
    supp_flat = replace(supp_text, r"\s+" => " ")

    @test occursin(raw"\section{Finite-size transferability of the optimal parameters}", supp_text)
    @test occursin(raw"\cref{app:finite_size_transferability}", read(joinpath(paper_dir, "journal_main.tex"), String))
    @test occursin(raw"\cref{fig:cooling_size_transfer_average}", supp_text)
    @test occursin("phase-averaged free-fermion scan", supp_flat)
    @test occursin("transfer within this free-fermion scan", supp_flat)
    @test !occursin(r"\blwo\b", supp_text)
    @test !occursin("demonstrating the scalability and robustness of the optimized protocol", supp_flat)
    @test !occursin("This highlights the protocol's robustness across different system sizes and phases", supp_flat)
end

@testset "GaussianPaper multi-frequency noise claims remain local" begin
    paper_dir = joinpath(@__DIR__, "..", "Notes", "GaussianPaper")
    main_text = read(joinpath(paper_dir, "journal_main.tex"), String)
    main_flat = replace(main_text, r"\s+" => " ")

    @test occursin("In this finite parameter sweep", main_flat)
    @test occursin("Across most panels in this finite sweep", main_flat)
    @test occursin("Within the finite parameter sweeps studied here", main_flat)
    @test occursin("the quantitative tolerance depends on the mode, coupling strength, cycle time, and noise model", main_flat)
    @test occursin("the benefit depends on the mode, coupling strength, and noise level", main_flat)
    @test occursin("not as a parameter-uniform robustness statement", main_flat)
    @test occursin("this comparison is not a parameter-uniform robustness theorem", main_flat)
    @test occursin("find parameter regimes in this solvable model where cooling gives lower relative energies", main_flat)
    @test occursin("leaving the quantitative noise tolerance to the conditions below", main_flat)
    @test !occursin("in general, using multiple frequencies", main_flat)
    @test !occursin(raw"consistently outperforms single-frequency cooling ($R=1$) across all parameter regimes", main_flat)
    @test !occursin("the results demonstrate a degree of robustness", main_flat)
    @test !occursin("This highlights the general utility of the multi-frequency approach", main_flat)
    @test !occursin("optimized cooling protocols that can significantly enhance cooling performance in the presence of noise", main_flat)
    @test !occursin("cooling generally achieves lower energies and is more resilient to noise", main_flat)
    @test !occursin("the former generally achieves lower energies", main_flat)
end
