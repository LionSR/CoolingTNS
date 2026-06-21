using Test

function braced_values(pattern::Regex, text::AbstractString)
    values = String[]
    for match in eachmatch(pattern, text)
        append!(values, strip.(split(match.captures[1], ",")))
    end
    return sort(unique(filter(!isempty, values)))
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
