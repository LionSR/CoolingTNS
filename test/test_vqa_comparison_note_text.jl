using Test

_vqa_normalize_ws(s::AbstractString) = replace(s, r"\s+" => " ")

@testset "VQA comparison note keeps robustness claims qualified" begin
    compare_path = joinpath(@__DIR__, "..", "Notes", "NotesTN", "compare_with_vqa.tex")
    compare_flat = _vqa_normalize_ws(read(compare_path, String))
    compare_lower = lowercase(compare_flat)

    @test occursin("finite-size variational baseline", compare_flat)
    @test occursin("not as a general statement about the relative scalability or noise robustness", compare_flat)
    @test occursin("does not by itself establish the large-system performance of VQA", compare_flat)
    @test occursin("do not constitute a controlled robustness theorem or a high-fidelity preparation result", compare_flat)
    @test occursin("exact-diagonalization checks at small \$N\$, bond-dimension convergence, trajectory or sampling convergence", compare_flat)
    @test occursin("Which mechanism is preferable is a numerical and physical question", compare_flat)
    @test occursin("the plotted noiseless run lowers the energy density over the optimization window", compare_flat)

    forbidden = [
        "can effectively minimize the energy and prepare states with high overlap",
        "demonstrates a higher level of resilience to noise",
        "ability to prepare low-energy states with high fidelity",
        "relatively strong depolarizing noise",
        "inherent error-correcting properties",
        "maintain a low-energy state despite the presence of noise",
        "highlights the potential advantages of the cooling algorithm in terms of noise resilience",
        "the energy decreases as we increase the number of iterations",
    ]
    @test all(!occursin(phrase, compare_lower) for phrase in forbidden)
end
