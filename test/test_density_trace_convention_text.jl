using Test

_density_trace_normalize_ws(s::AbstractString) = replace(s, r"\s+" => " ")

@testset "Density-matrix partial-trace canonicalization text" begin
    ed_backend_path = joinpath(@__DIR__, "..", "src", "ed_backend.jl")
    map_note_path = joinpath(@__DIR__, "..", "Notes", "NotesED", "MapToSpin.tex")
    tn_note_path = joinpath(@__DIR__, "..", "Notes", "NotesTN", "CoolingAlgTN.tex")

    ed_backend_flat = _density_trace_normalize_ws(read(ed_backend_path, String))
    map_flat = _density_trace_normalize_ws(read(map_note_path, String))
    tn_flat = _density_trace_normalize_ws(read(tn_note_path, String))

    @test occursin(
        "Trace out all qubits except those specified in keep_qubits. The reduced dense block is returned as its Hermitian, trace-one representative",
        ed_backend_flat,
    )
    @test occursin("matching the TN MPO post-trace convention", ed_backend_flat)

    @test occursin(raw"\ket{s_1,b_1,s_2,b_2,\ldots,s_N,b_N}", map_flat)
    @test occursin(raw"\label{eq:ed_partial_trace_index}", map_flat)
    @test occursin(raw"\texttt{partial\_trace\_ed} and \texttt{construct\_index}", map_flat)
    @test occursin(raw"\rho_{\mathrm{red}}^{\mathrm{num}}", map_flat)
    @test occursin(
        "This Hermitian, trace-one representative removes roundoff-level anti-Hermitian components",
        map_flat,
    )
    @test occursin("it does not define a different cooling channel", map_flat)

    @test occursin(
        "both density-matrix implementations store the reduced system state rather than the full system-bath state",
        tn_flat,
    )
    @test occursin(raw"\label{eq:density_post_trace_canonicalization}", tn_flat)
    @test occursin(
        "For the MPO path, the dagger in \\cref{eq:density_post_trace_canonicalization} is implemented by swapping the primed and unprimed site layers",
        tn_flat,
    )
    @test occursin(
        "This post-trace canonicalization is a numerical representative of the same mathematical state",
        tn_flat,
    )
    @test occursin(
        "The bath magnetization is computed from the bath reduced state before the bath is discarded",
        tn_flat,
    )
end
