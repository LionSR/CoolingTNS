using Test

include(joinpath(@__DIR__, "..", "scripts", "validation",
                 "check_largeN_artifact_provenance.jl"))

using .LargeNArtifactProvenanceChecker

@testset "Large-N artifact provenance checker" begin
    mktempdir() do dir
        doc_path = joinpath(dir, "evidence.md")
        artifact_dir = joinpath(dir, "runs")
        mkpath(artifact_dir)
        recorded_artifact = joinpath(
            artifact_dir,
            "largeN_multifrequency_tn_N64_R10_mcwf_continuous_stopcap_steps4_Dmax64_g0.3_te1_tau0.2_seed20260617.h5",
        )
        missing_artifact = joinpath(
            artifact_dir,
            "largeN_multifrequency_tn_N64_R5_mcwf_continuous_stopcap_steps4_Dmax64_g0.3_te1_tau0.2_seed20260617.h5",
        )
        write(recorded_artifact, "")
        write(missing_artifact, "")
        write(doc_path, "The run wrote $(basename(recorded_artifact)).\n")

        parsed = parse_artifact_provenance_args([
            "--doc",
            doc_path,
            recorded_artifact,
            missing_artifact,
        ])
        @test parsed.doc_paths == [doc_path]
        @test parsed.artifact_paths == [recorded_artifact, missing_artifact]
        equals_parsed = parse_artifact_provenance_args([
            "--doc=$doc_path",
            recorded_artifact,
        ])
        @test equals_parsed.doc_paths == [doc_path]
        @test equals_parsed.artifact_paths == [recorded_artifact]
        repeated_doc_parsed = parse_artifact_provenance_args([
            "--doc",
            doc_path,
            "--doc=$doc_path",
            recorded_artifact,
        ])
        @test repeated_doc_parsed.doc_paths == [doc_path, doc_path]
        @test repeated_doc_parsed.artifact_paths == [recorded_artifact]
        @test artifact_basenames(parsed.artifact_paths) ==
              [basename(recorded_artifact), basename(missing_artifact)]
        @test missing_artifact_basenames(parsed.artifact_paths, parsed.doc_paths) ==
              [basename(missing_artifact)]

        io = IOBuffer()
        @test provenance_check_exit_code([recorded_artifact], [doc_path]; io) == 0
        @test occursin("All 1 large-N artifact basename", String(take!(io)))

        io = IOBuffer()
        @test provenance_check_exit_code(
            [recorded_artifact, missing_artifact],
            [doc_path];
            io,
        ) == 1
        report = String(take!(io))
        @test occursin("Missing 1 of 2 large-N artifact", report)
        @test occursin(basename(missing_artifact), report)

        @test parse_artifact_provenance_args(["--help"]) === nothing
        @test_throws ArgumentError parse_artifact_provenance_args(["--doc"])
        @test_throws ArgumentError parse_artifact_provenance_args(["--doc="])
        @test_throws ArgumentError parse_artifact_provenance_args(["--unknown"])
        @test_throws ArgumentError parse_artifact_provenance_args(String[])
        @test_throws ArgumentError artifact_basenames([joinpath(dir, "absent.h5")])
        @test_throws ArgumentError missing_artifact_basenames([recorded_artifact], [joinpath(dir, "missing.md")])
    end
end
