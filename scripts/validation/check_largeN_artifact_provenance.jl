#!/usr/bin/env julia
"""
Check that supplied large-N HDF5 artifacts are explicitly named in provenance
documents.

Example:

    julia --project=. scripts/validation/check_largeN_artifact_provenance.jl \
        .worktree/campaign/*.h5

Use `--doc FILE` to replace the default provenance-document set.  The check
matches artifact basenames as filename tokens rather than absolute paths, so
local worktree locations may differ while the recorded HDF5 artifact name
remains auditable.
"""

module LargeNArtifactProvenanceChecker

export DEFAULT_PROVENANCE_DOCUMENTS,
       ProvenanceArgs,
       artifact_basenames,
       artifact_basename_is_recorded,
       missing_artifact_basenames,
       parse_artifact_provenance_args,
       provenance_check_exit_code,
       main

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_PROVENANCE_DOCUMENTS = String[
    joinpath(REPO_ROOT, "docs", "largeN_effective_bond_dimensions.md"),
    joinpath(REPO_ROOT, "docs", "multi_frequency_cooling_plan.md"),
    joinpath(REPO_ROOT, "Notes", "NotesTN", "CoolingAlgTN.tex"),
    joinpath(REPO_ROOT, "Notes", "NotesED", "MapToSpin.tex"),
]

struct ProvenanceArgs
    doc_paths::Vector{String}
    artifact_paths::Vector{String}
end

function usage(io::IO=stdout)
    println(
        io,
        "usage: julia --project=. scripts/validation/check_largeN_artifact_provenance.jl " *
        "[--doc FILE]... FILE.h5 [FILE2.h5 ...]",
    )
end

function parse_artifact_provenance_args(args::AbstractVector{<:AbstractString})
    doc_paths = String[]
    artifact_paths = String[]
    i = firstindex(args)
    while i <= lastindex(args)
        arg = args[i]
        if arg == "--help" || arg == "-h"
            return nothing
        elseif arg == "--doc"
            i == lastindex(args) && throw(ArgumentError("--doc requires a file path"))
            i += 1
            push!(doc_paths, String(args[i]))
        elseif startswith(arg, "--doc=")
            path = String(arg[length("--doc=") + 1:end])
            isempty(path) && throw(ArgumentError("--doc= requires a nonempty file path"))
            push!(doc_paths, path)
        elseif startswith(arg, "-")
            throw(ArgumentError("unknown option: $arg"))
        else
            push!(artifact_paths, String(arg))
        end
        i += 1
    end

    isempty(artifact_paths) &&
        throw(ArgumentError("at least one HDF5 artifact path is required"))
    return ProvenanceArgs(
        isempty(doc_paths) ? copy(DEFAULT_PROVENANCE_DOCUMENTS) : doc_paths,
        artifact_paths,
    )
end

function read_provenance_text(doc_paths::AbstractVector{<:AbstractString})
    io = IOBuffer()
    for path in doc_paths
        isfile(path) || throw(ArgumentError("provenance document does not exist: $path"))
        print(io, read(path, String))
        print(io, '\n')
    end
    return String(take!(io))
end

function artifact_basenames(artifact_paths::AbstractVector{<:AbstractString})
    names = String[]
    for path in artifact_paths
        isfile(path) || throw(ArgumentError("HDF5 artifact does not exist: $path"))
        push!(names, basename(path))
    end
    return names
end

is_filename_token_character(c::Char) = isletter(c) || isdigit(c) || c in ('_', '-', '.')
is_filename_boundary(c::Char) = !is_filename_token_character(c)
function is_after_filename_boundary(text::AbstractString, index::Integer)
    index > lastindex(text) && return true
    c = text[index]
    is_filename_boundary(c) && return true
    c == '.' || return false
    next_index = nextind(text, index)
    return next_index > lastindex(text) || isspace(text[next_index])
end

function artifact_basename_is_recorded(name::AbstractString, text::AbstractString)
    start = firstindex(text)
    while start <= lastindex(text)
        range = findnext(name, text, start)
        range === nothing && return false

        before_ok = first(range) == firstindex(text) ||
            is_filename_boundary(text[prevind(text, first(range))])
        after_index = nextind(text, last(range))
        after_ok = is_after_filename_boundary(text, after_index)
        before_ok && after_ok && return true
        start = nextind(text, first(range))
    end
    return false
end

function missing_artifact_basenames(
    artifact_paths::AbstractVector{<:AbstractString},
    doc_paths::AbstractVector{<:AbstractString},
)
    text = read_provenance_text(doc_paths)
    return String[
        name for name in artifact_basenames(artifact_paths)
        if !artifact_basename_is_recorded(name, text)
    ]
end

function provenance_check_exit_code(
    artifact_paths::AbstractVector{<:AbstractString},
    doc_paths::AbstractVector{<:AbstractString};
    io::IO=stdout,
)
    missing_names = missing_artifact_basenames(artifact_paths, doc_paths)
    if isempty(missing_names)
        println(
            io,
            "All $(length(artifact_paths)) large-N artifact basename(s) are named " *
            "in $(length(doc_paths)) provenance document(s).",
        )
        return 0
    end

    println(
        io,
        "Missing $(length(missing_names)) of $(length(artifact_paths)) large-N artifact " *
        "basename(s) from $(length(doc_paths)) provenance document(s):",
    )
    for name in missing_names
        println(io, "- $name")
    end
    return 1
end

function main(args::AbstractVector{<:AbstractString}=ARGS)
    try
        parsed = parse_artifact_provenance_args(args)
        parsed === nothing && (usage(); return 0)
        return provenance_check_exit_code(parsed.artifact_paths, parsed.doc_paths)
    catch err
        println(stderr, "error: ", err)
        usage(stderr)
        return 2
    end
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(LargeNArtifactProvenanceChecker.main(ARGS))
end
