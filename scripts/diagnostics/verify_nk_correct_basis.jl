#!/usr/bin/env julia
# Backward-compatible entry point for the JW occupation diagnostic.

if !isdefined(@__MODULE__, :verify_sign)
    include(joinpath(@__DIR__, "verify_sigma_z_sign.jl"))
end

function verify_nk(; kwargs...)
    return verify_sign(; kwargs...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify_nk()
end
