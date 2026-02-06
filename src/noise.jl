"""
    noise.jl

Noise models for tensor network simulations.
"""

using ITensors

# Custom Pauli operators for noise application
ITensors.op(::OpName"σx", ::SiteType"S=1/2", s::Index) = 2 * op("Sx", s)
ITensors.op(::OpName"σy", ::SiteType"S=1/2", s::Index) = 2 * op("Sy", s)
ITensors.op(::OpName"σz", ::SiteType"S=1/2", s::Index) = 2 * op("Sz", s)

"""
    apply_depolarizing_noise(ψ::MPS, sites, pe::Float64) -> MPS

Apply depolarizing noise to an MPS with probability pe per site.
Each site independently has probability pe/3 of receiving X, Y, or Z error.
"""
function apply_depolarizing_noise(ψ::MPS, sites, pe::Float64)::MPS
    noise_ops = Tuple{String, Int64}[]

    for j in eachindex(sites)
        p = rand()
        if p < pe / 3
            push!(noise_ops, ("σx", j))
        elseif p < 2 * pe / 3
            push!(noise_ops, ("σy", j))
        elseif p < pe
            push!(noise_ops, ("σz", j))
        end
    end

    isempty(noise_ops) && return ψ

    noise_gates = ops(noise_ops, sites)
    return apply(noise_gates, ψ)
end