using ITensors

ITensors.op(::OpName"σx", ::SiteType"S=1/2", s::Index) = 2 * op("Sx", s)
ITensors.op(::OpName"σy", ::SiteType"S=1/2", s::Index) = 2 * op("Sy", s)
ITensors.op(::OpName"σz", ::SiteType"S=1/2", s::Index) = 2 * op("Sz", s)


function apply_depolarizing_noise(ψ::MPS, sites, pe)
    # os = []
    os = Tuple{String, Int64}[]
    for j in eachindex(sites)
        p = rand()
        if p < pe / 3
            append!(os, ("σx", j))
        elseif p < 2 * pe / 3
            append!(os, ("σy", j))
        elseif p < pe
            append!(os, ("σz", j))
        end
    end
    println("sites: ", typeof(sites))
    println("os: ", typeof(os))
    noise_gates = ITensors.ops(os, sites)
    println("noise_gates: ", typeof(noise_gates))
    return apply(noise_gates, ψ)
end

function depolarizing_noise(pe, s)
    kraus = zeros(Complex{Float64}, 2, 2, 4)
    kraus[:, :, 1] = sqrt(1 - 3 / 4 * pe) * [
        1 0
        0 1
    ]
    kraus[:, :, 2] = sqrt(pe / 4) * [
        0 1
        1 0
    ]
    kraus[:, :, 3] = sqrt(pe / 4) * [
        0.0 -1.0im
        1.0im 0.0
    ]
    kraus[:, :, 4] = sqrt(pe / 4) * [
        1 0
        0 -1
    ]
    krausind = Index(size(kraus, 3); tags="kraus")
    depl_op = ITensors.itensor(kraus, prime.(s), ITensors.dag.(s), krausind)
    return depl_op
end