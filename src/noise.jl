using ITensors

ITensors.op(::OpName"σx", ::SiteType"S=1/2", s::Index) = 2 * op("Sx", s)
ITensors.op(::OpName"σy", ::SiteType"S=1/2", s::Index) = 2 * op("Sy", s)
ITensors.op(::OpName"σz", ::SiteType"S=1/2", s::Index) = 2 * op("Sz", s)


function apply_depolarizing_noise(ψ::MPS, sites, pe)
    os = Tuple{String, Int64}[]
    for j in eachindex(sites)
        p = rand()
        if p < pe / 3
            push!(os, ("σx", j))
        elseif pe / 3 <= p < 2 * pe / 3
            push!(os, ("σy", j))
        elseif 2 * pe / 3 <= p < pe
            push!(os, ("σz", j))
        end
    end
    if length(os) > 0
        noise_gates = ops(os, sites)
        return apply(noise_gates, ψ)
    end
    return ψ
end

function depolarizing_noise(site, pe)
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
    depl_op = ITensors.itensor(kraus, prime.(site), ITensors.dag.(site), krausind)
    return depl_op
end