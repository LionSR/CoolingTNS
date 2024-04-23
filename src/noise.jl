using ITensors

ITensors.op(::OpName"σx", ::SiteType"S=1/2", s::Index) = 2 * op("Sx", s)

ITensors.op(::OpName"σy", ::SiteType"S=1/2", s::Index) = 2 * op("Sy", s)

ITensors.op(::OpName"σz", ::SiteType"S=1/2", s::Index) = 2 * op("Sz", s)


function apply_depolarizing_noise(ψ, sites, pe)
    for j in eachindex(sites)
        p = rand()
        if p < pe / 3
            ψ = ITensors.product(Op(sites[j], "σx"), ψ)
        elseif p < 2 * pe / 3
            ψ = ITensors.product(Op(sites[j], "σy"), ψ)
        elseif p < pe
            ψ = ITensors.product(Op(sites[j], "σz"), ψ)
        end
    end
    return ψ
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