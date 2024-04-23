using ITensors

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