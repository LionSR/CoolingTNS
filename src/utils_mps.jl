using ITensors
using Random: AbstractRNG, default_rng

function energy(ψ::MPS, H::MPO)
    return real(inner(ψ', H, ψ) / inner(ψ, ψ))
end

function appendzeros_MPS(ψ::MPS, sites)
    N = length(ψ)
    ψ_appended = MPS(sites, "↓")
    for i = 1:N
        ψ0 = onehot(sites[2i] => 1)
        if i < N
            ll = sim(linkind(ψ, i))
            lr = linkind(ψ, i)
            δlr = delta(dag(lr), ll)
            ψ_appended[2i-1] = ψ[i] * δlr
            ψ_appended[2i] = ψ0 * dag(δlr)
        else
            ll = linkind(ψ_appended, 2i - 1)
            lT = ITensor(1, ll)
            ψ_appended[2i-1] = ψ[i] * lT
            ψ_appended[2i] = ψ0 * dag(lT)
        end
    end
    return ψ_appended
end

function sample_bath!(m::MPS)
    return sample_bath!(default_rng(), m)
end

function sample_bath!(rng::AbstractRNG, m::MPS)
    orthogonalize!(m, 2)
    return sample_bath(rng, m)
end

function sample_bath(m::MPS)
    return sample_bath(default_rng(), m)
end

function sample_bath(rng::AbstractRNG, m::MPS)
    N = div(length(m), 2)

    if abs(1.0 - norm(m[2])) > 1E-8
        error("sample: MPS is not normalized, norm=$(norm(m[2]))")
    end

    result = zeros(Int, N)
    m_rest = deepcopy(m)

    for i = 1:N
        j = i + 1
        orthogonalize!(m_rest, j)
        s = siteind(m_rest, j)
        d = dim(s)
        pdisc = 0.0
        r = rand(rng)
        n = 1
        An = ITensor()
        pn = 0.0
        A = m_rest[j]

        while n <= d
            projn = ITensor(s)
            projn[s=>n] = 1.0
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn
            (r < pdisc) && break
            n += 1
        end
        result[i] = n

        A = m_rest[i] * An
        A *= (1.0 / sqrt(pn))
        if i < N
            m_rest = MPS(vcat(m_rest[1:i-1], [A], m_rest[j+1:end]))
        else
            m_rest = MPS(vcat(m_rest[1:i-1], [A]))
        end
        # println("iter $i, n=$n, pn=$pn, norm(m_rest)=$(norm(m_rest))")
    end
    return result, m_rest
end
