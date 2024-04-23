using ITensors

function appendzeros_MPO(ρ::MPO, sites,)
    N = length(ρ)
    ρ_appended = MPO(sites, "Id")
    data_ρ = ITensors.data(ρ)
    dataρ_appended = ITensors.data(ρ_appended)
    for i = 1:N
        ψ0 = onehot(sites[2i] => 1)
        ρ0 = ψ0 * ψ0'
        if i < N
            ll = sim(linkind(ρ, i))
            lr = linkind(ρ, i)
            δlr = delta(dag(lr), ll)
            dataρ_appended[2i-1] = data_ρ[i] * δlr
            dataρ_appended[2i] = ρ0 * dag(δlr)
        else
            ll = linkind(ρ_appended, 2i - 1)
            lT = ITensor(1, ll)
            dataρ_appended[2i-1] = data_ρ[i] * lT
            dataρ_appended[2i] = ρ0 * dag(lT)
        end
    end
    # ρ_appended = MPO(dataρ_appended)
    ρ_appended = MPO([dataρ_appended[2j-1] * dataρ_appended[2j] for j in 1:N])
    ρ_appended
end

function partial_trace_bath(ρ_sb::MPO, sites, sites_sys)
    N = length(sites_sys)
    ρ_s = MPO([ρ_sb[i] * delta(sites[2i], sites[2i]') for i in 1:N])
    ρ_s
end