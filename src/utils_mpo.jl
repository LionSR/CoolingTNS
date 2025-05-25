using ITensors

function appendzeros_MPO(ρ::MPO, sites::Vector{<:Index})
    N = length(ρ)
    ρ_appended = MPO(sites, "Id")
    data_ρ = ITensors.data(ρ)
    dataρ_appended = ITensors.data(ρ_appended)
    for i = 1:N
        # Bath in excited state |↓⟩ = |1⟩ for cooling
        ψ0 = onehot(sites[2i] => 2)  # Changed from 1 to 2 for excited state
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

function partial_trace_bath(ρ_sb::MPO, sites::Vector{<:Index}, sites_sys::Vector{<:Index})
    N = length(sites_sys)
    ρ_s = MPO([ρ_sb[i] * delta(sites[2i], sites[2i]') for i in 1:N])
    ρ_s
end

# Alias for consistency with cooling_evolution_dispatch.jl
const appendbath_MPO = appendzeros_MPO

"""
    rdm_mpo(ρ::MPO, sites, site_indices)

Compute reduced density matrix by tracing out unwanted sites.
"""
function rdm_mpo(ρ::MPO, sites::Vector{<:Index}, site_indices)
    # For now, use partial trace of bath (assuming we want system only)
    # This is a simplified implementation
    sites_sys = sites[site_indices]
    return partial_trace_bath(ρ, sites, sites_sys)
end