using ITensors

# get_bath_ground_state is defined in utils_mps.jl, which is included before this file

"""
    appendzeros_MPO(ρ::MPO, sites::Vector{<:Index}, coupling::String="XX")

Append bath qubits in appropriate ground state density matrix to system MPO.
The bath state is the ground state of the bath field selected by
`get_bath_operator(coupling)`.
"""
function appendzeros_MPO(ρ::MPO, sites::Vector{<:Index}, coupling::String="XX")
    N = length(ρ)

    # Get bath ground state based on coupling type
    _, bath_amps = get_bath_ground_state(coupling)

    ρ_appended = MPO(sites, "Id")
    data_ρ = ITensors.data(ρ)
    dataρ_appended = ITensors.data(ρ_appended)

    for i = 1:N
        # Create bath density matrix |ψ₀⟩⟨ψ₀| from amplitudes
        s = sites[2i]
        ψ0 = ITensor(ComplexF64, s)
        for (state_idx, amp) in enumerate(bath_amps)
            if abs(amp) > 1e-15
                ψ0[s => state_idx] = amp
            end
        end
        ρ0 = ψ0 * prime(dag(ψ0), s)

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
    # Keep interlaced system+bath MPO (length = 2N)
    return MPO(dataρ_appended)
end

function partial_trace_bath(ρ_sb::MPO, sites::Vector{<:Index}, sites_sys::Vector{<:Index})
    N = length(sites_sys)
    # Trace out bath (even sites) and merge system+bath tensors back into N-site MPO
    return MPO([ρ_sb[2i-1] * ρ_sb[2i] * delta(sites[2i], sites[2i]') for i in 1:N])
end
