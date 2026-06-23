using ITensors

# get_bath_ground_state is defined with the shared coupling convention in coupling_utils.jl.

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
        s = sites[interleaved_bath_site(i)]
        ψ0 = ITensor(ComplexF64, s)
        for (state_idx, amp) in enumerate(bath_amps)
            if abs(amp) > 1e-15
                ψ0[s => state_idx] = amp
            end
        end
        # MPO convention: primed site index is the matrix row, unprimed is column.
        ρ0 = prime(ψ0, s) * dag(ψ0)

        if i < N
            ll = sim(linkind(ρ, i))
            lr = linkind(ρ, i)
            δlr = delta(dag(lr), ll)
            dataρ_appended[interleaved_system_site(i)] = data_ρ[i] * δlr
            dataρ_appended[interleaved_bath_site(i)] = ρ0 * dag(δlr)
        else
            ll = linkind(ρ_appended, interleaved_system_site(i))
            lT = ITensor(1, ll)
            dataρ_appended[interleaved_system_site(i)] = data_ρ[i] * lT
            dataρ_appended[interleaved_bath_site(i)] = ρ0 * dag(lT)
        end
    end
    # Keep interlaced system+bath MPO (length = 2N)
    return MPO(dataρ_appended)
end

function partial_trace_bath(ρ_sb::MPO, sites::Vector{<:Index}, N_sys::Int)
    # Trace out bath (even sites) and merge system+bath tensors back into N-site MPO
    return MPO([
        ρ_sb[interleaved_system_site(i)] *
        ρ_sb[interleaved_bath_site(i)] *
        delta(sites[interleaved_bath_site(i)], sites[interleaved_bath_site(i)]')
        for i in 1:N_sys
    ])
end

function partial_trace_system(ρ_sb::MPO, sites::Vector{<:Index}, N_bath::Int)
    # Trace out system (odd sites) and merge system+bath tensors back into N-site MPO
    return MPO([
        ρ_sb[interleaved_system_site(i)] *
        ρ_sb[interleaved_bath_site(i)] *
        delta(sites[interleaved_system_site(i)], sites[interleaved_system_site(i)]')
        for i in 1:N_bath
    ])
end

"""
Trace out the bath and return the canonical reduced system density MPO.

The returned MPO uses the post-trace representative implemented by
`_canonical_reduced_density_mpo`.
"""
function _reduced_system_density_mpo(ρ_sb::MPO, sites::Vector{<:Index}, N_sys::Int)
    return _canonical_reduced_density_mpo(partial_trace_bath(ρ_sb, sites, N_sys))
end

"""
Trace out the system and return the canonical reduced bath density MPO.

The returned MPO uses the post-trace representative implemented by
`_canonical_reduced_density_mpo`.
"""
function _reduced_bath_density_mpo(ρ_sb::MPO, sites::Vector{<:Index}, N_bath::Int)
    return _canonical_reduced_density_mpo(partial_trace_system(ρ_sb, sites, N_bath))
end

"""
    _canonical_reduced_density_mpo(ρ::MPO)

Project an MPO reduced-density operator to its Hermitian, trace-one
representative after a partial trace.
"""
function _canonical_reduced_density_mpo(ρ::MPO)
    ρ_h = 0.5 * (ρ + dag(swapprime(ρ, 0, 1)))
    trρ = tr(ρ_h)
    # TN evolution may include MPO truncation before this boundary.  Keep the
    # existing TN behavior by renormalizing the Hermitian representative rather
    # than rejecting trace drift that is not present in exact ED arithmetic.
    return ρ_h / trρ
end
