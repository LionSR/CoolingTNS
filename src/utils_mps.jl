using ITensors
using Random: AbstractRNG, default_rng

function energy(ψ::MPS, H::MPO)
    return real(inner(ψ', H, ψ) / inner(ψ, ψ))
end

"""
    get_bath_ground_state(coupling::String) -> (String, Vector{Float64})

Return the bath ground state name and amplitudes based on coupling type.
Bath Hamiltonian must NOT commute with coupling for energy transfer.

With Δ > 0 and bath H = (Δ/2) * Op:
- Ground state is eigenvalue -1 of Op
- XX, XY, XZ coupling → bath H = (Δ/2)Z → ground state |↓⟩ (Z=-1)
- ZZ, YZ coupling → bath H = (Δ/2)X → ground state |−⟩ (X=-1)

Bath absorbs energy |Δ| when excited from ground to excited state.
"""
function get_bath_ground_state(coupling::String)
    if coupling in ["ZZ", "YZ"]
        # X-basis ground state: |−⟩ = (|↑⟩ - |↓⟩)/√2 (eigenvalue X = -1)
        # In S=1/2 convention: state 1 = |↑⟩, state 2 = |↓⟩
        return "X-", [1/sqrt(2), -1/sqrt(2)]
    else
        # Z-basis ground state: |↓⟩ (state 2, eigenvalue Z = -1)
        return "Dn", [0.0, 1.0]
    end
end

"""
    appendzeros_MPS(ψ::MPS, sites::Vector{<:Index}, coupling::String="XX")

Append bath qubits in appropriate ground state to system MPS.
Input: ψ is MPS on system sites (N sites) with arbitrary bond dimensions
       coupling determines bath basis (XX→Z-basis bath, ZZ→X-basis bath)
Output: MPS on interleaved sites [sys₁, bath₁, sys₂, bath₂, ...] (2N sites)

For product state input (D=1), creates proper interleaved product state.
For entangled input (D>1), preserves entanglement within system while adding
bath qubits in product state.
"""
function appendzeros_MPS(ψ::MPS, sites::Vector{<:Index}, coupling::String="XX")
    N = length(ψ)  # Number of system sites
    @assert length(sites) == 2*N "sites must have 2N elements for N system qubits"

    # Get bath ground state based on coupling type
    _, bath_amps = get_bath_ground_state(coupling)

    # Get original site indices
    orig_sites = siteinds(ψ)

    # Build tensors for the combined MPS
    # Structure: [sys₁]-[bath₁]-[sys₂]-[bath₂]-...-[sysₙ]-[bathₙ]
    tensors = Vector{ITensor}(undef, 2*N)

    # Pre-create all new link indices
    # Combined MPS has 2N-1 links
    new_links = Vector{Index}(undef, 2*N-1)

    # Determine bond dimensions
    for i in 1:N
        sys_pos = 2*i - 1
        bath_pos = 2*i

        if i < N
            # Link between sys_i and bath_i has dimension = dim of original link i
            D = dim(linkind(ψ, i))
            new_links[sys_pos] = Index(D, "Link,l=$sys_pos")
            # Link between bath_i and sys_{i+1} also has dimension D
            new_links[bath_pos] = Index(D, "Link,l=$bath_pos")
        else
            # Last sys-bath pair: dimension 1 link
            new_links[sys_pos] = Index(1, "Link,l=$sys_pos")
        end
    end

    # Helper function to create bath tensor with correct ground state
    function make_bath_tensor(s_bath, left_link, right_link=nothing)
        if right_link === nothing
            # End tensor: only left link
            T = ITensor(ComplexF64, s_bath, left_link)
            for (state_idx, amp) in enumerate(bath_amps)
                if abs(amp) > 1e-15
                    T[s_bath => state_idx, left_link => 1] = amp
                end
            end
        else
            # Middle tensor: identity on bonds, bath state on site
            D = dim(left_link)
            T = ITensor(ComplexF64, s_bath, left_link, right_link)
            for d in 1:D
                for (state_idx, amp) in enumerate(bath_amps)
                    if abs(amp) > 1e-15
                        T[s_bath => state_idx, left_link => d, right_link => d] = amp
                    end
                end
            end
        end
        return T
    end

    # Build each tensor
    for i in 1:N
        sys_pos = 2*i - 1
        bath_pos = 2*i

        # Get system tensor and change its site index
        T_sys = copy(ψ[i])
        T_sys = replaceind(T_sys, orig_sites[i] => sites[sys_pos])

        # Replace link indices
        if i == 1 && N == 1
            # Single site system
            tensors[sys_pos] = T_sys * delta(new_links[sys_pos])
            tensors[bath_pos] = make_bath_tensor(sites[bath_pos], new_links[sys_pos])
        elseif i == 1
            # First site: only right link
            l_right_old = linkind(ψ, 1)
            T_sys = replaceind(T_sys, l_right_old => new_links[sys_pos])
            tensors[sys_pos] = T_sys
            tensors[bath_pos] = make_bath_tensor(sites[bath_pos], new_links[sys_pos], new_links[bath_pos])
        elseif i == N
            # Last site: only left link
            l_left_old = linkind(ψ, N-1)
            T_sys = replaceind(T_sys, l_left_old => new_links[sys_pos-1])
            T_sys = T_sys * delta(new_links[sys_pos])
            tensors[sys_pos] = T_sys
            tensors[bath_pos] = make_bath_tensor(sites[bath_pos], new_links[sys_pos])
        else
            # Middle site: both links
            l_left_old = linkind(ψ, i-1)
            l_right_old = linkind(ψ, i)
            T_sys = replaceind(T_sys, l_left_old => new_links[sys_pos-1])
            T_sys = replaceind(T_sys, l_right_old => new_links[sys_pos])
            tensors[sys_pos] = T_sys
            tensors[bath_pos] = make_bath_tensor(sites[bath_pos], new_links[sys_pos], new_links[bath_pos])
        end
    end

    result = MPS(tensors)
    orthogonalize!(result, 1)
    normalize!(result)

    return result
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
    # Layout: [sys₁, bath₁, sys₂, bath₂, ..., sysₙ, bathₙ]
    # Bath sites are at even indices: 2, 4, 6, ..., 2N
    # System sites are at odd indices: 1, 3, 5, ..., 2N-1
    N_total = length(m)
    N = div(N_total, 2)

    if abs(1.0 - norm(m)) > 1E-8
        error("sample_bath: MPS is not normalized, norm=$(norm(m))")
    end

    result = zeros(Int, N)
    m_working = copy(m)

    # Sample each bath site from right to left to avoid reindexing issues
    # Bath sites: 2N, 2N-2, ..., 4, 2
    for bath_idx in N:-1:1
        bath_site = 2 * bath_idx  # Even indices: 2, 4, 6, ..., 2N
        sys_site = 2 * bath_idx - 1  # Odd indices: 1, 3, 5, ..., 2N-1

        # Orthogonalize to bath site
        orthogonalize!(m_working, bath_site)
        s = siteind(m_working, bath_site)
        d = dim(s)

        # Sample the bath measurement outcome
        pdisc = 0.0
        r = rand(rng)
        n = 1
        An = ITensor()
        pn = 0.0
        A = m_working[bath_site]

        while n <= d
            projn = ITensor(s)
            projn[s=>n] = 1.0
            An = A * dag(projn)
            pn = real(scalar(dag(An) * An))
            pdisc += pn
            (r < pdisc) && break
            n += 1
        end
        result[bath_idx] = n

        # Contract the projected bath tensor with the system tensor
        # This effectively traces out the bath qubit
        A_sys = m_working[sys_site]
        A_combined = A_sys * An
        A_combined *= (1.0 / sqrt(max(pn, 1e-15)))

        # Remove the bath site from the MPS
        current_len = length(m_working)
        if bath_site < current_len
            # Bath site is not at the end
            new_tensors = vcat(
                m_working[1:sys_site-1],
                [A_combined],
                m_working[bath_site+1:current_len]
            )
        else
            # Bath site is at the end
            new_tensors = vcat(m_working[1:sys_site-1], [A_combined])
        end
        m_working = MPS(new_tensors)
    end

    # After sampling, m_working should contain only system sites
    if length(m_working) != N
        error("After bath sampling, MPS has incorrect length: expected $N, got $(length(m_working))")
    end

    normalize!(m_working)
    return result, m_working
end
