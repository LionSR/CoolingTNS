using ITensors
using Random: AbstractRNG, default_rng

function energy(ψ::MPS, H::MPO)
    return real(inner(ψ', H, ψ) / inner(ψ, ψ))
end

"""
    appendzeros_MPS(ψ::MPS, sites::Vector{<:Index}, coupling::String="XX")

Append bath qubits in appropriate ground state to system MPS.
Input: ψ is MPS on system sites (N sites) with arbitrary bond dimensions
       coupling determines the bath field through `get_bath_operator`
Output: MPS on interleaved sites [sys₁, bath₁, sys₂, bath₂, ...] (2N sites)

For product state input (D=1), creates proper interleaved product state.
For entangled input (D>1), preserves entanglement within system while adding
bath qubits in product state.
"""
function appendzeros_MPS(ψ::MPS, sites::Vector{<:Index}, coupling::String="XX")
    N = length(ψ)  # Number of system sites
    @assert length(sites) == interleaved_total_sites(N) "sites must have 2N elements for N system qubits"

    # Get bath ground state based on coupling type
    _, bath_amps = get_bath_ground_state(coupling)

    # Get original site indices
    orig_sites = siteinds(ψ)

    # Build tensors for the combined MPS
    # Structure: [sys₁]-[bath₁]-[sys₂]-[bath₂]-...-[sysₙ]-[bathₙ]
    tensors = Vector{ITensor}(undef, interleaved_total_sites(N))

    # Pre-create all new link indices
    # Combined MPS has 2N-1 links
    new_links = Vector{Index}(undef, interleaved_total_sites(N) - 1)

    # Determine bond dimensions
    for i in 1:N
        sys_pos = interleaved_system_site(i)
        bath_pos = interleaved_bath_site(i)

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
        if isnothing(right_link)
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
        sys_pos = interleaved_system_site(i)
        bath_pos = interleaved_bath_site(i)

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
    # Mutating/consuming version: avoid an extra full MPS copy.
    # This is safe in the cooling loop where the evolved system+bath MPS is
    # discarded after sampling.
    return _sample_bath_impl(rng, m; copy_input=false)
end

function sample_bath(m::MPS)
    return sample_bath(default_rng(), m)
end

function sample_bath(rng::AbstractRNG, m::MPS)
    # Non-mutating version (kept for general utility): work on a copy.
    return _sample_bath_impl(rng, m; copy_input=true)
end

function _sample_bath_impl(rng::AbstractRNG, m::MPS; copy_input::Bool)
    # Layout: [sys₁, bath₁, sys₂, bath₂, ..., sysₙ, bathₙ]
    N_total = length(m)
    N = div(N_total, 2)

    if abs(1.0 - norm(m)) > 1E-8
        error("sample_bath: MPS is not normalized, norm=$(norm(m))")
    end

    result = zeros(Int, N)
    m_working = copy_input ? copy(m) : m

    # Orthogonalize once to rightmost bath site (2N).
    # After this: sites 1..2N-1 are left-canonical, site 2N is the orth center.
    orthogonalize!(m_working, N_total)

    # Sample each bath site from right to left.
    # By preserving canonical form info (llim/rlim), each subsequent
    # orthogonalize! only needs to sweep ~1 site left instead of the full MPS.
    for bath_idx in N:-1:1
        bath_site = interleaved_bath_site(bath_idx)
        sys_site = interleaved_system_site(bath_idx)

        # Move orth center to bath_site (cheap: only sweeps from current center)
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
        A_sys = m_working[sys_site]
        A_combined = A_sys * An
        A_combined *= (1.0 / sqrt(max(pn, 1e-15)))

        # Remove the bath site from the MPS, preserving canonical form.
        # After contraction: sites 1..sys_site-1 are left-canonical,
        # sys_site holds the orth center, sites bath_site+1.. are right-canonical.
        current_len = length(m_working)
        if bath_site < current_len
            new_tensors = vcat(
                m_working[1:sys_site-1],
                [A_combined],
                m_working[bath_site+1:current_len]
            )
        else
            new_tensors = vcat(m_working[1:sys_site-1], [A_combined])
        end

        # Preserve canonical form: everything left of sys_site is left-canonical,
        # everything right of sys_site (in new indexing) is right-canonical.
        m_working = MPS(new_tensors, sys_site - 1, sys_site + 1)
    end

    if length(m_working) != N
        error("After bath sampling, MPS has incorrect length: expected $N, got $(length(m_working))")
    end

    normalize!(m_working)
    return result, m_working
end
