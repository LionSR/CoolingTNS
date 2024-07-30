using ITensors

function parse_coupling(coupling::String)
    if length(coupling) == 2 && coupling ⊆ "XYZ"
        return string(coupling[1]), string(coupling[2])
    else
        throw(ArgumentError("Invalid coupling: $coupling. Expected two-character string with X, Y, or Z."))
    end
end

function ham_ising(N, sites, ham_params)
    J, h = ham_params
    ampo = OpSum()
    for j = 1:(N-1)
        ampo .+= J, "Z", j, "Z", j + 1
    end
    for j = 1:N
        ampo .+= h, "X", j
    end
    H = MPO(ampo, sites)
    return H
end


function ham_niising(N, sites, ham_params)
    J, hx, hz = ham_params
    ampo = OpSum()
    for j = 1:(N-1)
        ampo .+= J, "Z", j, "Z", j + 1
    end
    for j = 1:N
        ampo .+= hx, "X", j
    end
    for j = 1:N
        ampo .+= hz, "Z", j
    end
    H = MPO(ampo, sites)
    return H
end


function ham_rydberg_dressed_ising(N, sites, ham_params)
    Ω, Δ, V0, Rc = ham_params
    ampo = OpSum()

    # Rabi oscillation terms
    for i = 1:N
        ampo += Ω / 2, "X", i
        ampo += -Δ, "ProjUp", i
    end

    # Long-range Ising interaction
    for i = 1:N
        for j = (i+1):N
            distance = abs(i - j)
            Vij = V0 / (1 + (distance / Rc)^6)
            ampo += Vij, "ProjUp", i, "ProjUp", j
        end
    end

    H = MPO(ampo, sites)
    return H
end


function ham_ising_sys_bath(N, sites, ham_params, coupling_params)
    J, h = ham_params
    g = coupling_params["g"]
    Δ = coupling_params["Δ"]
    op1, op2 = parse_coupling(coupling_params["coupling"])

    ampo = OpSum()
    for ind = 1:N
        s1 = 2ind - 1
        b1 = 2ind
        if ind < N
            s2 = 2ind + 1
            ampo += J, "Z", s1, "Z", s2
        end
        ampo += h, "X", s1
        ampo += -Δ / 2, "Z", b1
        ampo += g, op1, s1, op2, b1
    end
    return MPO(ampo, sites)
end


function ham_niising_sys_bath(N, sites, ham_params, coupling_params)
    J, hx, hz = ham_params
    g = coupling_params["g"]
    Δ = coupling_params["Δ"]
    op1, op2 = parse_coupling(coupling_params["coupling"])

    ampo = OpSum()
    for ind = 1:N
        s1 = 2ind - 1
        b1 = 2ind
        if ind < N
            s2 = 2ind + 1
            ampo += J, "Z", s1, "Z", s2
        end
        ampo += hx, "X", s1
        ampo += hz, "Z", s1
        ampo += -Δ / 2, "Z", b1
        ampo += g, op1, s1, op2, b1
    end
    return MPO(ampo, sites)
end

function ham_rydberg_dressed_ising_sys_bath(N, sites, ham_params, coupling_params)
    Ω, Δ, V0, Rc = ham_params
    g = coupling_params["g"]
    Δb = coupling_params["Δ"]
    op1, op2 = parse_coupling(coupling_params["coupling"])

    ampo = OpSum()

    # Precompute distances and interaction strengths
    Vij = [V0 / (1 + (d/Rc)^6) for d in 1:(N-1)]

    # System terms (Rydberg dressed Ising)
    for i = 1:N
        s = 2i - 1
        ampo += Ω/2, "X", s
        ampo += -Δ, "ProjUp", s
        
        # Long-range interactions
        for j = (i+1):N
            s2 = 2j - 1
            ampo += Vij[j-i], "ProjUp", s, "ProjUp", s2
        end
    end

    # Bath terms and system-bath coupling
    for i = 1:N
        s = 2i - 1
        b = 2i
        ampo += -Δb/2, "Z", b
        ampo += g, op1, s, op2, b
    end

    return MPO(ampo, sites)
end