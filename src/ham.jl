using ITensors

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


function ham_ising_sys_bath(N, sites, ham_params, g, Δ, coupling)
    J, h = ham_params
    op1, op2 = if length(coupling) == 2
        string(coupling[1]), string(coupling[2])
    else
        error("Invalid coupling: $coupling")
    end

    ampo = OpSum()
    for ind = 1:(N-1)
        s1 = 2ind - 1
        s2 = 2ind + 1
        b1 = 2ind
        ampo += J, "Z", s1, "Z", s2
        ampo += h, "X", s1
        ampo += -Δ / 2, "Z", b1
        ampo += g, op1, s1, op2, b1
    end
    sN = 2N - 1
    bN = 2N
    ampo += h, "X", sN
    ampo += -Δ / 2, "Z", bN
    ampo += g, op1, sN, op2, bN
    return MPO(ampo, sites)
end


function ham_niising_sys_bath(N, sites, ham_params, g, Δ, coupling)
    J, hx, hz = ham_params
    op1, op2 = if length(coupling) == 2
        string(coupling[1]), string(coupling[2])
    else
        error("Invalid coupling: $coupling")
    end

    ampo = OpSum()
    for ind = 1:(N-1)
        s1 = 2ind - 1
        s2 = 2ind + 1
        b1 = 2ind
        ampo += J, "Z", s1, "Z", s2
        ampo += hx, "X", s1
        ampo += hz, "Z", s1
        ampo += -Δ / 2, "Z", b1
        ampo += g, op1, s1, op2, b1
    end
    sN = 2N - 1
    bN = 2N
    ampo += hx, "X", sN
    ampo += hz, "Z", sN
    ampo += -Δ / 2, "Z", bN
    ampo += g, op1, sN, op2, bN
    return MPO(ampo, sites)
end


function heisenberg(N)
    os = OpSum()
    for j = 1:(N-1)
        os += 0.5, "S+", j, "S-", j + 1
        os += 0.5, "S-", j, "S+", j + 1
        os += "Sz", j, "Sz", j + 1
    end
    return os
end
