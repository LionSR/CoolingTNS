"""
    trotter.jl

Trotter circuit construction using multiple dispatch on HamiltonianModel and backend.
"""

using ITensors

# ============================================================================
# Bath-Only Trotter Circuit (model-agnostic)
# ============================================================================

"""
    build_trotter_circuit_bath_coupling(ham_params::HamiltonianParameters, backend::TNBackend, sites_sys, sites_bath, coupling_params, sim_params)

Build Trotter circuit for just the bath coupling terms. Model-agnostic since bath coupling is independent of system Hamiltonian.
"""
function build_trotter_circuit_bath_coupling(ham_params::HamiltonianParameters, ::TNBackend,
                                            sites_sys::Vector{<:Index}, sites_bath::Vector{<:Index},
                                            coupling_params::CouplingParameters, sim_params::UnifiedSimulationParameters)
    g, delta, coupling, tau = coupling_params.g, coupling_params.delta, coupling_params.coupling, sim_params.tau

    bath_op = get_bath_operator(coupling)

    gates = ITensor[]
    for ind in eachindex(sites_sys)
        s1, b1 = sites_sys[ind], sites_bath[ind]
        hb = delta / 2 * op(bath_op, b1)
        hsb = _tn_coupling_operator(s1, b1, coupling, g)
        push!(gates, exp(-1.0im * tau / 2 * hb))
        push!(gates, exp(-1.0im * tau / 2 * hsb))
    end
    append!(gates, reverse(gates))
    return gates
end

# ============================================================================
# Interleaved Trotter Circuit (for DM+Trotter on interleaved layout)
# ============================================================================

"""
    build_trotter_circuit_interleaved(ham_params::HamiltonianParameters, backend::TNBackend, sites, coupling_params, sim_params)

Build Trotter circuit on the interleaved layout [s1,b1,s2,b2,...,sN,bN] where
ALL gates act on adjacent sites. This avoids the non-adjacent gate problem
that occurs when system gates (e.g. ZZ on sites_sys[i], sites_sys[i+1]) skip
over bath qubits in the interleaved chain.

Gate structure (2nd-order symmetric Trotter):
  Forward half-step:
    1. 2-site gates on (si, bi): single-site system terms + bath H + coupling
    2. 3-site gates on (si, bi, s_{i+1}): ZZ interaction (J * Z_si ⊗ I_bi ⊗ Z_{s_{i+1}})
  Reverse half-step:
    3. Reverse of 3-site gates
    4. Reverse of 2-site gates
"""
function build_trotter_circuit_interleaved(ham_params::HamiltonianParameters, ::TNBackend,
                                           sites::Vector{<:Index}, coupling_params::CouplingParameters,
                                           sim_params::UnifiedSimulationParameters)
    error("build_trotter_circuit_interleaved not implemented for model $(typeof(ham_params.model))")
end

"""
    _tn_coupling_operator(sys_site, bath_site, coupling, g)

Return the local TN coupling operator `g * sum(O_S * O_B)` using the shared
`coupling_operator_terms` convention; for example, `"XY"` gives
`g * (X_S * Y_B + Y_S * X_B)`.
"""
function _tn_coupling_operator(sys_site::Index, bath_site::Index, coupling::String, g::Float64)
    return sum(
        g * op(sys_op, sys_site) * op(bath_op, bath_site)
        for (sys_op, bath_op) in coupling_operator_terms(coupling)
    )
end

function build_trotter_circuit_interleaved(ham_params::HamiltonianParameters{IsingModel},
                                           ::TNBackend, sites::Vector{<:Index},
                                           coupling_params::CouplingParameters,
                                           sim_params::UnifiedSimulationParameters)
    N = ham_params.N
    J, h = ham_params.params.J, ham_params.params.h
    g, delta, coupling, tau = coupling_params.g, coupling_params.delta, coupling_params.coupling, sim_params.tau
    bath_op = get_bath_operator(coupling)

    forward_gates = ITensor[]

    # Layer 1: 2-site gates on (s_i, b_i) at positions (2i-1, 2i)
    # Contains: h*X_{si} + (Δ/2)*bath_op_{bi} + g*op1_{si}*op2_{bi}
    for i in 1:N
        si = sites[2i-1]
        bi = sites[2i]
        h_local = h * op("X", si) * op("I", bi) +
                  delta / 2 * op("I", si) * op(bath_op, bi) +
                  _tn_coupling_operator(si, bi, coupling, g)
        push!(forward_gates, exp(-1.0im * tau / 2 * h_local))
    end

    # Layer 2: 3-site gates on (s_i, b_i, s_{i+1}) at positions (2i-1, 2i, 2i+1)
    # Contains: J * Z_{si} ⊗ I_{bi} ⊗ Z_{s_{i+1}}
    for i in 1:N-1
        si = sites[2i-1]
        bi = sites[2i]
        si1 = sites[2i+1]
        h_zz = J * op("Z", si) * op("I", bi) * op("Z", si1)
        push!(forward_gates, exp(-1.0im * tau / 2 * h_zz))
    end

    # Symmetric 2nd-order: forward + reverse
    gates = vcat(forward_gates, reverse(forward_gates))
    return gates
end

function build_trotter_circuit_interleaved(ham_params::HamiltonianParameters{NiIsingModel},
                                           ::TNBackend, sites::Vector{<:Index},
                                           coupling_params::CouplingParameters,
                                           sim_params::UnifiedSimulationParameters)
    N = ham_params.N
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    g, delta, coupling, tau = coupling_params.g, coupling_params.delta, coupling_params.coupling, sim_params.tau
    bath_op = get_bath_operator(coupling)

    forward_gates = ITensor[]

    # Layer 1: 2-site gates on (s_i, b_i) at positions (2i-1, 2i)
    # Contains: hx*X_{si} + hz*Z_{si} + (Δ/2)*bath_op_{bi} + g*op1_{si}*op2_{bi}
    for i in 1:N
        si = sites[2i-1]
        bi = sites[2i]
        h_local = hx * op("X", si) * op("I", bi) +
                  hz * op("Z", si) * op("I", bi) +
                  delta / 2 * op("I", si) * op(bath_op, bi) +
                  _tn_coupling_operator(si, bi, coupling, g)
        push!(forward_gates, exp(-1.0im * tau / 2 * h_local))
    end

    # Layer 2: 3-site gates on (s_i, b_i, s_{i+1}) at positions (2i-1, 2i, 2i+1)
    # Contains: J * Z_{si} ⊗ I_{bi} ⊗ Z_{s_{i+1}}
    for i in 1:N-1
        si = sites[2i-1]
        bi = sites[2i]
        si1 = sites[2i+1]
        h_zz = J * op("Z", si) * op("I", bi) * op("Z", si1)
        push!(forward_gates, exp(-1.0im * tau / 2 * h_zz))
    end

    # Symmetric 2nd-order: forward + reverse
    gates = vcat(forward_gates, reverse(forward_gates))
    return gates
end
