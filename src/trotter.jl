"""
    trotter.jl

Trotter circuit construction using multiple dispatch on HamiltonianModel and backend.
"""

using ITensors

# ============================================================================
# Helper Functions
# ============================================================================

"""
    build_bath_coupling_gate(s::Index, b::Index, g::Float64, delta::Float64, op1::String, op2::String, tau::Float64) -> ITensor

Build a single system-bath coupling gate: exp(-i * tau/2 * (g * op1_s * op2_b + delta/2 * Z_b))
"""
function build_bath_coupling_gate(s::Index, b::Index, g::Float64, delta::Float64, op1::String, op2::String, tau::Float64)::ITensor
    hsb = g * op(op1, s) * op(op2, b) + delta / 2 * op("I", s) * op("Z", b)
    return exp(-1.0im * tau / 2 * hsb)
end

# ============================================================================
# Trotter Circuit Construction
# ============================================================================

"""
    build_trotter_circuit(ham_params::HamiltonianParameters, backend::CoolingBackend, sites_sys, sites_bath, coupling_params, sim_params)

Generic interface for building Trotter circuits using double dispatch.
"""
function build_trotter_circuit(ham_params::HamiltonianParameters, backend::CoolingBackend, sites_sys::Vector{<:Index}, sites_bath::Vector{<:Index}, coupling_params, sim_params)
    error("build_trotter_circuit not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

function build_trotter_circuit(ham_params::HamiltonianParameters{IsingModel},
                              ::TNBackend, sites_sys::Vector{<:Index}, sites_bath::Vector{<:Index},
                              coupling_params::CouplingParameters, sim_params::UnifiedSimulationParameters)
    N = ham_params.N
    J, h = ham_params.params.J, ham_params.params.h
    g, delta, coupling, tau = coupling_params.g, coupling_params.delta, coupling_params.coupling, sim_params.tau
    op1, op2 = parse_coupling(coupling)

    gates = ITensor[]
    for ind in eachindex(sites_sys)
        s1, b1 = sites_sys[ind], sites_bath[ind]

        # System Hamiltonian terms
        hs = if ind < N
            J * op("Z", s1) * op("Z", sites_sys[ind+1]) + h * op("X", s1) * op("I", sites_sys[ind+1])
        else
            h * op("X", s1)
        end

        push!(gates, exp(-1.0im * tau / 2 * hs))
        push!(gates, build_bath_coupling_gate(s1, b1, g, delta, op1, op2, tau))
    end
    append!(gates, reverse(gates))
    return gates
end

function build_trotter_circuit(ham_params::HamiltonianParameters{NiIsingModel},
                              ::TNBackend, sites_sys::Vector{<:Index}, sites_bath::Vector{<:Index},
                              coupling_params::CouplingParameters, sim_params::UnifiedSimulationParameters)
    N = ham_params.N
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    g, delta, coupling, tau = coupling_params.g, coupling_params.delta, coupling_params.coupling, sim_params.tau
    op1, op2 = parse_coupling(coupling)

    gates = ITensor[]
    for ind in eachindex(sites_sys)
        s1, b1 = sites_sys[ind], sites_bath[ind]

        # System Hamiltonian terms
        hs = if ind < N
            J * op("Z", s1) * op("Z", sites_sys[ind+1]) + hx * op("X", s1) * op("I", sites_sys[ind+1]) + hz * op("Z", s1) * op("I", sites_sys[ind+1])
        else
            hx * op("X", s1) + hz * op("Z", s1)
        end

        push!(gates, exp(-1.0im * tau / 2 * hs))
        push!(gates, build_bath_coupling_gate(s1, b1, g, delta, op1, op2, tau))
    end
    append!(gates, reverse(gates))
    return gates
end

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
    op1, op2 = parse_coupling(coupling)

    gates = ITensor[]
    for ind in eachindex(sites_sys)
        s1, b1 = sites_sys[ind], sites_bath[ind]
        hb = delta / 2 * op("Z", b1)
        hsb = g * op(op1, s1) * op(op2, b1)
        push!(gates, exp(-1.0im * tau / 2 * hb))
        push!(gates, exp(-1.0im * tau / 2 * hsb))
    end
    append!(gates, reverse(gates))
    return gates
end

# ============================================================================
# System-Only Trotter Circuit (for density matrix evolution)
# ============================================================================

"""
    build_system_trotter_circuit(ham_params::HamiltonianParameters, sites_sys, sim_params)

Build Trotter circuit for system-only evolution.
"""
function build_system_trotter_circuit(ham_params::HamiltonianParameters, sites_sys::Vector{<:Index}, sim_params::UnifiedSimulationParameters)
    error("build_system_trotter_circuit not implemented for model $(typeof(ham_params.model))")
end

function build_system_trotter_circuit(ham_params::HamiltonianParameters{IsingModel},
                                     sites_sys::Vector{<:Index}, sim_params::UnifiedSimulationParameters)
    N, tau = ham_params.N, sim_params.tau
    J, h = ham_params.params.J, ham_params.params.h

    gates = ITensor[]

    # Two-site ZZ terms
    for i in 1:N-1
        push!(gates, exp(-1.0im * tau / 2 * J * op("Z", sites_sys[i]) * op("Z", sites_sys[i+1])))
    end
    # One-site X terms
    for i in 1:N
        push!(gates, exp(-1.0im * tau / 2 * h * op("X", sites_sys[i])))
    end

    append!(gates, reverse(gates))
    return gates
end

function build_system_trotter_circuit(ham_params::HamiltonianParameters{NiIsingModel},
                                     sites_sys::Vector{<:Index}, sim_params::UnifiedSimulationParameters)
    N, tau = ham_params.N, sim_params.tau
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz

    gates = ITensor[]

    # Two-site ZZ terms
    for i in 1:N-1
        push!(gates, exp(-1.0im * tau / 2 * J * op("Z", sites_sys[i]) * op("Z", sites_sys[i+1])))
    end
    # One-site X and Z terms
    for i in 1:N
        push!(gates, exp(-1.0im * tau / 2 * hx * op("X", sites_sys[i])))
        push!(gates, exp(-1.0im * tau / 2 * hz * op("Z", sites_sys[i])))
    end

    append!(gates, reverse(gates))
    return gates
end
