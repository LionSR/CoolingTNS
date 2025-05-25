"""
    trotter_dispatch.jl

Trotter circuit construction using multiple dispatch on HamiltonianModel and backend.
"""

using ITensors
include("parameter_types.jl")

# ============================================================================
# Utility Functions
# ============================================================================

"""
    parse_coupling(coupling::String) -> (String, String)

Parse coupling string (e.g., "XX", "YZ") into individual operator strings.
"""
function parse_coupling(coupling::String)
    if length(coupling) != 2
        error("Coupling must be a two-character string like 'XX', 'YZ', etc.")
    end
    return string(coupling[1]), string(coupling[2])
end

# ============================================================================
# Trotter Circuit Construction
# ============================================================================

"""
    build_trotter_circuit(ham_params::HamiltonianParameters, backend::CoolingBackend, sites_sys, sites_bath, coupling_params, sim_params)

Generic interface for building Trotter circuits using double dispatch.
"""
function build_trotter_circuit(ham_params::HamiltonianParameters, backend::CoolingBackend, sites_sys, sites_bath, coupling_params, sim_params)
    error("build_trotter_circuit not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

function build_trotter_circuit(ham_params::HamiltonianParameters{IsingModel}, 
                              backend::TNBackend, sites_sys, sites_bath, coupling_params::CouplingParameters, sim_params::TensorNetworkParameters)
    N = length(sites_sys)
    J, h = ham_params.params.J, ham_params.params.h
    g, Δ, coupling, tau = coupling_params.g, coupling_params.delta, coupling_params.coupling, sim_params.tau
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
        
        # System-bath coupling
        hsb = g * op(op1, s1) * op(op2, b1) - Δ / 2 * op("I", s1) * op("Z", b1)
        
        push!(gates, exp(-1.0im * tau / 2 * hs), exp(-1.0im * tau / 2 * hsb))
    end
    append!(gates, reverse(gates))
    return gates
end

function build_trotter_circuit(ham_params::HamiltonianParameters{NiIsingModel}, 
                              backend::TNBackend, sites_sys, sites_bath, coupling_params::CouplingParameters, sim_params::TensorNetworkParameters)
    N = length(sites_sys)
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    g, Δ, coupling, tau = coupling_params.g, coupling_params.delta, coupling_params.coupling, sim_params.tau
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
        
        # System-bath coupling
        hsb = g * op(op1, s1) * op(op2, b1) - Δ / 2 * op("I", s1) * op("Z", b1)
        
        push!(gates, exp(-1.0im * tau / 2 * hs), exp(-1.0im * tau / 2 * hsb))
    end
    append!(gates, reverse(gates))
    return gates
end

"""
    build_trotter_circuit_bath_coupling(ham_params::HamiltonianParameters, backend::CoolingBackend, sites_sys, sites_bath, coupling_params, sim_params)

Build Trotter circuit for just the bath coupling terms using double dispatch.
"""
function build_trotter_circuit_bath_coupling(ham_params::HamiltonianParameters, backend::CoolingBackend, sites_sys, sites_bath, coupling_params, sim_params)
    error("build_trotter_circuit_bath_coupling not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

function build_trotter_circuit_bath_coupling(ham_params::HamiltonianParameters, backend::TNBackend, sites_sys, sites_bath, coupling_params::CouplingParameters, sim_params::TensorNetworkParameters)
    N = length(sites_sys)
    g, Δ, coupling, tau = coupling_params.g, coupling_params.delta, coupling_params.coupling, sim_params.tau
    op1, op2 = parse_coupling(coupling)

    gates = ITensor[]
    for ind in eachindex(sites_sys)
        s1, b1 = sites_sys[ind], sites_bath[ind]
        hb = -Δ / 2 * op("Z", b1)
        hsb = g * op(op1, s1) * op(op2, b1)
        push!(gates, exp(-1.0im * tau / 2 * hb), exp(-1.0im * tau / 2 * hsb))
    end
    append!(gates, reverse(gates))
    return gates
end