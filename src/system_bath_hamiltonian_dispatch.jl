"""
    system_bath_hamiltonian_dispatch.jl

System+bath Hamiltonian construction using multiple dispatch on HamiltonianModel and backend.
"""

using ITensors
using Yao
include("parameter_types.jl")
include("coupling_utils.jl")
include("system_hamiltonian_dispatch.jl")



# ============================================================================
# System-Bath Hamiltonian Construction Interface  
# ============================================================================

"""
    construct_system_bath_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites, coupling_params)

Generic interface for constructing full system+bath Hamiltonians with double dispatch.
"""
function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites, coupling_params)
    error("construct_system_bath_hamiltonian not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

# ============================================================================
# Tensor Network System-Bath Hamiltonians
# ============================================================================

function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters{IsingModel}, 
                                         backend::TNBackend, sites, coupling_params::CouplingParameters)
    J, h = ham_params.params.J, ham_params.params.h
    g, Δ, coupling = coupling_params.g, coupling_params.delta, coupling_params.coupling
    
    N = length(sites) ÷ 2
    # Use site indices (integers) instead of Index objects
    sys_sites = 1:2:2N-1
    bath_sites = 2:2:2N
    
    op1, op2 = parse_coupling(coupling)
    
    terms = OpSum()
    
    # System Hamiltonian
    for i in 1:N-1
        terms += J, "Z", sys_sites[i], "Z", sys_sites[i+2]  # i+2 because sys_sites are odd
    end
    for i in 1:N
        terms += h, "X", sys_sites[i]
    end
    
    # Bath Hamiltonians  
    for i in 1:N
        terms += -Δ/2, "Z", bath_sites[i]
    end
    
    # System-Bath coupling
    for i in 1:N
        terms += g, op1, sys_sites[i], op2, bath_sites[i]
    end
    
    return MPO(terms, sites)
end

function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, 
                                         backend::TNBackend, sites, coupling_params::CouplingParameters)
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    g, Δ, coupling = coupling_params.g, coupling_params.delta, coupling_params.coupling
    
    N = length(sites) ÷ 2
    sites_sys = sites[1:2:2N-1]
    sites_bath = sites[2:2:2N]
    
    op1, op2 = parse_coupling(coupling)
    
    terms = OpSum()
    
    # System Hamiltonian
    for i in 1:N-1
        terms += J, "Z", sites_sys[i], "Z", sites_sys[i+1]
    end
    for i in 1:N
        terms += hx, "X", sites_sys[i]
        terms += hz, "Z", sites_sys[i]
    end
    
    # Bath Hamiltonians
    for i in 1:N
        terms += -Δ/2, "Z", sites_bath[i]
    end
    
    # System-Bath coupling
    for i in 1:N
        terms += g, op1, sites_sys[i], op2, sites_bath[i]
    end
    
    return MPO(terms, sites)
end

# ============================================================================
# ED System-Bath Hamiltonians  
# ============================================================================

function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters{IsingModel}, 
                                         backend::EDBackend, nbits::Int, coupling_params::CouplingParameters)
    J, h = ham_params.params.J, ham_params.params.h
    g = coupling_params.g
    Δ = coupling_params.delta
    coupling = coupling_params.coupling
    
    # Parse coupling operators
    op1_str, op2_str = parse_coupling(coupling)
    op_map = Dict("X" => X, "Y" => Y, "Z" => Z)
    op1, op2 = op_map[op1_str], op_map[op2_str]
    
    N_sys = nbits ÷ 2  # Number of system spins
    # System sites (odd) and bath sites (even)
    sys_sites = 1:2:nbits-1
    bath_sites = 2:2:nbits
    
    # System Hamiltonian (reuse system-only construction)
    H_sys = construct_system_hamiltonian(ham_params, backend, N_sys)
    
    # Expand to full system+bath space
    H_sys_expanded = sum([
        map(i -> J * put(nbits, sys_sites[i]=>Z) * put(nbits, sys_sites[i+1]=>Z), 1:N_sys-1)...,
        map(s -> h * put(nbits, s=>X), sys_sites)...
    ])
    
    # Bath and coupling
    H_bath_coupling = sum(map(i -> -Δ/2 * put(nbits, bath_sites[i]=>Z) + 
                                   g * put(nbits, sys_sites[i]=>op1) * put(nbits, bath_sites[i]=>op2), 1:N_sys))
    
    return H_sys_expanded + H_bath_coupling
end

function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, 
                                         backend::EDBackend, nbits::Int, coupling_params::CouplingParameters)
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    g = coupling_params.g
    Δ = coupling_params.delta
    coupling = coupling_params.coupling
    
    # Parse coupling operators
    op1_str, op2_str = parse_coupling(coupling)
    op_map = Dict("X" => X, "Y" => Y, "Z" => Z)
    op1, op2 = op_map[op1_str], op_map[op2_str]
    
    N_sys = nbits ÷ 2  # Number of system spins
    # System sites (odd) and bath sites (even)
    sys_sites = 1:2:nbits-1
    bath_sites = 2:2:nbits
    
    # System Hamiltonian using functional style
    H_sys = sum([
        # ZZ interactions between system spins
        map(i -> J * put(nbits, sys_sites[i]=>Z) * put(nbits, sys_sites[i+1]=>Z), 1:N_sys-1)...,
        # X field on system spins
        map(s -> hx * put(nbits, s=>X), sys_sites)...,
        # Z field on system spins
        map(s -> hz * put(nbits, s=>Z), sys_sites)...
    ])
    
    # Bath Hamiltonian
    H_bath = sum(map(b -> -Δ/2 * put(nbits, b=>Z), bath_sites))
    
    # System-bath coupling
    H_coupling = sum(map(i -> g * put(nbits, sys_sites[i]=>op1) * put(nbits, bath_sites[i]=>op2), 1:N_sys))
    
    return H_sys + H_bath + H_coupling
end