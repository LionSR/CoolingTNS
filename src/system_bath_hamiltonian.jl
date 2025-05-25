"""
    system_bath_hamiltonian.jl

System+bath Hamiltonian construction using multiple dispatch on HamiltonianModel and backend.
"""

using ITensors
using Yao
# parameter_types.jl already included by parent

# system_hamiltonian.jl included by parent



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
                                         backend::TNBackend, sites::Vector{<:Index}, coupling_params::CouplingParameters)
    J, h = ham_params.params.J, ham_params.params.h
    g, Δ, coupling = coupling_params.g, coupling_params.delta, coupling_params.coupling
    
    N = ham_params.N
    # Use site indices (integers) instead of Index objects
    sys_sites = 1:2:2N-1
    bath_sites = 2:2:2N
    
    op1, op2 = parse_coupling(coupling)
    
    terms = OpSum()
    
    # System Hamiltonian
    for i in 1:N-1
        # sys_sites[i] is the i-th system site, sys_sites[i+1] is the next system site
        terms += J, "Z", 2i-1, "Z", 2(i+1)-1  # Direct calculation: site 1,3,5,7...
    end
    for i in 1:N
        terms += h, "X", 2i-1  # System sites: 1,3,5,7...
    end
    
    # Bath Hamiltonians  
    for i in 1:N
        terms += -Δ/2, "Z", 2i  # Bath sites: 2,4,6,8...
    end
    
    # System-Bath coupling
    for i in 1:N
        terms += g, op1, 2i-1, op2, 2i  # System site coupled to adjacent bath site
    end
    
    return MPO(terms, sites)
end

function construct_system_bath_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, 
                                         backend::TNBackend, sites::Vector{<:Index}, coupling_params::CouplingParameters)
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    g, Δ, coupling = coupling_params.g, coupling_params.delta, coupling_params.coupling
    
    N = ham_params.N
    # Use site indices (integers) instead of Index objects
    sys_sites = 1:2:2N-1
    bath_sites = 2:2:2N
    
    op1, op2 = parse_coupling(coupling)
    
    terms = OpSum()
    
    # System Hamiltonian
    for i in 1:N-1
        # sys_sites[i] is the i-th system site, sys_sites[i+1] is the next system site
        terms += J, "Z", 2i-1, "Z", 2(i+1)-1  # Direct calculation: site 1,3,5,7...
    end
    for i in 1:N
        terms += hx, "X", 2i-1  # System sites: 1,3,5,7...
        terms += hz, "Z", 2i-1
    end
    
    # Bath Hamiltonians
    for i in 1:N
        terms += Δ/2, "Z", 2i  # Bath sites: 2,4,6,8...
    end
    
    # System-Bath coupling
    for i in 1:N
        terms += g, op1, 2i-1, op2, 2i  # System site coupled to adjacent bath site
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
    
    N_sys = ham_params.N  # Number of system spins
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
    H_bath_coupling = sum(map(i -> Δ/2 * put(nbits, bath_sites[i]=>Z) + 
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
    
    N_sys = ham_params.N  # Number of system spins
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
    H_bath = sum(map(b -> Δ/2 * put(nbits, b=>Z), bath_sites))
    
    # System-bath coupling
    H_coupling = sum(map(i -> g * put(nbits, sys_sites[i]=>op1) * put(nbits, bath_sites[i]=>op2), 1:N_sys))
    
    return H_sys + H_bath + H_coupling
end



"""
    construct_zero_coupling_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites)

Create Hamiltonian with zero coupling for Trotter evolution using double dispatch.
"""
function construct_zero_coupling_hamiltonian(ham_params::HamiltonianParameters, backend::CoolingBackend, sites)
    error("construct_zero_coupling_hamiltonian not implemented for model $(typeof(ham_params.model)) and backend $(typeof(backend))")
end

function construct_zero_coupling_hamiltonian(ham_params::HamiltonianParameters{IsingModel}, backend::TNBackend, sites::Vector{<:Index})
    J, h = ham_params.params.J, ham_params.params.h
    N = ham_params.N
    zero_coupling_params = BasicCouplingParameters("XX", 0.0, 1, 0.0, 0.0)  # coupling, g, steps, te, delta
    return construct_system_bath_hamiltonian(ham_params, backend, sites, zero_coupling_params)
end

function construct_zero_coupling_hamiltonian(ham_params::HamiltonianParameters{NiIsingModel}, backend::TNBackend, sites::Vector{<:Index})
    J, hx, hz = ham_params.params.J, ham_params.params.hx, ham_params.params.hz
    N = ham_params.N
    zero_coupling_params = BasicCouplingParameters("XX", 0.0, 1, 0.0, 0.0)  # coupling, g, steps, te, delta
    return construct_system_bath_hamiltonian(ham_params, backend, sites, zero_coupling_params)
end