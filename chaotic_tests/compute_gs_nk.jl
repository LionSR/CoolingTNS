using CoolingTNS
using HDF5
using LinearAlgebra

"""
Compute the momentum distribution n_k for the numerical ground state
using the same Jordan-Wigner operators as in the simulation.
"""
function compute_ground_state_nk(N::Int, J::Float64, h::Float64, bc::Symbol)
    # Create Hamiltonian parameters
    ham_params = CoolingTNS.HamiltonianParameters(
        CoolingTNS.IsingModel(),
        N,
        (J=J, h=h),
        bc
    )
    
    # Create system Hamiltonian
    println("Constructing system Hamiltonian...")
    H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N)
    
    # Find ground state
    println("Finding ground state...")
    E0, ψ0 = CoolingTNS.find_ground_state(H_sys, CoolingTNS.EDBackend())
    
    println("Ground state energy: E0/N = ", E0/N)
    
    # Compute momentum distribution for ground state
    println("Computing momentum distribution for ground state...")
    k_values, n_k_gs = CoolingTNS.measure_momentum_distribution_ed(ψ0, ham_params)
    
    return k_values, n_k_gs, E0
end

# Test with the same parameters as the simulation
N = 4
J = 1.0
h = 2.0
bc = :periodic

k_values, n_k_gs, E0 = compute_ground_state_nk(N, J, h, bc)

println("\nGround state momentum distribution:")
println("k/π values: ", round.(k_values/π, digits=3))
println("n_k values: ", round.(n_k_gs, digits=3))

# Save to file for plotting
h5open("gs_nk_data.h5", "w") do file
    write(file, "k_values", k_values)
    write(file, "n_k_gs", n_k_gs)
    write(file, "E0", E0)
    write(file, "N", N)
    write(file, "J", J)
    write(file, "h", h)
end

println("\nGround state data saved to gs_nk_data.h5")