using CoolingTNS

# Create Ising system same as test
N = 4
J = 1.0  
h = 0.5

# Create Hamiltonian parameters using proper constructor
ham_params = CoolingTNS.HamiltonianParameters(CoolingTNS.IsingModel(), N, (J=J, h=h))

# Build system Hamiltonian
backend = CoolingTNS.EDBackend()
H_sys = CoolingTNS.construct_system_hamiltonian(ham_params, backend, N)

# Compute gap
gap = CoolingTNS.compute_gap_ed(H_sys)

println("System parameters: N=$N, J=$J, h=$h")
println("Energy gap: $gap")

# Also check ground state energy
E0, ψ0, _ = CoolingTNS.ground_state_ed(H_sys)
println("Ground state energy: $E0")
println("Ground state energy per site: $(E0/N)")

# Check what the system-bath Hamiltonian looks like
coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.1, 10, 0.5, nothing)
println("\nBuilding system-bath Hamiltonian...")
H_sb = CoolingTNS.construct_system_bath_hamiltonian(ham_params, backend, 8, coupling_params)

# Check gap used in system-bath Hamiltonian
println("Delta used in coupling_params: $(coupling_params.delta)")

# Check energies of system-bath Hamiltonian
vals_sb, _, _ = CoolingTNS.eigsolve(H_sb, 4, :SR; krylovdim=min(30, size(H_sb, 1)))
E0_sb = real(vals_sb[1])
E1_sb = real(vals_sb[2])
println("System-bath ground state energy: $E0_sb")
println("System-bath first excited energy: $E1_sb")
println("System-bath gap: $(E1_sb - E0_sb)")