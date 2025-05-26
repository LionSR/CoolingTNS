#!/usr/bin/env julia

using LinearAlgebra
using SparseArrays

# Include necessary files
include("src/parameter_types.jl")
include("src/cooling_types.jl")
include("src/ed_backend.jl")
include("src/system_hamiltonian.jl")
include("src/system_bath_hamiltonian.jl")
include("src/ground_state.jl")

# Very simple test: N=1 (1 system qubit + 1 bath qubit)
N = 1
println("=== Simple Cooling Test (N=$N) ===")

# Create Ising model parameters
ham_params = HamiltonianParameters(IsingModel(), N, (J=1.0, h=2.0))

# Get system Hamiltonian
H_sys = construct_system_hamiltonian(ham_params, EDBackend(), N)
println("\nSystem Hamiltonian:")
display(Matrix(H_sys))

# Find ground state and gap
E0, ψ0, gap = ground_state_ed(H_sys)
println("\nGround state energy: $E0")
println("Gap: $gap")

# Set up resonant cooling
Δ = -gap  # Negative for cooling
println("\nResonant frequency Δ = $Δ")

# Create coupling parameters
coupling_params = BasicCouplingParameters("XX", 0.1, 1, 0.5, Δ)

# Build system+bath Hamiltonian
H_sb = construct_system_bath_hamiltonian(ham_params, EDBackend(), 2*N, coupling_params)
println("\nSystem+Bath Hamiltonian (2 qubits total):")
display(Matrix(H_sb))

# Check that it's real and symmetric
println("\nIs H_sb real? $(isreal(H_sb))")
println("Is H_sb symmetric? $(issymmetric(H_sb))")

# Check eigenvalues
vals_sb = eigvals(Matrix(H_sb))
println("\nEigenvalues of H_sb: $vals_sb")

# Initial state: system in excited state |1⟩, bath in ground state |0⟩
# In the alternating layout: |sys=1, bath=0⟩ = |10⟩
println("\n\n=== Evolution Test ===")
println("Initial state: |1⟩_sys ⊗ |0⟩_bath")

# Create initial density matrix
ρ_init = zeros(Float64, 4, 4)
ρ_init[3,3] = 1.0  # |10⟩⟨10| (binary: 10 = 2 in decimal, but 1-indexed so position 3)
println("\nInitial density matrix:")
display(ρ_init)

# Initial system energy
# Trace out bath: ρ_sys = Tr_bath(ρ_total)
ρ_sys_init = zeros(Float64, 2, 2)
ρ_sys_init[1,1] = ρ_init[1,1] + ρ_init[3,3]  # |0⟩⟨0|
ρ_sys_init[2,2] = ρ_init[2,2] + ρ_init[4,4]  # |1⟩⟨1|
println("\nInitial system density matrix:")
display(ρ_sys_init)

E_sys_init = tr(Matrix(H_sys) * ρ_sys_init)
println("\nInitial system energy: $E_sys_init")

# Evolve for time t
t = 0.5
ρ_ed = EDDensityMatrix(ρ_init, 2)
ρ_evolved = evolve_ed(H_sb, ρ_ed, t)

println("\n\nEvolved density matrix:")
display(ρ_evolved.data)

# Trace out bath
ρ_sys_final = zeros(Float64, 2, 2)
ρ_sys_final[1,1] = ρ_evolved.data[1,1] + ρ_evolved.data[3,3]
ρ_sys_final[2,2] = ρ_evolved.data[2,2] + ρ_evolved.data[4,4]
ρ_sys_final[1,2] = ρ_evolved.data[1,2] + ρ_evolved.data[3,4]
ρ_sys_final[2,1] = ρ_evolved.data[2,1] + ρ_evolved.data[4,3]

println("\nFinal system density matrix:")
display(ρ_sys_final)

E_sys_final = tr(Matrix(H_sys) * ρ_sys_final)
println("\nFinal system energy: $E_sys_final")
println("Energy change: $(E_sys_final - E_sys_init)")

if E_sys_final < E_sys_init
    println("\n✅ System is cooling! Energy decreased by $(E_sys_init - E_sys_final)")
else
    println("\n❌ System is heating! Energy increased by $(E_sys_final - E_sys_init)")
end

# Check purity
purity_init = tr(ρ_sys_init^2)
purity_final = tr(ρ_sys_final^2)
println("\nSystem purity: $purity_init → $purity_final")

# Check detailed evolution
println("\n\n=== Detailed Analysis ===")
println("Bath energy (Δ/2 * Z): $(Δ/2)")
println("Coupling strength: $(coupling_params.g)")

# What should happen: system in excited state should swap energy with cold bath
# Expected: energy flows from hot system to cold bath