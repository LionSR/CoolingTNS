#!/usr/bin/env julia

# Test ED cooling physics
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using LinearAlgebra
using SparseArrays
using KrylovKit

# Include only necessary files
include("src/parameter_types.jl")
include("src/cooling_types.jl")
include("src/ed_backend.jl")
include("src/system_hamiltonian.jl")
include("src/system_bath_hamiltonian.jl")
include("src/ground_state.jl")

# Import the coupling parameters type
BasicCouplingParameters = Main.BasicCouplingParameters

# Test parameters
N = 3
J = 1.0
h = -2.0
g = 0.2

# Create Hamiltonian
ham_params = HamiltonianParameters(IsingModel(), N, (J=J, h=h), :open)
backend = EDBackend()

# System Hamiltonian
H_sys = construct_system_hamiltonian(ham_params, backend, N)
println("System Hamiltonian:")
display(Matrix(H_sys))

# Find ground state
e₀, ϕ₀, gap = find_ground_state(H_sys, backend)
println("\nGround state energy: ", e₀)
println("Ground state energy per site: ", e₀/N)
println("Energy gap: ", gap)

# Create system+bath Hamiltonian
coupling_params = (coupling="XX", g=g, delta=nothing)
# Create proper coupling parameters
coupling_params_typed = BasicCouplingParameters(coupling_params.coupling, coupling_params.g, 1, 1.0, coupling_params.delta)
H_sb = construct_system_bath_hamiltonian(ham_params, backend, 2*N, coupling_params_typed)
println("\nSystem+bath Hamiltonian size: ", size(H_sb))

# Initial state: |000⟩ ⊗ |000⟩ (all up)
ψ0_sys = zero_state_ed(N)
ψ0_bath = zero_state_ed(N)
ψ0 = kron_states_ed(ψ0_sys, ψ0_bath)
println("\nInitial state energy: ", expect_ed(H_sb, ψ0))

# Evolve for time t
t = 1.0
ψ_evolved = evolve_ed(H_sb, ψ0, t)
println("Energy after evolution: ", expect_ed(H_sb, ψ_evolved))

# Measure and collapse bath
bath_qubits = [2*i for i in 1:N]  # Even positions
ψ_sys_final, bath_outcomes = measure_ed!(ψ_evolved, bath_qubits)
println("\nBath measurement outcomes: ", bath_outcomes)

# System energy after cooling
E_sys_final = expect_ed(H_sys, ψ_sys_final)
println("System energy after cooling: ", E_sys_final)
println("System energy per site after cooling: ", E_sys_final/N)

# Check if cooling occurred
E_sys_initial = expect_ed(H_sys, ψ0_sys)
println("\nInitial system energy: ", E_sys_initial)
println("Final system energy: ", E_sys_final)
println("Ground state energy: ", e₀)
println("Did cooling occur? ", E_sys_final < E_sys_initial)
println("Energy moved toward ground state? ", abs(E_sys_final - e₀) < abs(E_sys_initial - e₀))