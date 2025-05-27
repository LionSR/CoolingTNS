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
include("src/cooling_evolution.jl")
include("src/initial_state.jl")
include("src/bath_measurements.jl")

# Test with N=2 using the actual alternating initial state
N = 2
println("=== Alternating Initial State Test (N=$N) ===")

# Create Ising model parameters
ham_params = HamiltonianParameters(IsingModel(), N, (J=1.0, h=2.0))

# Get system Hamiltonian and ground state
H_sys = construct_system_hamiltonian(ham_params, EDBackend(), N)
E0, ψ0, gap = ground_state_ed(H_sys)

println("\nSystem ground state energy: $E0")
println("Gap: $gap")

# Create alternating state as in the code
function get_alternating_config(N)
    config = 0
    for i in 0:(N-1)
        if isodd(i+1)  # Julia is 1-indexed
            config |= (1 << i)
        end
    end
    return config
end

alt_config = get_alternating_config(N)
println("\nAlternating config: $alt_config (binary: $(string(alt_config, base=2, pad=N)))")

# For N=2: i=0 (position 1 in Julia) is odd -> bit 0 set
#          i=1 (position 2 in Julia) is even -> bit 1 not set
# So config = 1 (binary: 01) = |01⟩ = |↑↓⟩

# Create the state
ψ_alt = product_state_ed(N, alt_config)
println("Alternating state: |01⟩ = |↑↓⟩")

# Convert to density matrix
ρ_sys_init = state_to_density_ed(ψ_alt)
println("\nInitial system density matrix:")
display(ρ_sys_init.data)

# Calculate initial energy
E_sys_init = expect_ed(H_sys, ρ_sys_init)
println("\nInitial system energy: $E_sys_init")
println("Initial energy/N: $(E_sys_init/N)")

# Set up resonant cooling
Δ = -gap
coupling_params = BasicCouplingParameters("XX", 0.1, 1, 0.5, Δ)

# Build system+bath Hamiltonian
H_sb = construct_system_bath_hamiltonian(ham_params, EDBackend(), 2*N, coupling_params)

# Create full initial state with alternating system and ground state bath
# System: |01⟩, Bath: |00⟩
# Full state: |s1=0, b1=0, s2=1, b2=0⟩ = |0010⟩ = 2 in binary
ρ_init_full = zeros(Float64, 2^(2*N), 2^(2*N))
ρ_init_full[3,3] = 1.0  # |0010⟩⟨0010| (2+1=3 due to 1-indexing)

println("\nFull initial state: |0010⟩ (s1=0, b1=0, s2=1, b2=0)")

# Evolve
t = 0.5
ρ_ed = EDDensityMatrix(ρ_init_full, 2*N)
ρ_evolved = evolve_ed(H_sb, ρ_ed, t)

# Trace out bath
ρ_sys_final = trace_out_bath_ed(ρ_evolved, N)
E_sys_final = expect_ed(H_sys, ρ_sys_final)

println("\n\nAfter evolution (t=$t):")
println("Final system energy: $E_sys_final")
println("Final energy/N: $(E_sys_final/N)")
println("Energy change: $(E_sys_final - E_sys_init)")

if E_sys_final < E_sys_init
    println("\n✅ System is cooling!")
else
    println("\n❌ System is heating!")
end

# Compare with all-up state |00⟩
println("\n\n=== Comparison with |00⟩ initial state ===")
ρ_00 = zeros(Float64, 2^N, 2^N)
ρ_00[1,1] = 1.0
E_00 = real(tr(H_sys * ρ_00))
println("|00⟩ energy: $E_00, energy/N: $(E_00/N)")

# And with all-down state |11⟩
ρ_11 = zeros(Float64, 2^N, 2^N)
ρ_11[4,4] = 1.0
E_11 = real(tr(H_sys * ρ_11))
println("|11⟩ energy: $E_11, energy/N: $(E_11/N)")

println("\nSo alternating |01⟩ has energy/N = $(E_sys_init/N)")
println("Ground state has energy/N = $(E0/N)")

# Check eigenstate overlaps
println("\n\nEigenstate decomposition of |01⟩:")
vals, vecs = eigen(Matrix(H_sys))
for i in 1:4
    overlap = abs2(vecs[:,i]' * ψ_alt.data)
    if overlap > 1e-10
        println("E$i = $(vals[i]): overlap = $overlap")
    end
end