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

# Test with N=2 system using product state initial condition
N = 2
println("=== Product State Cooling Test (N=$N) ===")

# Create Ising model parameters
ham_params = HamiltonianParameters(IsingModel(), N, (J=1.0, h=2.0))

# Get system Hamiltonian and ground state
H_sys = construct_system_hamiltonian(ham_params, EDBackend(), N)
E0, ψ0, gap = ground_state_ed(H_sys)

println("\nSystem ground state energy: $E0")
println("Gap: $gap")

# Set up resonant cooling
Δ = -gap
coupling_params = BasicCouplingParameters("XX", 0.1, 1, 0.5, Δ)

# Build system+bath Hamiltonian
H_sb = construct_system_bath_hamiltonian(ham_params, EDBackend(), 2*N, coupling_params)

println("\nResonant frequency Δ = $Δ")
println("Bath energies: $(Δ/2) per bath qubit")

# Create initial product state |00⟩_sys (all spins up)
# This is what the cooling simulation uses by default
println("\n\n=== Initial State Setup ===")
println("Using product state: |00⟩_sys (all spins up)")

# System product state |00⟩
ρ_sys_init = zeros(Float64, 2^N, 2^N)
ρ_sys_init[1,1] = 1.0  # |00⟩⟨00|

# Bath initial state - should be thermal at high T (maximally mixed)
# But let's start with ground state |00⟩_bath to match the simulation
ρ_bath_init = zeros(Float64, 2^N, 2^N)
ρ_bath_init[1,1] = 1.0  # |00⟩⟨00|

# Combined initial state with alternating layout
# For N=2: |s1 b1 s2 b2⟩ = |0000⟩
ρ_init_full = zeros(Float64, 2^(2*N), 2^(2*N))
ρ_init_full[1,1] = 1.0  # |0000⟩⟨0000|

println("\nInitial system energy:")
E_sys_init = tr(H_sys * ρ_sys_init)
println("E_sys = $E_sys_init")
println("E_sys/N = $(E_sys_init/N)")

# This is far from the ground state!
overlap_init = real(ψ0.data' * ρ_sys_init * ψ0.data)
println("Initial overlap with ground state: $overlap_init")

# Now evolve
println("\n\n=== Time Evolution ===")
t = 0.5
ρ_ed = EDDensityMatrix(ρ_init_full, 2*N)
ρ_evolved = evolve_ed(H_sb, ρ_ed, t)

# Trace out bath using the ED backend function
ρ_sys_final = trace_out_bath_ed(ρ_evolved, N)

println("\nFinal system density matrix:")
display(ρ_sys_final.data)

E_sys_final = expect_ed(H_sys, ρ_sys_final)
println("\nFinal system energy: $E_sys_final")
println("Final system energy/N: $(E_sys_final/N)")
println("Energy change: $(E_sys_final - E_sys_init)")

overlap_final = real(ψ0.data' * ρ_sys_final.data * ψ0.data)
println("\nFinal overlap with ground state: $overlap_final")

if E_sys_final < E_sys_init
    println("\n✅ System is cooling! Energy decreased.")
else
    println("\n❌ System is heating! Energy increased.")
end

# Let's also check what happens with different evolution times
println("\n\n=== Energy vs Time ===")
times = [0.1, 0.2, 0.5, 1.0, 2.0]
for t in times
    ρ_t = evolve_ed(H_sb, ρ_ed, t)
    ρ_sys_t = trace_out_bath_ed(ρ_t, N)
    E_t = expect_ed(H_sys, ρ_sys_t)
    println("t = $t: E/N = $(E_t/N)")
end

# Check if the issue is with the initial bath state
println("\n\n=== Alternative: Bath in excited state ===")
# Try with bath in maximally excited state |11⟩
ρ_init_alt = zeros(Float64, 2^(2*N), 2^(2*N))
# |s1=0, b1=1, s2=0, b2=1⟩ = |0101⟩ = 5 in binary
ρ_init_alt[6,6] = 1.0  # |0101⟩⟨0101| (5+1=6 due to 1-indexing)

ρ_ed_alt = EDDensityMatrix(ρ_init_alt, 2*N)
ρ_evolved_alt = evolve_ed(H_sb, ρ_ed_alt, 0.5)
ρ_sys_final_alt = trace_out_bath_ed(ρ_evolved_alt, N)
E_sys_final_alt = expect_ed(H_sys, ρ_sys_final_alt)

println("With bath initially excited:")
println("Final E/N = $(E_sys_final_alt/N)")
println("Energy change: $(E_sys_final_alt - E_sys_init)")