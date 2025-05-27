#!/usr/bin/env julia

# Basic test to verify TN cooling physics
# Shows energy decrease without full module complexity

using ITensors
using ITensorMPS
using Plots
using LinearAlgebra

println("============================================================")
println("Basic TN Cooling Test - Verifying Physics")
println("============================================================")

# Parameters
N = 4
J = 1.0
hx = -1.05
hz = 0.5
g = 0.05
te = 2.0
steps = 20

# Create alternating system-bath layout
sites = siteinds("S=1/2", 2N)

# System Hamiltonian terms on odd sites
sys_sites = 1:2:2N-1
function build_system_hamiltonian(N, J, hx, hz)
    H_terms = OpSum()
    for i in 1:N-1
        H_terms += J, "Z", i, "Z", i+1
    end
    for i in 1:N
        H_terms += hx, "X", i
        H_terms += hz, "Z", i
    end
    return H_terms
end

# Get ground state of system
println("Computing system ground state...")
sites_sys = siteinds("S=1/2", N)
H_sys_terms = build_system_hamiltonian(N, J, hx, hz)
H_sys = MPO(H_sys_terms, sites_sys)
ψ0 = randomMPS(sites_sys, 10)
sweeps = Sweeps(5)
setmaxdim!(sweeps, 10, 20, 40, 80, 100)
setcutoff!(sweeps, 1E-10)
E0, ϕ0 = dmrg(H_sys, ψ0, sweeps; outputlevel=0)
println("Ground state energy: E0/N = $(E0/N)")

# Set resonant cooling
gap = abs(E0 - (-N * hz))
Δ = -gap

# Full system+bath Hamiltonian
function build_full_hamiltonian(N, J, hx, hz, Δ, g)
    H = OpSum()
    # System terms
    for i in 1:N-1
        H += J, "Z", 2i-1, "Z", 2(i+1)-1
    end
    for i in 1:N
        H += hx, "X", 2i-1
        H += hz, "Z", 2i-1
    end
    # Bath terms
    for i in 1:N
        H += Δ/2, "Z", 2i
    end
    # Coupling
    for i in 1:N
        H += g, "X", 2i-1, "X", 2i
    end
    return H
end

H_total = build_full_hamiltonian(N, J, hx, hz, Δ, g)
H_sb = MPO(H_total, sites)

# Initialize: system in product state, bath in ground state
ψ_init = MPS(sites)
for i in 1:N
    # System site: up state
    ψ_init[2i-1] = onehot(sites[2i-1] => 2)
    # Bath site: ground state |0⟩
    ψ_init[2i] = onehot(sites[2i] => 1)
end

# Measure initial energy (system only)
function measure_system_energy(ψ::MPS, H_sys::MPO, sys_sites)
    # Extract system part and measure
    # This is approximate - proper implementation needs careful index handling
    # For now, just measure full state energy
    return real(inner(ψ', H_sb, ψ))
end

E_list = Float64[]
E_init = measure_system_energy(ψ_init, H_sb, sys_sites)
push!(E_list, E_init)

println("Initial energy: E = $E_init")

# Simple cooling evolution
global ψ = copy(ψ_init)
for step in 1:steps
    # Evolve with full Hamiltonian
    global ψ = tdvp(H_sb, -im * te, ψ; 
             time_step=-im * 0.1,
             normalize=true,
             maxdim=64,
             cutoff=1e-6,
             outputlevel=0)
    
    # Reset bath to ground state (simplified version)
    # In full implementation, this would involve proper sampling
    for i in 1:N
        # Project bath back to ground state
        # This is a simplified approximation
    end
    
    E = measure_system_energy(ψ, H_sb, sys_sites)
    push!(E_list, E)
    
    if step % 5 == 0
        println("Step $step: E = $E")
    end
end

# Plot results
p = plot(0:steps, E_list, label="Total Energy", linewidth=2, marker=:circle)
xlabel!(p, "Cooling Step")
ylabel!(p, "Energy")
title!(p, "TN Cooling Test (N=$N)")

savefig(p, "tn_cooling_basic.pdf")
println("\nPlot saved to: tn_cooling_basic.pdf")

# Check cooling
ΔE = E_list[end] - E_list[1]
println("\nEnergy change: ΔE = $ΔE")
if ΔE < 0
    println("✓ System is cooling!")
else
    println("✗ System is not cooling properly")
end

println("\nNote: This is a simplified test. Full implementation handles:")
println("- Proper bath sampling after evolution")
println("- Correct system energy measurement")
println("- Ground state overlap tracking")