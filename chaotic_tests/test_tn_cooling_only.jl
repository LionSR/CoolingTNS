#!/usr/bin/env julia

# Minimal test for TN cooling without full module import
# This directly tests the corrected cooling physics

using ITensors
using ITensorMPS
using Plots

println("============================================================")
println("Testing TN Cooling - Direct Implementation")
println("============================================================")

# Parameters
N = 4
J = 1.0
hx = -1.05
hz = 0.5
g = 0.05
te = 2.0
steps = 50
Dmax = 64
cutoff = 1e-6

# Create sites for system+bath
sites = siteinds("S=1/2", 2N)

# Construct system Hamiltonian (niIsing)
function construct_system_hamiltonian_tn(sites, N, J, hx, hz)
    sys_sites = 1:2:2N-1
    
    terms = OpSum()
    # ZZ interactions
    for i in 1:N-1
        terms += J, "Z", sys_sites[i], "Z", sys_sites[i+1]
    end
    # X and Z fields
    for i in 1:N
        terms += hx, "X", sys_sites[i]
        terms += hz, "Z", sys_sites[i]
    end
    
    return MPO(terms, sites)
end

# Construct system+bath Hamiltonian
function construct_system_bath_hamiltonian_tn(sites, N, J, hx, hz, g, Δ)
    sys_sites = 1:2:2N-1
    bath_sites = 2:2:2N
    
    terms = OpSum()
    
    # System Hamiltonian
    for i in 1:N-1
        terms += J, "Z", sys_sites[i], "Z", sys_sites[i+1]
    end
    for i in 1:N
        terms += hx, "X", sys_sites[i]
        terms += hz, "Z", sys_sites[i]
    end
    
    # Bath Hamiltonians (Δ/2 * Z with Δ < 0 for cooling)
    for i in 1:N
        terms += Δ/2, "Z", bath_sites[i]
    end
    
    # System-Bath coupling (XX coupling)
    for i in 1:N
        terms += g, "X", sys_sites[i], "X", bath_sites[i]
    end
    
    return MPO(terms, sites)
end

# Include the existing utility functions
include("src/utils_mps.jl")

# Get ground state using DMRG
println("Computing ground state...")
sites_sys_only = siteinds("S=1/2", N)

# Construct system-only Hamiltonian
function construct_system_only_hamiltonian(sites, N, J, hx, hz)
    terms = OpSum()
    # ZZ interactions
    for i in 1:N-1
        terms += J, "Z", i, "Z", i+1
    end
    # X and Z fields
    for i in 1:N
        terms += hx, "X", i
        terms += hz, "Z", i
    end
    return MPO(terms, sites)
end

H_sys_only = construct_system_only_hamiltonian(sites_sys_only, N, J, hx, hz)
ψ0 = randomMPS(sites_sys_only, 10)
sweeps = Sweeps(5)
setmaxdim!(sweeps, 10, 20, 40, 80, 100)
setcutoff!(sweeps, 1E-10)
E0, ϕ0 = dmrg(H_sys_only, ψ0, sweeps; outputlevel=0)
println("Ground state energy: E0/N = $(E0/N)")

# Set delta for resonant cooling
gap = abs(E0 - (-N * hz))  # Approximate gap
Δ = -gap  # Negative for cooling

# Construct full Hamiltonian
H_sb = construct_system_bath_hamiltonian_tn(sites, N, J, hx, hz, g, Δ)

# Initialize system in product state
ψ_sys = productMPS(sites_sys_only, [1 for _ in 1:N])  # All spins up

# Storage for results
E_list = Float64[]
overlap_list = Float64[]

# Initial measurements
E_init = real(inner(ψ_sys', H_sys_only, ψ_sys))
overlap_init = abs2(inner(ψ_sys, ϕ0))
push!(E_list, E_init)
push!(overlap_list, overlap_init)

println("Initial: E/N = $(E_init/N), overlap = $overlap_init")

# Cooling loop
for step in 1:steps
    # Append fresh bath in ground state using existing function
    ψ_sb = appendzeros_MPS(ψ_sys, sites)
    
    # Time evolution using TDVP
    ψ_evolved = tdvp(H_sb, -im * te, ψ_sb; 
                     time_step=-im * 0.1, 
                     reverse_step=false, 
                     normalize=true, 
                     maxdim=Dmax, 
                     cutoff=cutoff, 
                     outputlevel=0)
    
    # Sample bath and get system state using existing function
    v_b, ψ_sys_new = sample_bath(ψ_evolved)
    global ψ_sys = ψ_sys_new
    
    # Measurements
    E = real(inner(ψ_sys', H_sys_only, ψ_sys))
    overlap = abs2(inner(ψ_sys, ϕ0))
    push!(E_list, E)
    push!(overlap_list, overlap)
    
    if step % 10 == 0
        println("Step $step: E/N = $(E/N), overlap = $overlap")
    end
end

# Final results
println("\nFinal results:")
println("E/N = $(E_list[end]/N) (should approach $(E0/N))")
println("Overlap = $(overlap_list[end]) (should approach 1)")

# Plotting
p1 = plot(0:steps, E_list ./ N, label="Energy/N", linewidth=2)
hline!(p1, [E0/N], label="Ground state", linestyle=:dash, color=:black)
xlabel!(p1, "Step")
ylabel!(p1, "Energy per spin")
title!(p1, "TN Cooling: Energy Evolution")

p2 = plot(0:steps, overlap_list, label="GS Overlap", linewidth=2, color=:red)
xlabel!(p2, "Step")
ylabel!(p2, "Overlap")
title!(p2, "Ground State Overlap")
ylims!(p2, (0, 1))

p = plot(p1, p2, layout=(2,1), size=(600,800))
savefig(p, "tn_cooling_test.pdf")
println("\nPlot saved to: tn_cooling_test.pdf")

# Check if cooling worked
if E_list[end] < E_list[1]
    println("\n✓ SUCCESS: System cooled! Energy decreased from $(E_list[1]/N) to $(E_list[end]/N)")
else
    println("\n✗ FAILURE: System did not cool. Energy went from $(E_list[1]/N) to $(E_list[end]/N)")
end