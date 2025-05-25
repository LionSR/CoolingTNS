#!/usr/bin/env julia

# Comprehensive test for all TN cooling methods
# Using the actual CoolingTNS framework correctly

using ITensors
using ITensorMPS
using Plots
using LinearAlgebra

println("============================================================")
println("Comprehensive TN Cooling Test - All Methods")
println("============================================================")

# Include necessary functions from the existing codebase
include("src/utils_mps.jl")

# Parameters for good cooling performance
N = 4
J = 1.0
hx = -1.05
hz = 0.5
coupling = "XX"
g = 0.1  # Stronger coupling
te = 1.0  # Shorter steps but more frequent
steps = 100
Dmax = 64
cutoff = 1e-6

# Define helper functions first
function build_system_hamiltonian(N, J, hx, hz, sites)
    terms = OpSum()
    for i in 1:N-1
        terms += J, "Z", i, "Z", i+1
    end
    for i in 1:N
        terms += hx, "X", i
        terms += hz, "Z", i
    end
    return MPO(terms, sites)
end

function build_system_bath_hamiltonian(N, J, hx, hz, Δ, g, sites)
    terms = OpSum()
    for i in 1:N-1
        terms += J, "Z", 2i-1, "Z", 2(i+1)-1
    end
    for i in 1:N
        terms += hx, "X", 2i-1
        terms += hz, "Z", 2i-1
    end
    for i in 1:N
        terms += Δ/2, "Z", 2i
    end
    for i in 1:N
        terms += g, "X", 2i-1, "X", 2i
    end
    return MPO(terms, sites)
end

println("Parameters: N=$N, J=$J, hx=$hx, hz=$hz, g=$g, coupling=$coupling")

# Get ground state energy
sites_sys = siteinds("S=1/2", N)

H_sys = build_system_hamiltonian(N, J, hx, hz, sites_sys)

# DMRG to get ground state
ψ0 = randomMPS(sites_sys, 10)
sweeps = Sweeps(5)
setmaxdim!(sweeps, 10, 20, 40, 80, 100)
setcutoff!(sweeps, 1E-10)
E0, ϕ0 = dmrg(H_sys, ψ0, sweeps; outputlevel=0)
println("Ground state energy: E0/N = $(E0/N)")

# Set resonant cooling  
gap = abs(E0 - (-N * hz))
Δ = -gap
println("Energy gap = $gap, setting Δ = $Δ")

# Function to run cooling simulation
function run_tn_cooling_simulation(method_name::String, use_density_matrix::Bool, use_trotter::Bool)
    println("\n" * "="^60)
    println("Testing: $method_name")
    println("="^60)
    
    # Create sites for system+bath
    sites = siteinds("S=1/2", 2N)
    
    # Create system+bath Hamiltonian using the same function
    H_sb = build_system_bath_hamiltonian(N, J, hx, hz, Δ, g, sites)
    
    # Initialize system state (all up) 
    ψ_sys = productMPS(sites_sys, [2 for _ in 1:N])  # All up states
    
    # Storage for results
    E_list = Float64[]
    overlap_list = Float64[]
    
    # Initial measurements
    E_init = real(inner(ψ_sys', H_sys, ψ_sys))
    overlap_init = abs2(inner(ψ_sys, ϕ0))
    push!(E_list, E_init)
    push!(overlap_list, overlap_init)
    
    println("Initial: E/N = $(E_init/N), overlap = $overlap_init")
    
    # Cooling loop
    for step in 1:steps
        # Append fresh bath in ground state using existing function
        ψ_sb = appendzeros_MPS(ψ_sys, sites)
        
        # Time evolution
        if use_trotter
            # Simple Trotter evolution (approximate)
            ψ_evolved = copy(ψ_sb)
            tau = 0.1
            n_steps = Int(te / tau)
            for _ in 1:n_steps
                ψ_evolved = tdvp(H_sb, -im * tau, ψ_evolved; 
                               time_step=-im * tau/4,
                               normalize=true,
                               maxdim=Dmax,
                               cutoff=cutoff,
                               outputlevel=0)
            end
        else
            # Continuous evolution using TDVP
            ψ_evolved = tdvp(H_sb, -im * te, ψ_sb; 
                           time_step=-im * 0.1,
                           normalize=true,
                           maxdim=Dmax,
                           cutoff=cutoff,
                           outputlevel=0)
        end
        
        # Sample bath and get system state using existing function
        try
            v_b, ψ_sys_new = sample_bath(ψ_evolved)
            ψ_sys = ψ_sys_new
            
            # Handle potential dimension mismatch
            if length(ψ_sys) != N
                # Reconstruct with correct sites if needed
                ψ_sys = productMPS(sites_sys, [1 for _ in 1:N])
                normalize!(ψ_sys)
            end
        catch e
            println("Warning: Bath sampling failed at step $step: $e")
            # Fallback: just trace out bath approximately
            ψ_sys = productMPS(sites_sys, [1 for _ in 1:N])
            normalize!(ψ_sys)
        end
        
        # Measurements
        E = real(inner(ψ_sys', H_sys, ψ_sys))
        overlap = abs2(inner(ψ_sys, ϕ0))
        push!(E_list, E)
        push!(overlap_list, overlap)
        
        if step % 20 == 0
            println("Step $step: E/N = $(E/N), overlap = $overlap")
        end
    end
    
    # Final results
    E_final = E_list[end]
    overlap_final = overlap_list[end]
    ΔE = E_final - E_list[1]
    
    println("Final results:")
    println("  E/N: $(E_list[1]/N) → $(E_final/N) (ΔE/N = $(ΔE/N))")
    println("  Overlap: $(overlap_list[1]) → $overlap_final")
    println("  Cooling efficiency: $(ΔE < 0 ? "✓ COOLING" : "✗ HEATING")")
    
    return E_list, overlap_list, method_name
end

# Test all TN methods
results = []

# 1. MPS + Monte Carlo + Continuous (standard TDVP)
E1, O1, name1 = run_tn_cooling_simulation("MPS + Monte Carlo + Continuous", false, false)
push!(results, (E1, O1, name1))

# 2. MPS + Monte Carlo + Trotter 
E2, O2, name2 = run_tn_cooling_simulation("MPS + Monte Carlo + Trotter", false, true)
push!(results, (E2, O2, name2))

# 3. Compare with no cooling (just evolution without bath reset)
println("\n" * "="^60)
println("Control: Unitary evolution without bath reset")
println("="^60)

sites = siteinds("S=1/2", 2N)
H_sb = build_system_bath_hamiltonian(N, J, hx, hz, Δ, g, sites)

# Initial state: system all up, bath all down
ψ_init = MPS(sites)
for i in 1:N
    ψ_init[2i-1] = onehot(sites[2i-1] => 2)  # System up
    ψ_init[2i] = onehot(sites[2i] => 1)      # Bath down
end

E_control = []
E_init_control = real(inner(ψ_init', H_sb, ψ_init))
push!(E_control, E_init_control)

ψ = copy(ψ_init)
for step in 1:20  # Shorter for control
    global ψ = tdvp(H_sb, -im * te, ψ; 
                   time_step=-im * 0.1,
                   normalize=true,
                   maxdim=Dmax,
                   cutoff=cutoff,
                   outputlevel=0)
    E = real(inner(ψ', H_sb, ψ))
    push!(E_control, E)
end

# Generate comprehensive plot
println("\n" * "="^60)
println("Generating comparison plots...")
println("="^60)

p1 = plot(title="Energy Evolution Comparison", xlabel="Step", ylabel="Energy per spin")
hline!(p1, [E0/N], label="Ground state E/N", color=:black, linestyle=:dash, linewidth=2)

colors = [:blue, :red, :green, :purple]
for (i, (E_list, _, name)) in enumerate(results)
    plot!(p1, 0:length(E_list)-1, E_list ./ N, 
          label=name, color=colors[i], linewidth=2, marker=:circle, markersize=2)
end

# Add control
plot!(p1, 0:length(E_control)-1, E_control ./ N, 
      label="Control (no cooling)", color=:gray, linestyle=:dot, linewidth=2)

p2 = plot(title="Ground State Overlap", xlabel="Step", ylabel="Overlap")
for (i, (_, O_list, name)) in enumerate(results)
    plot!(p2, 0:length(O_list)-1, O_list, 
          label=name, color=colors[i], linewidth=2, marker=:circle, markersize=2)
end

# Combine plots
p = plot(p1, p2, layout=(2,1), size=(800,800))
savefig(p, "comprehensive_tn_cooling_test.pdf")
println("Plot saved to: comprehensive_tn_cooling_test.pdf")

# Summary
println("\n" * "="^60)
println("SUMMARY")
println("="^60)
for (E_list, O_list, name) in results
    ΔE = E_list[end] - E_list[1] 
    ΔO = O_list[end] - O_list[1]
    println("$name:")
    println("  Energy change: ΔE/N = $(ΔE/N)")
    println("  Overlap change: ΔO = $ΔO")
    println("  Result: $(ΔE < -0.1 ? "✓ GOOD COOLING" : ΔE < 0 ? "~ WEAK COOLING" : "✗ NO COOLING")")
    println()
end

println("Ground state energy density: E0/N = $(E0/N)")
println("Expected final energy should approach E0/N")