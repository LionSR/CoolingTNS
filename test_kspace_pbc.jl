#!/usr/bin/env julia

"""
Quick test to demonstrate k-space measurements with periodic BC.
"""

using CoolingTNS

# Test with periodic BC (only works for Ising model)
println("Running ED simulation with periodic BC...")
run(`julia Cooling.jl --N 6 --problem Ising --backend ED --bc periodic --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.3 --te 2.0 --steps 50`)

println("\nCheck Results/Figs/ for the generated k-space plots!")