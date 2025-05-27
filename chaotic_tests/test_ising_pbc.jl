#!/usr/bin/env julia

"""
Test k-space measurements with Ising model (not niIsing).
"""

println("Testing k-space measurements with transverse field Ising model...")
println("Running with periodic BC:")

run(`julia Cooling.jl --N 6 --problem Ising --backend ED --bc periodic --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.3 --te 2.0 --steps 20 --J 1.0 --h 2.0`)

println("\nCheck Results/ and Results/Figs/ for output files!")