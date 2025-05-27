#!/usr/bin/env julia

"""
Test ED simulation speed with cached evolution operators.
"""

using CoolingTNS

println("="^60)
println("ED Backend Speed Test")
println("="^60)

# Test parameters
N = 8  # System size
steps = 30  # Number of cooling steps

println("\nTest parameters:")
println("- System size: N = $N")
println("- Cooling steps: $steps")
println("- Problem: Transverse field Ising with PBC")

# Run with timing
println("\nRunning ED simulation with cached evolution operators...")

start_time = time()

# Run the simulation
run(`julia Cooling.jl --N $N --problem Ising --backend ED --bc periodic --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.3 --te 2.0 --steps $steps --J 1.0 --h 2.0`)

end_time = time()
total_time = end_time - start_time

println("\n" * "="^60)
println("Timing Results:")
println("- Total time: $(round(total_time, digits=2)) seconds")
println("- Time per step: $(round(total_time/steps, digits=3)) seconds")
println("\nThe evolution operator exp(-iHt) is now cached,")
println("computed only once instead of $steps times!")
println("="^60)