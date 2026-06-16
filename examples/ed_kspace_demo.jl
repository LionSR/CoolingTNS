#!/usr/bin/env julia

"""
Demonstration of k-space measurements for ED simulations with periodic/antiperiodic BC.

This example shows:
1. How to run ED simulations with different boundary conditions
2. How k-space measurements are automatically performed
3. How to interpret the momentum distribution plots
"""

using CoolingTNS
using Printf

println("="^60)
println("K-Space Measurement Demo for ED Backend")
println("="^60)

# Parameters for the demonstration
N = 6  # Small system size for ED
J = 1.0
h = 2.0  # Transverse field
coupling = "XX"
g = 0.3
te = 2.0
steps = 30

println("\nSystem parameters:")
println("- N = $N spins")
println("- Hamiltonian: Transverse field Ising (J=$J, h=$h)")
println("- Coupling: $coupling with strength g=$g")
println("- Evolution time per step: te=$te")
println("- Total cooling steps: $steps")

println("\nNOTE: K-space measurements are only available for the integrable Ising model,")
println("      not for the non-integrable (niIsing) model.")

# Test 1: Periodic BC
println("\n" * "="^40)
println("Test 1: Periodic Boundary Conditions")
println("="^40)

cmd_pbc = `julia Cooling.jl --N $N --problem Ising --backend ED --bc periodic --sim_method monte_carlo --evolution_method continuous --coupling $coupling --g $g --te $te --steps $steps --J $J --h $h`
println("\nRunning: $cmd_pbc")
run(cmd_pbc)

println("\n✓ Periodic BC simulation complete!")
println("  Check Results/Figs/ for momentum_dist_*.pdf and momentum_dist_heatmap_*.pdf")

# Test 2: Antiperiodic BC
println("\n" * "="^40)
println("Test 2: Antiperiodic Boundary Conditions")
println("="^40)

cmd_apbc = `julia Cooling.jl --N $N --problem Ising --backend ED --bc antiperiodic --sim_method monte_carlo --evolution_method continuous --coupling $coupling --g $g --te $te --steps $steps --J $J --h $h`
println("\nRunning: $cmd_apbc")
run(cmd_apbc)

println("\n✓ Antiperiodic BC simulation complete!")

# Test 3: Density Matrix method
println("\n" * "="^40)
println("Test 3: Density Matrix with Periodic BC")
println("="^40)

cmd_dm = `julia Cooling.jl --N $N --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling $coupling --g $g --te $te --steps $steps --J $J --h $h`
println("\nRunning: $cmd_dm")
run(cmd_dm)

println("\n✓ Density matrix simulation complete!")

# Explanation
println("\n" * "="^60)
println("Understanding the K-Space Plots")
println("="^60)

println("""
The generated plots show:

1. **momentum_dist_*.pdf**: 
   - Shows n_k (occupation number) vs k (momentum) at different cooling steps
   - Initial state (usually uniform or specific pattern based on product state)
   - Evolution shows population transfer to lower energy modes
   - Resonant momenta are marked by solving ε_k ≈ |δ|

2. **momentum_dist_heatmap_*.pdf**:
   - 2D heatmap showing n_k vs (k, cooling step)
   - Visualizes the full evolution of all momentum modes
   - Darker regions indicate higher occupation
   - Shows which modes are being cooled most effectively

Key physics:
- For the Ising model, the positive code-unit quasiparticle energies are
  ε_k = 2 sqrt(J^2 + h^2) sqrt(1 - sin(2θ) cos(2πk/N)), with θ = atan(h, J)
- The bath frequency δ is an energy; resonant momenta are modes where ε_k ≈ |δ|
- Different boundary conditions (PBC vs APBC) have different allowed k values
- PBC: k ∈ {-N/2+1, ..., N/2}
- APBC: k ∈ {-(N-1)/2, ..., (N-1)/2}
""")

println("\nAll plots saved in Results/Figs/")
println("Demo complete! 🎉")
