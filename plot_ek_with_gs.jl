using CoolingTNS

# Parameters
N = 10  # Larger N for smoother curves
J = 1.0
h = 2.0
bc = :periodic
g = 0.3
delta = g * h  # Resonant frequency

# Plot energy dispersion with ground state occupation
println("Generating energy dispersion plot with ground state occupation...")
CoolingTNS.plot_dispersion_with_ground_state(N, J, h, bc; delta=delta, save_fig=true, 
                                            filename="IsingN$(N)J$(J)h$(h)bc$(bc)")

# Also try with different parameters to see the variation
println("\nGenerating plot with J = -1.0 (ferromagnetic)...")
CoolingTNS.plot_dispersion_with_ground_state(N, -1.0, h, bc; delta=delta, save_fig=true, 
                                            filename="IsingN$(N)J-1.0h$(h)bc$(bc)")

println("\nGenerating plot with smaller h = 0.5...")
CoolingTNS.plot_dispersion_with_ground_state(N, J, 0.5, bc; delta=g*0.5, save_fig=true, 
                                            filename="IsingN$(N)J$(J)h0.5bc$(bc)")