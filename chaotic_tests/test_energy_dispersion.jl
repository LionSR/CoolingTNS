using CoolingTNS

# Parameters for the simulation
N = 6
J = 1.0
h = 2.0
bc = :periodic
g = 0.3
delta = g * h  # Resonant frequency

# Plot energy dispersion
println("Generating energy dispersion plot...")
CoolingTNS.plot_energy_dispersion(N, J, h, bc; delta=delta, save_fig=true, 
                                  filename="IsingN$(N)J$(J)h$(h)bc$(bc)")

# Run simulation with density matrix method
println("\nRunning ED simulation with density matrix method...")
run(`julia --project=. Cooling.jl --N $N --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling XX --g $g --te 2.0 --steps 20 --J $J --h $h`)