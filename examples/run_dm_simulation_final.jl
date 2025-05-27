using CoolingTNS

# Parameters
N = 6
J = 1.0
h = 2.0
g = 0.3
te = 2.0
steps = 20

println("Running density matrix simulation with complex Jordan-Wigner operators...")
println("Parameters: N=$N, J=$J, h=$h, g=$g, te=$te, steps=$steps")

# Run the simulation
run(`julia --project=. Cooling.jl --N $N --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling XX --g $g --te $te --steps $steps --J $J --h $h`)

# Check if the file was created
dm_file = "Results/Cooling_HamIsingJ$(J)h$(h)bcperiodic_CouplingXXg$(g)te$(te)steps$(steps)_SimEDDM.h5"

if isfile(dm_file)
    println("\nSimulation completed successfully!")
    println("Data saved to: $dm_file")
    
    # Generate plots
    include("plot_existing_dm.jl")
else
    println("\nError: Simulation file not found at $dm_file")
end