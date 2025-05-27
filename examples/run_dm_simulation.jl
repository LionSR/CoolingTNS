using CoolingTNS

# Parameters
N = 6
J = 1.0
h = 2.0
g = 0.3
te = 2.0
steps = 20

println("Running ED simulation with density matrix method...")
run(`julia --project=. Cooling.jl --N $N --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling XX --g $g --te $te --steps $steps --J $J --h $h`)

# Find the generated file
dm_file = "Results/Cooling_HamIsingJ$(J)h$(h)bcperiodic_CouplingXXg$(g)te$(te)steps$(steps)_SimEDDM.h5"

if isfile(dm_file)
    println("\nSimulation completed. Generating plots...")
    
    # Generate n_k and e_k evolution plots
    include("plot_actual_cooling_evolution.jl")
else
    println("Error: DM simulation file not found at $dm_file")
end