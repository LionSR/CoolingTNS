using CoolingTNS

# Parameters
N = 8
J = 1.0
h = 2.0
g = 0.3
te = 2.0
steps = 20

println("Running ED simulation with periodic BC...")
run(`julia --project=. Cooling.jl --N $N --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling XX --g $g --te $te --steps $steps --J $J --h $h`)

# Plot e_k evolution
println("\nGenerating e_k evolution plot...")
pbc_file = "Results/Cooling_HamIsingJ$(J)h$(h)bcperiodic_CouplingXXg$(g)te$(te)steps$(steps)_SimEDDM.h5"

if isfile(pbc_file)
    CoolingTNS.plot_ek_evolution(pbc_file; save_fig=true)
else
    println("File not found: $pbc_file")
end