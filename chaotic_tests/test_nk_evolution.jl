using CoolingTNS

# Parameters
N = 8
J = 1.0
h = 2.0
g = 0.3
te = 2.0
steps = 30

println("Running ED simulations with density matrix method...")

# Run with periodic BC
println("\n1. Running with periodic BC...")
run(`julia --project=. Cooling.jl --N $N --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling XX --g $g --te $te --steps $steps --J $J --h $h`)

# Run with anti-periodic BC
println("\n2. Running with anti-periodic BC...")
run(`julia --project=. Cooling.jl --N $N --problem Ising --backend ED --bc antiperiodic --sim_method density_matrix --evolution_method continuous --coupling XX --g $g --te $te --steps $steps --J $J --h $h`)

# Plot n_k evolution for both
println("\nGenerating n_k evolution plots...")

# Find the generated files
pbc_file = "Results/Cooling_HamIsingJ$(J)h$(h)bcperiodic_CouplingXXg$(g)te$(te)steps$(steps)_SimEDDM.h5"
apbc_file = "Results/Cooling_HamIsingJ$(J)h$(h)bcantiperiodic_CouplingXXg$(g)te$(te)steps$(steps)_SimEDDM.h5"

if isfile(pbc_file)
    println("Plotting PBC results...")
    CoolingTNS.plot_nk_evolution(pbc_file; save_fig=true)
else
    println("PBC file not found: $pbc_file")
end

if isfile(apbc_file)
    println("Plotting APBC results...")
    CoolingTNS.plot_nk_evolution(apbc_file; save_fig=true)
else
    println("APBC file not found: $apbc_file")
end

println("\nDone!")