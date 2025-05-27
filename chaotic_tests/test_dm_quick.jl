using CoolingTNS

# Run a very short DM simulation just to test
println("Running quick DM test...")
run(`julia --project=. Cooling.jl --N 4 --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling XX --g 0.3 --te 0.5 --steps 5 --J 1.0 --h 2.0`)

# Check if file was created
dm_file = "Results/Cooling_HamIsingJ1.0h2.0bcperiodic_CouplingXXg0.3te0.5steps5_SimEDDM.h5"
println("\nFile created: ", isfile(dm_file))