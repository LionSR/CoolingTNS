using CoolingTNS

# Run a shorter simulation
println("Running short DM simulation...")
run(`julia --project=. Cooling.jl --N 6 --problem Ising --backend ED --bc periodic --sim_method density_matrix --evolution_method continuous --coupling XX --g 0.3 --te 1.0 --steps 10 --J 1.0 --h 2.0`)