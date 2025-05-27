using CoolingTNS

# Re-run the existing MC simulation to get updated k_values
println("Running quick ED simulation with Monte Carlo...")
run(`julia --project=. Cooling.jl --N 6 --problem Ising --backend ED --bc periodic --sim_method monte_carlo --evolution_method continuous --coupling XX --g 0.3 --te 2.0 --steps 5 --J 1.0 --h 2.0`)