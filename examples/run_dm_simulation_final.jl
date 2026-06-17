using CoolingTNS

include(joinpath(@__DIR__, "ed_dm_example_utils.jl"))

println("Running density matrix simulation with complex Jordan-Wigner operators...")
params = ed_dm_ising_example()
println("Parameters: N=$(params.N), J=$(params.J), h=$(params.h), g=$(params.g), te=$(params.te), steps=$(params.steps)")

run_ed_dm_ising_driver(params)

if isfile(params.result_path)
    println("\nSimulation completed successfully!")
    println("Data saved to: $(params.result_path)")
    plot_ed_dm_kspace_results(params.result_path)
else
    println("\nError: Simulation file not found at $(params.result_path)")
end
