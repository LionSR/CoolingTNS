include(joinpath(@__DIR__, "ed_dm_example_utils.jl"))

params = ed_dm_ising_example()

println("Running ED density-matrix simulation.")
println("N=$(params.N), J=$(params.J), h=$(params.h), g=$(params.g), te=$(params.te), steps=$(params.steps), bc=$(params.bc)")

run_ed_dm_ising_driver(params)

if isfile(params.result_path)
    println("\nSimulation completed.")
    println("Data saved to: $(params.result_path)")
    println("The cooling driver also writes the standard energy and k-space figures under Results/Figs.")
else
    println("\nError: simulation file not found at $(params.result_path)")
end
