using CoolingTNS

include(joinpath(@__DIR__, "ed_dm_example_utils.jl"))

params = ed_dm_ising_example()

println("Running density matrix simulation with N=$(params.N)...")
println("Parameters: J=$(params.J), h=$(params.h), g=$(params.g), te=$(params.te), steps=$(params.steps), bc=$(params.bc)")
run_ed_dm_ising_driver(params)

if isfile(params.result_path)
    println("\nGenerating canonical k-space plots from $(params.result_path)...")
    plot_ed_dm_kspace_results(params.result_path)
else
    println("Error: DM file not found at $(params.result_path)")
end
