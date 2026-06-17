using CoolingTNS

include(joinpath(@__DIR__, "ed_dm_example_utils.jl"))

println("Running ED simulation with density matrix method...")
params = ed_dm_ising_example()
run_ed_dm_ising_driver(params)

if isfile(params.result_path)
    println("\nSimulation completed. Generating plots...")
    plot_ed_dm_kspace_results(params.result_path)
else
    println("Error: DM simulation file not found at $(params.result_path)")
end
