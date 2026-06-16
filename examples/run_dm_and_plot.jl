include(joinpath(@__DIR__, "ed_dm_example_utils.jl"))

params = ed_dm_ising_example()

println("Running ED density-matrix simulation.")
println("N=$(params.N), J=$(params.J), h=$(params.h), g=$(params.g), te=$(params.te), steps=$(params.steps), bc=$(params.bc)")

run_ed_dm_ising_driver(params)

if isfile(params.result_path)
    println("\nGenerating n_k and e_k evolution plots from $(params.result_path).")

    include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_nk_evolution.jl"))
    include(joinpath(@__DIR__, "..", "scripts", "plotting", "plot_ek_evolution.jl"))

    plot_nk_evolution(params.result_path)
    plot_ek_evolution(params.result_path)
else
    println("Error: simulation file not found at $(params.result_path)")
end
