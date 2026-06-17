using Test
using CoolingTNS

include(joinpath(@__DIR__, "..", "examples", "ed_dm_example_utils.jl"))

@testset "ED density-matrix k-space examples" begin
    params = ed_dm_ising_example()
    expected_filename = CoolingTNS.create_filename(
        params.ham_params,
        params.coupling_params,
        params.sim_params,
        params.backend,
    )

    @test params.filename == expected_filename
    @test params.result_path == joinpath(
        ED_DM_EXAMPLE_ROOT,
        "Results",
        expected_filename * ".h5",
    )
    @test params.measure_modes
    @test occursin("HamIsingN6bcperiodicJ1.0h2.0", params.filename)
    @test occursin("--bc periodic", string(ed_dm_driver_command(params)))
    @test occursin("--measure_modes", string(ed_dm_driver_command(params)))
    @test !occursin("--measure_modes", string(ed_dm_driver_command(ed_dm_ising_example(; measure_modes=false))))
    @test CoolingTNS.parse_commandline(["--measure_modes"])["measure_modes"]

    example_files = [
        "run_dm_and_plot.jl",
        "run_dm_simulation.jl",
        "run_dm_simulation_final.jl",
    ]
    obsolete_filename_prefix = "Cooling_" * "HamIsingJ"
    obsolete_plot_scripts = [
        "plot_actual_" * "cooling_evolution.jl",
        "plot_existing_" * "dm.jl",
        "plot_dm_" * "results.jl",
    ]
    obsolete_dispersion = "-2 * sqrt" * "(J^2 + h^2 + 2*J*h*cos(k))"
    obsolete_energy_label = "e_k = " * "ε_k n_k"

    for file in example_files
        text = read(joinpath(ED_DM_EXAMPLE_ROOT, "examples", file), String)
        @test !occursin(obsolete_filename_prefix, text)
        @test all(script -> !occursin(script, text), obsolete_plot_scripts)
        @test !occursin(obsolete_dispersion, text)
        @test !occursin(obsolete_energy_label, text)
    end
end
