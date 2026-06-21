using CoolingTNS

const ED_DM_EXAMPLE_ROOT = normpath(joinpath(@__DIR__, ".."))
const ED_DM_COOLING_DRIVER = joinpath(ED_DM_EXAMPLE_ROOT, "Cooling.jl")
const ED_DM_PLOTTING_DIR = joinpath(ED_DM_EXAMPLE_ROOT, "scripts", "plotting")
const ED_DM_PLOTTERS_LOADED = Ref(false)

_problem_cli(::CoolingTNS.HamiltonianParameters{CoolingTNS.IsingModel}) = "Ising"
_backend_cli(::CoolingTNS.EDBackend) = "ED"
_sim_method_cli(::CoolingTNS.DensityMatrix) = "density_matrix"
_evolution_method_cli(::CoolingTNS.ContinuousEvolution) = "continuous"

"""
    ed_dm_ising_example(; kwargs...) -> NamedTuple

Construct the typed parameters for the ED density-matrix Ising k-space example
and derive the expected HDF5 path with `CoolingTNS.create_filename`.
"""
function ed_dm_ising_example(;
    N::Int=6,
    J::Real=1.0,
    h::Real=2.0,
    coupling::AbstractString="XX",
    g::Real=0.3,
    te::Real=2.0,
    steps::Int=20,
    bc::Symbol=:periodic,
    measure_modes::Bool=true,
)
    backend = CoolingTNS.EDBackend()
    ham_params = CoolingTNS.IsingParameters(N, Float64(J), Float64(h); bc=bc)
    coupling_params = CoolingTNS.BasicCouplingParameters(
        String(coupling),
        Float64(g),
        steps,
        Float64(te),
        nothing,
    )
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.DensityMatrix(),
        CoolingTNS.ContinuousEvolution(),
    )
    filename = CoolingTNS.create_filename(ham_params, coupling_params, sim_params, backend)

    return (
        N=N,
        J=Float64(J),
        h=Float64(h),
        coupling=String(coupling),
        g=Float64(g),
        te=Float64(te),
        steps=steps,
        bc=bc,
        backend=backend,
        ham_params=ham_params,
        coupling_params=coupling_params,
        sim_params=sim_params,
        measure_modes=measure_modes,
        filename=filename,
        result_path=joinpath(ED_DM_EXAMPLE_ROOT, "Results", filename * ".h5"),
    )
end

"""
    ed_dm_driver_command(params) -> Cmd

Build the `Cooling.jl` command corresponding to the typed parameters in
`params`, including `--measure_modes` when the example will plot mode
observables and positive gap labels.
"""
function ed_dm_driver_command(params)
    cmd = `$(Base.julia_cmd()) --project=$(ED_DM_EXAMPLE_ROOT) $(ED_DM_COOLING_DRIVER) --N $(params.N) --problem $(_problem_cli(params.ham_params)) --backend $(_backend_cli(params.backend)) --bc $(String(params.bc)) --sim_method $(_sim_method_cli(params.sim_params.sim_method)) --evolution_method $(_evolution_method_cli(params.sim_params.evolution_method)) --coupling $(params.coupling) --g $(params.g) --te $(params.te) --steps $(params.steps) --J $(params.J) --h $(params.h)`
    return params.measure_modes ? `$cmd --measure_modes` : cmd
end

function run_ed_dm_ising_driver(params)
    run(Cmd(ed_dm_driver_command(params); dir=ED_DM_EXAMPLE_ROOT))
end

function load_ed_dm_kspace_plotters!()
    ED_DM_PLOTTERS_LOADED[] && return nothing
    Base.include(@__MODULE__, joinpath(ED_DM_PLOTTING_DIR, "plot_nk_evolution.jl"))
    Base.include(@__MODULE__, joinpath(ED_DM_PLOTTING_DIR, "plot_ek_evolution.jl"))
    ED_DM_PLOTTERS_LOADED[] = true
    return nothing
end

"""
    plot_ed_dm_kspace_results(result_path; steps_to_plot=nothing)

Use the shared plotting scripts to emit the Fourier occupation plot and, when
the file contains `h_k` mode data, the Bogoliubov mode-energy contribution
plot.
"""
function plot_ed_dm_kspace_results(result_path::AbstractString; steps_to_plot=nothing)
    load_ed_dm_kspace_plotters!()

    nk_plot = Base.invokelatest(
        getfield(@__MODULE__, :plot_nk_evolution),
        result_path;
        steps_to_plot=steps_to_plot,
        save_fig=true,
    )
    mode_energy_plot = Base.invokelatest(
        getfield(@__MODULE__, :plot_ek_evolution),
        result_path;
        steps_to_plot=steps_to_plot,
        save_fig=true,
    )

    return (nk_plot=nk_plot, mode_energy_plot=mode_energy_plot)
end
