using CoolingTNS

const ED_DM_EXAMPLE_ROOT = normpath(joinpath(@__DIR__, ".."))
const ED_DM_COOLING_DRIVER = joinpath(ED_DM_EXAMPLE_ROOT, "Cooling.jl")

function ed_dm_ising_example(;
    N::Int=6,
    J::Real=1.0,
    h::Real=2.0,
    coupling::AbstractString="XX",
    g::Real=0.3,
    te::Real=2.0,
    steps::Int=20,
    bc::Symbol=:periodic,
)
    backend = CoolingTNS.EDBackend()
    ham_params = CoolingTNS.IsingParameters(N, J, h, bc)
    coupling_params = CoolingTNS.BasicCouplingParameters(coupling, g, steps, te, nothing)
    sim_params = CoolingTNS.UnifiedSimulationParameters(
        CoolingTNS.DensityMatrix(),
        CoolingTNS.ContinuousEvolution(),
    )
    filename = CoolingTNS.create_filename(ham_params, coupling_params, sim_params, backend)

    return (
        N=N,
        J=J,
        h=h,
        coupling=coupling,
        g=g,
        te=te,
        steps=steps,
        bc=bc,
        filename=filename,
        result_path=joinpath(ED_DM_EXAMPLE_ROOT, "Results", filename * ".h5"),
    )
end

function run_ed_dm_ising_driver(params)
    run(
        `$(Base.julia_cmd()) --project=$(ED_DM_EXAMPLE_ROOT) $(ED_DM_COOLING_DRIVER) --N $(params.N) --problem Ising --backend ED --bc $(String(params.bc)) --sim_method density_matrix --evolution_method continuous --coupling $(params.coupling) --g $(params.g) --te $(params.te) --steps $(params.steps) --J $(params.J) --h $(params.h)`,
    )
end
