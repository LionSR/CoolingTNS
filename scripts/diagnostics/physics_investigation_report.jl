"""
    physics_investigation_report.jl

Physics investigation and validation script for CoolingTNS.

This script focuses on two recurring questions:

1. Tensor-network backend (TN) Monte Carlo + Continuous evolution sometimes shows an
   *increase* of the system energy over a few cooling cycles.

2. Exact diagonalization backend (ED) cooling can appear ``very slow''.

The core protocol is the repeated application of the cooling map

    ρ ↦ Tr_B[ U (ρ ⊗ ρ_B) U† ]

where U = exp(-i H_SB t) and the bath is re-prepared in its ground state ρ_B at the
beginning of each cycle.

This file is written as a runnable report:

    julia --project=. scripts/diagnostics/physics_investigation_report.jl

Useful short exact-channel control:

    julia --project=. scripts/diagnostics/physics_investigation_report.jl \
        --only ed-scan --ed-scan-N 2 --ed-scan-steps 3

Command-line options preserve the historical defaults unless overridden.

"""

using CoolingTNS
using LinearAlgebra
using Random
using Statistics
using Printf
using Test

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

"""Run `f()` with stdout silenced."""
with_silenced_stdout(f::Function) = redirect_stdout(f, devnull)

function pretty_header(title::AbstractString)
    println("\n" * "="^80)
    println(title)
    println("="^80)
end

function print_energy_table(; N::Int, E_list::AbstractVector, ov_list::AbstractVector, purity_list=nothing)
    if purity_list === nothing
        println("step\tE/N\t\tGS overlap")
        for s in eachindex(E_list)
            @printf("%3d\t% .9f\t%.9f\n", s - 1, E_list[s] / N, ov_list[s])
        end
    else
        println("step\tE/N\t\tGS overlap\tpurity")
        for s in eachindex(E_list)
            @printf("%3d\t% .9f\t%.9f\t%.9f\n", s - 1, E_list[s] / N, ov_list[s], purity_list[s])
        end
    end
end

function system_spectrum_gap_ed(ham_params)
    N = ham_params.N
    H = Matrix(CoolingTNS.construct_system_hamiltonian(ham_params, CoolingTNS.EDBackend(), N))
    vals = sort(eigvals(Symmetric(H)))
    return vals, vals[2] - vals[1]
end

const PHYSICS_REPORT_SECTIONS = (:small_ed, :ed_trotter, :mc, :ed_scan)
const PHYSICS_REPORT_SECTION_NAMES = Dict(
    "small-ed" => :small_ed,
    "ed-trotter" => :ed_trotter,
    "mc" => :mc,
    "ed-scan" => :ed_scan,
)

"""
    PhysicsInvestigationReportConfig(; sections, n_traj, ed_scan_N, ed_scan_steps)

Configuration for the runnable physics-investigation report.  The defaults
recover the historical no-argument report.
"""
Base.@kwdef struct PhysicsInvestigationReportConfig{S<:Tuple{Vararg{Symbol}}}
    sections::S = PHYSICS_REPORT_SECTIONS
    n_traj::Int = 200
    ed_scan_N::Int = 4
    ed_scan_steps::Int = 50
end

const DEFAULT_PHYSICS_INVESTIGATION_REPORT_CONFIG = PhysicsInvestigationReportConfig()

function _physics_report_arg_value(args, i, flag)
    i < length(args) || throw(ArgumentError("$flag requires a value"))
    value = args[i + 1]
    startswith(value, "--") && throw(ArgumentError("$flag requires a value"))
    return value
end

function _parse_physics_report_sections(value::AbstractString)
    normalized = strip(value)
    isempty(normalized) && throw(ArgumentError("--only requires at least one section"))
    normalized == "all" && return PHYSICS_REPORT_SECTIONS

    sections = Symbol[]
    for raw_name in split(normalized, ",")
        name = strip(raw_name)
        haskey(PHYSICS_REPORT_SECTION_NAMES, name) ||
            throw(ArgumentError("unknown report section '$name'"))
        section = PHYSICS_REPORT_SECTION_NAMES[name]
        section in sections || push!(sections, section)
    end
    return Tuple(sections)
end

function _validate_physics_report_config(config::PhysicsInvestigationReportConfig)
    isempty(config.sections) && throw(ArgumentError("at least one report section must be selected"))
    all(section -> section in PHYSICS_REPORT_SECTIONS, config.sections) ||
        throw(ArgumentError("unknown report section in configuration"))
    config.n_traj > 0 || throw(ArgumentError("--traj must be positive"))
    config.ed_scan_N >= 2 || throw(ArgumentError("--ed-scan-N must be at least 2"))
    config.ed_scan_steps > 0 || throw(ArgumentError("--ed-scan-steps must be positive"))
    return config
end

function print_physics_report_usage(io=stdout)
    println(io, "Usage: julia --project=. scripts/diagnostics/physics_investigation_report.jl [options]")
    println(io)
    println(io, "Options:")
    println(io, "  --only LIST           comma-separated sections: small-ed,ed-trotter,mc,ed-scan, or all")
    println(io, "  --traj INT            Monte Carlo trajectories; below the default is a pilot sample")
    println(io, "  --ed-scan-N INT       system size for the ED cooling-rate scan")
    println(io, "  --ed-scan-steps INT   cooling cycles for the ED cooling-rate scan")
    println(io, "  --help                print this message")
    println(io)
    println(io, "Environment:")
    println(io, "  COOLINGTNS_TRAJ       default for --traj when the flag is omitted")
end

function parse_physics_investigation_report_args(args=ARGS; io=stdout)
    if any(arg -> arg == "--help" || arg == "-h", args)
        print_physics_report_usage(io)
        return nothing
    end

    config = Dict{Symbol,Any}(
        :sections => DEFAULT_PHYSICS_INVESTIGATION_REPORT_CONFIG.sections,
        :n_traj => DEFAULT_PHYSICS_INVESTIGATION_REPORT_CONFIG.n_traj,
        :ed_scan_N => DEFAULT_PHYSICS_INVESTIGATION_REPORT_CONFIG.ed_scan_N,
        :ed_scan_steps => DEFAULT_PHYSICS_INVESTIGATION_REPORT_CONFIG.ed_scan_steps,
    )

    i = 1
    traj_seen = false
    while i <= length(args)
        arg = args[i]
        if arg == "--only"
            config[:sections] = _parse_physics_report_sections(
                _physics_report_arg_value(args, i, arg),
            )
            i += 2
        elseif arg == "--traj"
            config[:n_traj] = parse(Int, _physics_report_arg_value(args, i, arg))
            traj_seen = true
            i += 2
        elseif arg == "--ed-scan-N"
            config[:ed_scan_N] = parse(Int, _physics_report_arg_value(args, i, arg))
            i += 2
        elseif arg == "--ed-scan-steps"
            config[:ed_scan_steps] = parse(Int, _physics_report_arg_value(args, i, arg))
            i += 2
        else
            throw(ArgumentError("unknown option: $arg"))
        end
    end

    if !traj_seen && (:mc in config[:sections])
        config[:n_traj] = parse(
            Int,
            get(ENV, "COOLINGTNS_TRAJ", string(DEFAULT_PHYSICS_INVESTIGATION_REPORT_CONFIG.n_traj)),
        )
    end

    return _validate_physics_report_config(PhysicsInvestigationReportConfig(;
        sections=config[:sections],
        n_traj=config[:n_traj],
        ed_scan_N=config[:ed_scan_N],
        ed_scan_steps=config[:ed_scan_steps],
    ))
end

_physics_report_section_enabled(config, section::Symbol) = section in config.sections
_physics_report_should_execute() = abspath(PROGRAM_FILE) == abspath(@__FILE__)

function run_ed_dm_case(; ham_params, coupling::String, g::Float64, te::Float64, steps::Int, delta=nothing)
    cp = CoolingTNS.BasicCouplingParameters(coupling, g, steps, te, delta)
    sim = CoolingTNS.UnifiedSimulationParameters(CoolingTNS.DensityMatrix(), CoolingTNS.ContinuousEvolution(); pe=0.0)
    prob = CoolingTNS.setup_problem(CoolingTNS.EDBackend(), ham_params, cp, sim)
    st0 = CoolingTNS.setup_initial_state(prob, sim, "product", 0.0)
    res = with_silenced_stdout() do
        CoolingTNS.run_cooling(prob, st0, prob.extra.coupling_params, sim, ham_params)
    end
    return prob, res
end

function run_small_ed_section()
    pretty_header("1) ED DM+Continuous: N=2 transverse-field Ising")

    @testset "ED DM+Continuous cools for N=2 Ising" begin
        N = 2
        ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.2, 10, 1.0, nothing)
        sim_params = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(), CoolingTNS.ContinuousEvolution(); pe=0.0
        )

        prob = CoolingTNS.setup_problem(CoolingTNS.EDBackend(), ham_params, coupling_params, sim_params)
        Δ = prob.extra.coupling_params.delta

        vals, gap_exact = system_spectrum_gap_ed(ham_params)

        @printf("System eigenvalues: %s\n", string(round.(vals, digits=6)))
        @printf("Exact gap           = %.12f\n", gap_exact)
        @printf("Auto Δ (setup)      = %.12f\n", Δ)
        @printf("Ground energy E0/N  = %.12f\n", prob.e₀ / N)

        @test isapprox(Δ, gap_exact; atol=1e-9, rtol=1e-9)

        state0 = CoolingTNS.setup_initial_state(prob, sim_params, "product", 0.0)
        results = with_silenced_stdout() do
            CoolingTNS.run_cooling(prob, state0, prob.extra.coupling_params, sim_params, ham_params)
        end

        E_list = results[CoolingTNS.RESULT_ENERGY]
        ov_list = results[CoolingTNS.RESULT_GROUND_STATE_OVERLAP]
        purity_list = results[CoolingTNS.RESULT_PURITY]

        print_energy_table(N=N, E_list=E_list, ov_list=ov_list, purity_list=purity_list)

        # Cooling is not necessarily monotone step-by-step in general, but here ED DM is.
        @test E_list[end] < E_list[1]
    end
end

function run_ed_trotter_section()
    pretty_header("2) ED DM: Continuous vs (time-sliced) 'Trotter' evolution")

    @testset "ED continuous vs trotter agreement (N=4 Ising)" begin
        N = 4
        ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.2, 5, 1.0, nothing)

        sim_cont = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(), CoolingTNS.ContinuousEvolution(); pe=0.0
        )
        sim_trot = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(), CoolingTNS.TrotterEvolution(); tau=0.1, pe=0.0
        )

        prob_cont = CoolingTNS.setup_problem(CoolingTNS.EDBackend(), ham_params, coupling_params, sim_cont)
        prob_trot = CoolingTNS.setup_problem(CoolingTNS.EDBackend(), ham_params, coupling_params, sim_trot)

        st0_cont = CoolingTNS.setup_initial_state(prob_cont, sim_cont, "product", 0.0)
        st0_trot = CoolingTNS.setup_initial_state(prob_trot, sim_trot, "product", 0.0)

        res_cont = with_silenced_stdout() do
            CoolingTNS.run_cooling(prob_cont, st0_cont, prob_cont.extra.coupling_params, sim_cont, ham_params)
        end
        res_trot = with_silenced_stdout() do
            CoolingTNS.run_cooling(prob_trot, st0_trot, prob_trot.extra.coupling_params, sim_trot, ham_params)
        end

        E_cont = res_cont[CoolingTNS.RESULT_ENERGY]
        E_trot = res_trot[CoolingTNS.RESULT_ENERGY]

        maxdiff = maximum(abs.(E_cont .- E_trot))

        @printf("Auto Δ = %.12f\n", prob_cont.extra.coupling_params.delta)
        @printf("E0/N   = %.12f\n", prob_cont.e₀ / N)
        @printf("E_start/N (cont,trot) = (%.9f, %.9f)\n", E_cont[1] / N, E_trot[1] / N)
        @printf("E_end/N   (cont,trot) = (%.9f, %.9f)\n", E_cont[end] / N, E_trot[end] / N)
        @printf("max |E_cont - E_trot| = %.3e\n", maxdiff)

        @test maxdiff < 1e-10
        @test E_cont[end] < E_cont[1]
    end
end

function run_mc_section(n_traj::Int)
    pretty_header("3) TN MC+Continuous energy increase: stochastic (Kraus) trajectories")

    @testset "Monte Carlo trajectories can heat even if average cools" begin
        Random.seed!(1234)

        N = 2
        ham_params = CoolingTNS.NiIsingParameters(N, 1.0, -1.05, 0.5)
        coupling_params = CoolingTNS.BasicCouplingParameters("XX", 0.1, 3, 1.0, nothing)

        # Deterministic reference: ED density matrix
        sim_ed_dm = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.DensityMatrix(), CoolingTNS.ContinuousEvolution(); pe=0.0
        )
        prob_ed_dm = CoolingTNS.setup_problem(CoolingTNS.EDBackend(), ham_params, coupling_params, sim_ed_dm)
        st0_ed_dm = CoolingTNS.setup_initial_state(prob_ed_dm, sim_ed_dm, "product", 0.0)
        res_ed_dm = with_silenced_stdout() do
            CoolingTNS.run_cooling(prob_ed_dm, st0_ed_dm, prob_ed_dm.extra.coupling_params, sim_ed_dm, ham_params)
        end

        # Stochastic unravellings: ED MC and TN MC
        sim_ed_mc = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(), CoolingTNS.ContinuousEvolution(); pe=0.0
        )
        prob_ed_mc = CoolingTNS.setup_problem(CoolingTNS.EDBackend(), ham_params, coupling_params, sim_ed_mc)

        sim_tn_mc = CoolingTNS.UnifiedSimulationParameters(
            CoolingTNS.MonteCarloWavefunction(), CoolingTNS.ContinuousEvolution();
            Dmax=40, cutoff=1e-10, tau=0.1, pe=0.0
        )
        prob_tn_mc = CoolingTNS.setup_problem(CoolingTNS.TNBackend(), ham_params, coupling_params, sim_tn_mc)

        function sample_final_energies(prob, sim_params; backend_label="")
            E_end = zeros(Float64, n_traj)
            E_start = nothing
            for t in 1:n_traj
                st0 = CoolingTNS.setup_initial_state(prob, sim_params, "product", 0.0)
                res = with_silenced_stdout() do
                    CoolingTNS.run_cooling(prob, st0, prob.extra.coupling_params, sim_params, ham_params)
                end
                if E_start === nothing
                    E_start = res[CoolingTNS.RESULT_ENERGY][1]
                end
                E_end[t] = res[CoolingTNS.RESULT_ENERGY][end]
            end
            return E_start::Float64, E_end
        end

        E_start_ed, E_end_ed = sample_final_energies(prob_ed_mc, sim_ed_mc; backend_label="ED")
        E_start_tn, E_end_tn = sample_final_energies(prob_tn_mc, sim_tn_mc; backend_label="TN")

        # Basic sanity: same initial energy
        @test isapprox(E_start_ed, E_start_tn; atol=1e-10, rtol=1e-10)

        println("Reference ED DM energy trajectory (deterministic):")
        print_energy_table(
            N=N,
            E_list=res_ed_dm[CoolingTNS.RESULT_ENERGY],
            ov_list=res_ed_dm[CoolingTNS.RESULT_GROUND_STATE_OVERLAP],
            purity_list=res_ed_dm[CoolingTNS.RESULT_PURITY],
        )

        function summarize(name, E_start, E_end)
            μ = mean(E_end)
            σμ = std(E_end) / sqrt(length(E_end))
            frac_heat = mean(E_end .> E_start)
            qs = quantile(E_end, [0.0, 0.1, 0.5, 0.9, 1.0])
            @printf("\n%s (n=%d trajectories)\n", name, length(E_end))
            @printf("  E_start/N          = %.9f\n", E_start / N)
            @printf("  mean(E_end)/N      = %.9f ± %.9f\n", μ / N, σμ / N)
            @printf("  frac(E_end>E_start)= %.3f\n", frac_heat)
            println("  quantiles(E_end/N)  = ", round.(qs ./ N, digits=6))

            return frac_heat
        end

        frac_heat_ed = summarize("ED MC", E_start_ed, E_end_ed)
        frac_heat_tn = summarize("TN MC", E_start_tn, E_end_tn)

        @test length(E_end_ed) == n_traj
        @test length(E_end_tn) == n_traj
        @test all(isfinite, E_end_ed)
        @test all(isfinite, E_end_tn)

        if n_traj >= DEFAULT_PHYSICS_INVESTIGATION_REPORT_CONFIG.n_traj
            # Key diagnostic:
            # - The Monte Carlo energy is a random variable; a single trajectory can heat.
            # - At the default sample size there should be both heating and cooling outcomes.
            @test any(E_end_ed .> E_start_ed)
            @test any(E_end_ed .< E_start_ed)
            @test any(E_end_tn .> E_start_tn)
            @test any(E_end_tn .< E_start_tn)

            # In this parameter regime, heating outcomes dominate
            # (rare, strong cooling events drive the mean).
            @test frac_heat_ed > 0.5
            @test frac_heat_tn > 0.5
        else
            println(
                "\nPilot trajectory count: skipping rare-outcome assertions. " *
                "Use --traj $(DEFAULT_PHYSICS_INVESTIGATION_REPORT_CONFIG.n_traj) " *
                "for the full stochastic diagnostic.",
            )
        end
    end
end

function run_ed_scan_section(; N::Int, steps::Int)
    pretty_header("4) ED cooling rate sensitivity (parameter scan)")

    @testset "ED cooling rate improves with stronger coupling / longer cycle time" begin
        ham_params = CoolingTNS.IsingParameters(N, 1.0, 1.0)

        println("g\tte\tΔ\t\tE_start/N\tE_end/N\t\tΔE/N")
        for g in (0.05, 0.10, 0.20, 0.40)
            prob, res = run_ed_dm_case(ham_params=ham_params, coupling="XX", g=g, te=1.0, steps=steps)
            E0 = res[CoolingTNS.RESULT_ENERGY][1]
            E1 = res[CoolingTNS.RESULT_ENERGY][end]
            @printf("%.2f\t%.1f\t%.3f\t% .6f\t% .6f\t%+.6f\n",
                    g, 1.0, prob.extra.coupling_params.delta, E0 / N, E1 / N, (E1 - E0) / N)
        end

        println("\nScan te at fixed g=0.2")
        for te in (0.2, 0.5, 1.0, 2.0)
            prob, res = run_ed_dm_case(ham_params=ham_params, coupling="XX", g=0.2, te=te, steps=steps)
            E0 = res[CoolingTNS.RESULT_ENERGY][1]
            E1 = res[CoolingTNS.RESULT_ENERGY][end]
            @printf("%.2f\t%.1f\t%.3f\t% .6f\t% .6f\t%+.6f\n",
                    0.2, te, prob.extra.coupling_params.delta, E0 / N, E1 / N, (E1 - E0) / N)
        end

        println("\nCoupling type comparison (fixed g=0.2, te=1.0)")
        println("coupling\tΔ\t\tE_end/N")
        results_by_coupling = Dict{String,Float64}()
        for coupling in ("XX", "ZZ", "YY")
            prob, res = run_ed_dm_case(ham_params=ham_params, coupling=coupling, g=0.2, te=1.0, steps=steps)
            E1 = res[CoolingTNS.RESULT_ENERGY][end]
            results_by_coupling[coupling] = E1
            @printf("%s\t\t%.3f\t% .6f\n", coupling, prob.extra.coupling_params.delta, E1 / N)
        end

        # Basic ordering check: larger g should cool at least as much as smaller g in this scan.
        prob_lo, res_lo = run_ed_dm_case(ham_params=ham_params, coupling="XX", g=0.05, te=1.0, steps=steps)
        prob_hi, res_hi = run_ed_dm_case(ham_params=ham_params, coupling="XX", g=0.40, te=1.0, steps=steps)
        @test res_hi[CoolingTNS.RESULT_ENERGY][end] < res_lo[CoolingTNS.RESULT_ENERGY][end]

        # Coupling dependence (just check that different couplings change the result)
        @test length(unique(values(results_by_coupling))) > 1
    end
end

function print_physics_report_summary(config::PhysicsInvestigationReportConfig)
    pretty_header("Summary")
    if _physics_report_section_enabled(config, :small_ed)
        println("- ED DM simulations are deterministic and show energy decrease for small Ising systems.")
        println("- Auto Δ is currently the system gap (E1-E0), verified against exact spectra for ED.")
    end
    if _physics_report_section_enabled(config, :ed_trotter)
        println("- ED 'Trotter' (time slicing) matches ED continuous evolution to numerical precision.")
    end
    if _physics_report_section_enabled(config, :mc)
        println("- Monte Carlo trajectories (ED MC and TN MC) are stochastic Kraus trajectories: a single trajectory can heat,")
        println("  and typical trajectories can heat even when the average map cools. Tests should therefore avoid requiring")
        println("  monotone cooling for a single Monte Carlo trajectory.")
    end
    if _physics_report_section_enabled(config, :ed_scan)
        println("- Cooling rate is strongly sensitive to parameters (g, te, coupling type). Increasing g or te can significantly")
        println("  accelerate energy decrease, consistent with weak-coupling / resonance expectations.")
    end
end

"""
    run_physics_investigation_report(config)

Run the selected diagnostic sections and print a summary of the sections that
were actually executed.
"""
function run_physics_investigation_report(config::PhysicsInvestigationReportConfig)
    if _physics_report_section_enabled(config, :small_ed)
        run_small_ed_section()
    end
    if _physics_report_section_enabled(config, :ed_trotter)
        run_ed_trotter_section()
    end
    if _physics_report_section_enabled(config, :mc)
        run_mc_section(config.n_traj)
    end
    if _physics_report_section_enabled(config, :ed_scan)
        run_ed_scan_section(N=config.ed_scan_N, steps=config.ed_scan_steps)
    end
    print_physics_report_summary(config)
    return nothing
end

if _physics_report_should_execute()
    report_config = parse_physics_investigation_report_args()
    report_config === nothing ? exit() : run_physics_investigation_report(report_config)
end
